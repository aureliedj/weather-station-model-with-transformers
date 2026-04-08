"""
test.py

Test-set inference and evaluation for Station-MAE.

Loads a trained checkpoint, runs the model on the held-out test split, and
reports a comprehensive set of metrics in both normalised and physical units.

Usage
-----
    # Evaluate best checkpoint from a training run
    python test.py --data_root /path/to/peakweather --checkpoint checkpoints/run1/best.pt

    # Evaluate last checkpoint, custom window/delta
    python test.py --data_root /path/to/peakweather \\
                   --checkpoint checkpoints/run1/last.pt \\
                   --window 288 --max_delta 18 \\
                   --save_dir results/run1

    # Multi-checkpoint comparison (runs sequentially, saves individual CSVs)
    python test.py --data_root /path/to/peakweather \\
                   --checkpoint checkpoints/run1/best.pt checkpoints/run2/best.pt \\
                   --save_dir results/

Arguments
---------
  Data
    --data_root   STR   Path to PeakWeather data directory (required)
    --cache_dir   STR   Pre-built tensor cache (defaults to data_root)
    --window      INT   Input window steps (default 288 = 48 h at 10-min res)
    --max_delta   INT   Max lead-time steps (default 18 = 3 h)
    --batch_size  INT   Inference batch size (default 32)
    --num_workers INT   DataLoader workers (default 4)

  Model
    --checkpoint  STR   Path to .pt checkpoint file (required; can be repeated)
    --d_model     INT   Must match the saved model (default 128)
    --enc_heads   INT   (default 4)
    --enc_layers  INT   (default 4)
    --dec_heads   INT   (default 4)
    --dec_layers  INT   (default 2)
    --mlp_ratio   FLT   (default 4.0)
    --mask_ratio  FLT   Masking fraction used at eval (default 0.5)

  Output
    --save_dir    STR   Directory for metrics CSV and plots (default "test_results")
    --no_plots        Skip matplotlib plots

Metrics computed
----------------
    Per variable (temperature, pressure, humidity, wind_u, wind_v):
        RMSE, MAE, Bias (MBE), R²  — in both normalised and physical units

    Wind-derived:
        Wind speed RMSE / MAE / Bias  (m/s, from denormalised u+v)
        Wind direction MAE            (degrees, circular)

    Per lead-time (delta = 1..max_delta):
        Overall RMSE (normalised), per-variable RMSE (normalised)

    Persistence baseline:
        Uses the last input time-step as the naive forecast.
        Reports persistence RMSE per variable and skill score:
            skill = 1 - RMSE_model / RMSE_persistence   (higher = better)
"""

import argparse
import csv
import os
import sys

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station-MAE test-set evaluation")

    # Data
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--window",      type=int, default=288)
    p.add_argument("--max_delta",   type=int, default=18)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Model architecture — must match the saved checkpoint
    p.add_argument("--checkpoint",  type=str, nargs="+", required=True,
                   help="One or more .pt checkpoint paths to evaluate")
    p.add_argument("--d_model",     type=int, default=128)
    p.add_argument("--enc_heads",   type=int, default=4)
    p.add_argument("--enc_layers",  type=int, default=4)
    p.add_argument("--dec_heads",   type=int, default=4)
    p.add_argument("--dec_layers",  type=int, default=2)
    p.add_argument("--mlp_ratio",   type=float, default=4.0)
    p.add_argument("--mask_ratio",  type=float, default=0.5)
    p.add_argument("--device",      type=str, default=None)

    # Output
    p.add_argument("--save_dir",    type=str, default="test_results")
    p.add_argument("--no_plots",    action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_persistence_metrics(
    loader:    DataLoader,
    device:    torch.device,
    obs_stats: dict,
) -> dict[str, float]:
    """
    Compute metrics for the persistence baseline: predict the last observed
    value of the input window as the forecast at all lead-times.

    The baseline is applied only to masked stations (same evaluation protocol
    as the model) and only where sensor data is present at the target step.

    Returns dict with same key structure as evaluate_full() for easy comparison.
    """
    from model.embeddings import TARGET_VARIABLE_NAMES, NUM_TARGET_VARIABLES

    std_t  = obs_stats["std"].cpu()

    # Use records list (same approach as evaluate_full) so per-delta grouping
    # works correctly on the masked-station level.
    records: list[dict] = []

    for batch in loader:
        x           = batch["x"]            # (B, W, N, V)
        y_raw       = batch["y"]            # (B, N, V) or (B, K, N, V)
        y_mask_raw  = batch["y_mask"]
        delta_steps = batch["delta_steps"]  # (B,) or (B, K)

        # Handle multi-delta: take first delta only
        if y_raw.dim() == 4:
            y_raw       = y_raw[:, 0]
            y_mask_raw  = y_mask_raw[:, 0]
            delta_steps = delta_steps[:, 0]

        B, W, N, V = x.shape

        # Persistence: last input time-step → target (same for all lead-times)
        persist_pred  = x[:, -1, :, :NUM_TARGET_VARIABLES]       # (B, N, 5)
        y_target      = y_raw[:, :, :NUM_TARGET_VARIABLES]       # (B, N, 5)
        y_mask_target = y_mask_raw[:, :, :NUM_TARGET_VARIABLES]  # (B, N, 5)

        n_masked = max(1, int(round(N * 0.5)))   # same ratio as model (mask_ratio=0.5)
        for b in range(B):
            m_idx = torch.randperm(N)[:n_masked]
            records.append({
                "pred":   persist_pred[b, m_idx],       # (n_m, 5)
                "target": y_target[b, m_idx],
                "mask":   y_mask_target[b, m_idx],
                "delta":  int(delta_steps[b].item()),
            })

    preds_all   = torch.cat([r["pred"]   for r in records], dim=0)
    targets_all = torch.cat([r["target"] for r in records], dim=0)
    masks_all   = torch.cat([r["mask"]   for r in records], dim=0).bool()

    metrics: dict[str, float] = {}

    # Per-variable overall
    for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
        std_v = float(std_t[v].item())
        m = masks_all[:, v]
        if m.sum() == 0:
            metrics[f"persist_{var_name}_rmse_norm"] = float("nan")
            metrics[f"persist_{var_name}_rmse"]      = float("nan")
            continue
        err_norm = preds_all[m, v] - targets_all[m, v]
        metrics[f"persist_{var_name}_rmse_norm"] = float(
            err_norm.pow(2).mean().sqrt().item()
        )
        metrics[f"persist_{var_name}_rmse"] = float(
            (err_norm * std_v).pow(2).mean().sqrt().item()
        )

    # Overall (normalised)
    if masks_all.sum() > 0:
        err_all = preds_all[masks_all] - targets_all[masks_all]
        metrics["persist_overall_rmse_norm"] = float(
            err_all.pow(2).mean().sqrt().item()
        )

    # Per-delta (group records by delta)
    unique_deltas = sorted({r["delta"] for r in records})
    for d in unique_deltas:
        recs_d = [r for r in records if r["delta"] == d]
        p_d = torch.cat([r["pred"]   for r in recs_d], dim=0)
        t_d = torch.cat([r["target"] for r in recs_d], dim=0)
        m_d = torch.cat([r["mask"]   for r in recs_d], dim=0).bool()
        if m_d.sum() > 0:
            err = p_d[m_d] - t_d[m_d]
            metrics[f"persist_delta_{d:02d}_overall_rmse_norm"] = float(
                err.pow(2).mean().sqrt().item()
            )

    return metrics


# ---------------------------------------------------------------------------
# Skill score table
# ---------------------------------------------------------------------------

def compute_skill_scores(
    model_metrics: dict[str, float],
    persist_metrics: dict[str, float],
) -> dict[str, float]:
    """
    Skill score = 1 - RMSE_model / RMSE_persistence.

    A positive score means the model beats persistence.
    Score = 1.0 is a perfect forecast; score ≤ 0 means persistence is better.
    """
    from model.embeddings import TARGET_VARIABLE_NAMES

    skill: dict[str, float] = {}

    for var_name in TARGET_VARIABLE_NAMES:
        r_m = model_metrics.get(f"{var_name}_norm_rmse", float("nan"))
        # evaluate_full() writes "{var}_norm_rmse" for normalised metrics
        if r_m != r_m:   # NaN fallback — try alternate key format
            r_m = model_metrics.get(f"{var_name}_rmse_norm", float("nan"))
        r_p = persist_metrics.get(f"persist_{var_name}_rmse_norm", float("nan"))
        if r_p > 0:
            skill[f"skill_{var_name}"] = float(1.0 - r_m / r_p)
        else:
            skill[f"skill_{var_name}"] = float("nan")

    # Overall
    r_m_ov = model_metrics.get("overall_rmse_norm", float("nan"))
    r_p_ov = persist_metrics.get("persist_overall_rmse_norm", float("nan"))
    if r_p_ov > 0:
        skill["skill_overall"] = float(1.0 - r_m_ov / r_p_ov)

    # Per-delta
    delta_keys = sorted(
        k for k in model_metrics
        if k.startswith("delta_") and k.endswith("_overall_rmse_norm")
    )
    for dk in delta_keys:
        d     = int(dk.split("_")[1])
        r_m_d = model_metrics[dk]
        r_p_d = persist_metrics.get(f"persist_delta_{d:02d}_overall_rmse_norm",
                                    float("nan"))
        if r_p_d > 0:
            skill[f"skill_delta_{d:02d}"] = float(1.0 - r_m_d / r_p_d)

    return skill


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------

def save_metrics_csv(
    all_results: list[dict],   # list of {"label": str, "metrics": dict, "skill": dict}
    save_dir: str,
) -> str:
    """Save all runs into a single wide CSV and return the file path."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "test_metrics.csv")

    # Collect all unique metric keys across all runs
    all_keys: list[str] = []
    for r in all_results:
        for k in list(r["metrics"].keys()) + list(r["skill"].keys()):
            if k not in all_keys:
                all_keys.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint"] + all_keys)
        for r in all_results:
            row = [r["label"]]
            combined = {**r["metrics"], **r["skill"]}
            row += [f"{combined.get(k, float('nan')):.6f}" for k in all_keys]
            writer.writerow(row)

    return path


# ---------------------------------------------------------------------------
# Plot: per-delta RMSE curve
# ---------------------------------------------------------------------------

def plot_delta_rmse(
    all_results: list[dict],
    persist_metrics: dict,
    save_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    from model.embeddings import TARGET_VARIABLE_NAMES

    PALETTE = ["steelblue", "tomato", "seagreen", "darkorchid", "darkorange"]

    # Gather delta values
    delta_vals = sorted({
        int(k.split("_")[1])
        for r in all_results
        for k in r["metrics"]
        if k.startswith("delta_") and k.endswith("_overall_rmse_norm")
    })
    if not delta_vals:
        return

    mins = [d * 10 for d in delta_vals]

    # ── Overall RMSE vs lead-time ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, r in enumerate(all_results):
        rmse_d = [r["metrics"].get(f"delta_{d:02d}_overall_rmse_norm", float("nan"))
                  for d in delta_vals]
        ax.plot(mins, rmse_d, "o-", color=PALETTE[i % len(PALETTE)],
                label=r["label"], linewidth=1.5, markersize=5)

    # Persistence baseline
    persist_d = [persist_metrics.get(f"persist_delta_{d:02d}_overall_rmse_norm",
                                     float("nan"))
                 for d in delta_vals]
    if any(not np.isnan(x) for x in persist_d):
        ax.plot(mins, persist_d, "k--", lw=1.2, label="persistence", alpha=0.6)

    ax.set_xlabel("Lead-time (minutes)")
    ax.set_ylabel("Overall RMSE (normalised)")
    ax.set_title("Model RMSE vs lead-time (masked stations)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "delta_rmse_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # ── Per-variable RMSE vs lead-time (grid of subplots) ─────────────
    n_vars = len(TARGET_VARIABLE_NAMES)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.5 * n_vars, 3.5))

    for vi, var_name in enumerate(TARGET_VARIABLE_NAMES):
        ax = axes[vi]
        for i, r in enumerate(all_results):
            rmse_d = [r["metrics"].get(f"delta_{d:02d}_{var_name}_rmse_norm",
                                       float("nan"))
                      for d in delta_vals]
            ax.plot(mins, rmse_d, "o-", color=PALETTE[i % len(PALETTE)],
                    label=r["label"] if vi == 0 else None,
                    linewidth=1.3, markersize=4)
        ax.set_title(var_name, fontsize=9)
        ax.set_xlabel("Lead (min)")
        if vi == 0:
            ax.set_ylabel("RMSE (norm.)")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7,
               bbox_to_anchor=(1.0, 1.0))
    plt.suptitle("Per-variable RMSE vs lead-time", y=1.02, fontsize=10)
    plt.tight_layout()
    out2 = os.path.join(save_dir, "delta_rmse_per_variable.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # ── Device — priority: CUDA (cloud GPU) > MPS (Apple Silicon) > CPU ──
    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() and
                      torch.backends.mps.is_built() else
            "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cuda.enable_flash_sdp(True)

    print(f"\n[test.py]  device={device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset

    cache_dir = args.cache_dir or args.data_root

    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    # Build train_ds only to get obs_stats (do not iterate over it)
    print("Building train dataset for normalisation statistics …")
    train_ds = StationMAEDataset(
        ds, window_size=args.window, delta_steps=args.max_delta, split="train",
        num_delta_per_sample=1, max_delta_steps=args.max_delta,
        cache_dir=cache_dir,
    )
    obs_stats = train_ds.obs_stats
    print("  obs_stats ready (train split)")

    print("Building test dataset …")
    test_ds = StationMAEDataset(
        ds, window_size=args.window, delta_steps=args.max_delta, split="test",
        obs_stats=obs_stats,
        num_delta_per_sample=1,          # random delta in [1, max_delta] per sample
        max_delta_steps=args.max_delta,
        cache_dir=cache_dir,
    )
    print(f"  test samples: {len(test_ds):,}")

    _use_persistent = (args.num_workers > 0)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_use_persistent,
        prefetch_factor=(4 if _use_persistent else None),
    )

    # ── Persistence baseline ─────────────────────────────────────────────
    print("\nComputing persistence baseline …")
    persist_metrics = compute_persistence_metrics(test_loader, device, obs_stats)

    from model.embeddings import TARGET_VARIABLE_NAMES
    print("  Persistence RMSE (normalised):")
    for var_name in TARGET_VARIABLE_NAMES:
        r = persist_metrics.get(f"persist_{var_name}_rmse_norm", float("nan"))
        print(f"    {var_name:<14}  {r:.5f}")
    print(f"    {'[overall]':<14}  {persist_metrics.get('persist_overall_rmse_norm', float('nan')):.5f}")

    # ── Per-checkpoint evaluation ─────────────────────────────────────────
    all_results: list[dict] = []

    for ckpt_path in args.checkpoint:
        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")

        # Load checkpoint metadata
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_epoch    = ckpt.get("epoch",      "?")
        ckpt_val_loss = ckpt.get("val_loss",   float("nan"))
        print(f"  Saved at epoch {ckpt_epoch},  val_loss={ckpt_val_loss:.5f}")

        # Build model
        from model.mae import StationMAE
        model = StationMAE(
            d_model=args.d_model,
            enc_heads=args.enc_heads,
            enc_layers=args.enc_layers,
            dec_heads=args.dec_heads,
            dec_layers=args.dec_layers,
            mlp_ratio=args.mlp_ratio,
            dropout=0.0,               # no dropout at inference
            mask_ratio=args.mask_ratio,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Parameters: {model.count_parameters():,}")

        # Run full evaluation
        from engine.evaluate import evaluate_full, print_full_metrics
        print("  Running evaluate_full() …")
        metrics = evaluate_full(model, test_loader, device, obs_stats=obs_stats)

        # Skill scores vs persistence
        skill = compute_skill_scores(metrics, persist_metrics)

        # Print
        print_full_metrics(metrics, obs_stats=obs_stats)

        print("── Skill scores vs persistence (higher = better, 1.0 = perfect) ──")
        for var_name in TARGET_VARIABLE_NAMES:
            s = skill.get(f"skill_{var_name}", float("nan"))
            bar_len = max(0, int(s * 20)) if not (s != s) else 0  # NaN guard
            bar = "█" * bar_len
            print(f"  {var_name:<14}  {s:+.3f}  {bar}")
        s_ov = skill.get("skill_overall", float("nan"))
        print(f"  {'[overall]':<14}  {s_ov:+.3f}")

        label = os.path.splitext(os.path.basename(ckpt_path))[0]
        all_results.append({
            "label":   label,
            "path":    ckpt_path,
            "epoch":   ckpt_epoch,
            "metrics": metrics,
            "skill":   skill,
        })

    if not all_results:
        print("\nNo valid checkpoints evaluated.")
        return

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = save_metrics_csv(all_results, args.save_dir)
    print(f"\nMetrics saved to: {csv_path}")

    # ── Save persistence metrics too ─────────────────────────────────────
    persist_csv = os.path.join(args.save_dir, "persistence_metrics.csv")
    with open(persist_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(persist_metrics.items()):
            writer.writerow([k, f"{v:.6f}"])
    print(f"Persistence metrics saved to: {persist_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating plots …")
        plot_delta_rmse(all_results, persist_metrics, args.save_dir)

    print(f"\nAll results written to: {os.path.abspath(args.save_dir)}")
    print("Done.")


if __name__ == "__main__":
    main()
