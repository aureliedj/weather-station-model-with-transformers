"""
test.py

Test-set inference and evaluation for Station-MAE.

Loads a trained checkpoint, runs the model on the held-out test split, and
reports a comprehensive set of metrics in both normalised and physical units.

Usage
-----
    # Minimal — all settings auto-read from the Lightning checkpoint
    python test.py --data_root /path/to/peakweather --checkpoint checkpoints/run1/best.ckpt

    # Override a specific setting (e.g. larger batch for faster inference)
    python test.py --data_root /path/to/peakweather \\
                   --checkpoint checkpoints/run1/best.ckpt \\
                   --batch_size 64 --save_dir results/run1

    # Multi-checkpoint comparison (settings read from each checkpoint individually)
    python test.py --data_root /path/to/peakweather \\
                   --checkpoint checkpoints/run1/best.ckpt checkpoints/run2/best.ckpt \\
                   --save_dir results/

    # With Weights & Biases logging
    python test.py --data_root /path/to/peakweather \\
                   --checkpoint checkpoints/run1/best.ckpt \\
                   --wandb_project station-mae --wandb_run_name test-best

Arguments
---------
  Data
    --data_root        STR   Path to PeakWeather data directory (required)
    --cache_dir        STR   Pre-built tensor cache (defaults to data_root)
    --window           INT   Input window steps (auto-read from checkpoint; fallback 288)
    --max_delta        INT   Max lead-time steps (auto-read from checkpoint; fallback 18)
    --batch_size       INT   Inference batch size (default 32)
    --num_workers      INT   DataLoader workers (default 4)

  Model
    --checkpoint       STR   Path to .ckpt checkpoint file (required; can be repeated).
                             For Lightning checkpoints all arch + data settings are
                             auto-detected; only --data_root is strictly required.
    --d_model          INT   Auto-read from checkpoint (override if needed)
    --enc_heads        INT   Auto-read from checkpoint
    --enc_layers       INT   Auto-read from checkpoint
    --dec_heads        INT   Auto-read from checkpoint
    --dec_layers       INT   Auto-read from checkpoint
    --mlp_ratio        FLT   Auto-read from checkpoint
    --mask_ratio       FLT   Auto-read from checkpoint
    --no_spatial_attn        Auto-read from checkpoint
    --temporal_window  INT   Auto-read from checkpoint

  Output
    --save_dir         STR   Directory for metrics CSV and plots (default "test_results")
    --no_plots               Skip matplotlib plots
    --gap_fill_repeats INT   Random masks per window for gap-filling eval (default 3)
    --wandb_project    STR   WandB project name; omit to disable WandB logging
    --wandb_run_name   STR   WandB run name (default: "test-<checkpoint_stem>")

Evaluation modes
----------------
  Standard (all deltas, all stations):
    Per variable RMSE, MAE, Bias, R² — normalised + physical units.
    Wind speed / direction.  Per-delta RMSE table.
    Skill score vs persistence baseline.
    → Saved to:  test_metrics.csv

  Lead-1 / 10-min forecast (delta = 1 step, all stations):
    Same metrics as standard eval but restricted to a fixed 10-min lead-time.
    → Saved to:  lead1_metrics.csv

  Spatial gap-filling (delta = 0, masked stations only):
    Target = last input timestep; model reconstructs masked stations from context.
    n_repeats independent random masks applied per window for stable estimates.
    Metrics computed only on the hidden (masked) stations.
    → Saved to:  gap_filling_metrics.csv

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
    # default=None → auto-read from checkpoint; explicit CLI value always wins
    p.add_argument("--window",      type=int, default=None,
                   help="Input window in 10-min steps (default: read from checkpoint, "
                        "fallback 72 = 12 h)")
    p.add_argument("--max_delta",   type=int, default=None,
                   help="Max lead-time in 10-min steps (default: read from checkpoint, "
                        "fallback 36 = 6 h)")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Model architecture — all flags are auto-detected from Lightning checkpoints.
    # Pass any of these explicitly to override the saved value.
    p.add_argument("--checkpoint",  type=str, nargs="+", required=True,
                   help="One or more .pt / .ckpt checkpoint paths to evaluate. "
                        "For Lightning checkpoints all arch and data settings are "
                        "read automatically; only --data_root is required alongside.")
    p.add_argument("--d_model",     type=int,   default=None)
    p.add_argument("--enc_heads",   type=int,   default=None)
    p.add_argument("--enc_layers",  type=int,   default=None)
    p.add_argument("--dec_heads",   type=int,   default=None)
    p.add_argument("--dec_layers",  type=int,   default=None)
    p.add_argument("--mlp_ratio",   type=float, default=None)
    p.add_argument("--mask_ratio",  type=float, default=None)
    p.add_argument("--factorised_encoder", action="store_true", default=None,
                   help="Axial attention encoder (auto-detected from Lightning checkpoints)")
    p.add_argument("--no_spatial_attn", action="store_true", default=None,
                   help="Temporal-only encoder (auto-detected from Lightning checkpoints)")
    p.add_argument("--temporal_window", type=int, default=None,
                   help="Local temporal attention window (auto-detected from Lightning checkpoints)")
    p.add_argument("--cross_attn_decoder", action="store_true", default=None,
                   help="Cross-attention decoder (auto-detected from Lightning checkpoints)")
    p.add_argument("--device",      type=str, default=None)

    # Output
    p.add_argument("--save_dir",          type=str, default="test_results")
    p.add_argument("--no_plots",          action="store_true")
    p.add_argument("--gap_fill_repeats",  type=int, default=3,
                   help="Number of random masks per window for gap-filling eval (default 3)")
    p.add_argument("--wandb_project",     type=str, default=None,
                   help="WandB project name; omit to disable WandB logging")
    p.add_argument("--wandb_run_name",    type=str, default=None,
                   help="WandB run name (default: 'test-<checkpoint_stem>')")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint cfg helper
# ---------------------------------------------------------------------------

def _read_lightning_cfg(path: str) -> dict:
    """
    Load a Lightning .ckpt and return its saved hyper_parameters["cfg"] dict.
    Returns {} if the file is not a Lightning checkpoint or the key is absent.
    Does NOT load model weights — only the cfg sub-dict is read.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("hyper_parameters", {}).get("cfg", {})
    except Exception:
        return {}


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

    Evaluated over ALL N stations wherever sensor data is present at the
    target step, consistent with the model's all-station evaluation protocol.

    Returns dict with same key structure as evaluate_full() for easy comparison.
    """
    from model.embeddings import TARGET_VARIABLE_NAMES, NUM_TARGET_VARIABLES

    std_t = obs_stats["std"].cpu()

    # Batch-level records matching evaluate_full() structure
    records: list[dict] = []

    for batch in loader:
        x           = batch["x"]            # (B, W, N, V)
        y_raw       = batch["y"]            # (B, N, V) or (B, K, N, V)
        y_mask_raw  = batch["y_mask"]
        delta_steps = batch["delta_steps"]  # (B,) or (B, K)

        if y_raw.dim() == 4:
            y_raw       = y_raw[:, 0]
            y_mask_raw  = y_mask_raw[:, 0]
            delta_steps = delta_steps[:, 0]

        B, W, N, V = x.shape

        # Persistence: last observed input step as forecast for every lead-time
        persist_pred  = x[:, -1, :, :NUM_TARGET_VARIABLES]       # (B, N, 5)
        y_target      = y_raw[:, :, :NUM_TARGET_VARIABLES]       # (B, N, 5)
        y_mask_target = y_mask_raw[:, :, :NUM_TARGET_VARIABLES]  # (B, N, 5)

        # Flatten all N stations — no masking simulation needed
        records.append({
            "pred":   persist_pred.reshape(B * N, NUM_TARGET_VARIABLES),
            "target": y_target.reshape(B * N, NUM_TARGET_VARIABLES),
            "mask":   y_mask_target.reshape(B * N, NUM_TARGET_VARIABLES),
            "deltas": delta_steps.tolist(),
            "N":      N,
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

    # Per-delta (expand batch records → per-sample then group by delta)
    sample_records: list[dict] = []
    for r in records:
        N_r = r["N"]
        for b, d in enumerate(r["deltas"]):
            sample_records.append({
                "pred":   r["pred"]  [b * N_r : (b + 1) * N_r],
                "target": r["target"][b * N_r : (b + 1) * N_r],
                "mask":   r["mask"]  [b * N_r : (b + 1) * N_r],
                "delta":  d,
            })

    unique_deltas = sorted({sr["delta"] for sr in sample_records})
    for d in unique_deltas:
        recs_d = [sr for sr in sample_records if sr["delta"] == d]
        p_d = torch.cat([sr["pred"]   for sr in recs_d], dim=0)
        t_d = torch.cat([sr["target"] for sr in recs_d], dim=0)
        m_d = torch.cat([sr["mask"]   for sr in recs_d], dim=0).bool()
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

def save_extra_metrics_csv(
    all_results: list[dict],   # list of {"label": str, "metrics": dict}
    save_dir: str,
    filename: str,
) -> str:
    """Save an auxiliary metrics table (lead-1 or gap-filling) to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    all_keys: list[str] = []
    for r in all_results:
        for k in r["metrics"]:
            if k not in all_keys:
                all_keys.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint"] + all_keys)
        for r in all_results:
            row = [r["label"]]
            row += [f"{r['metrics'].get(k, float('nan')):.6f}" for k in all_keys]
            writer.writerow(row)

    return path


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

    # ── Auto-read settings from first Lightning checkpoint ────────────────
    # Loads only the cfg sub-dict (no weights); fast even for large checkpoints.
    # CLI flags always win when explicitly provided (non-None).
    _ckpt_cfg: dict = {}
    for _p in args.checkpoint:
        if os.path.exists(_p):
            _ckpt_cfg = _read_lightning_cfg(_p)
            if _ckpt_cfg:
                print(f"Auto-detected settings from checkpoint: {_p}")
                break

    def _resolve(cli_val, cfg_key: str, fallback):
        """Return CLI value if set (not None), else checkpoint cfg, else fallback."""
        if cli_val is not None:
            return cli_val
        return _ckpt_cfg.get(cfg_key, fallback)

    # Data settings — resolved before datasets are built
    window          = _resolve(args.window,    "window",    288)
    max_delta       = _resolve(args.max_delta, "max_delta", 18)
    exclude_stations = _ckpt_cfg.get("exclude_stations", None) or None

    # Arch settings — resolved per-checkpoint below, but define defaults here
    _d_model    = _resolve(args.d_model,    "d_model",    128)
    _enc_heads  = _resolve(args.enc_heads,  "enc_heads",  4)
    _enc_layers = _resolve(args.enc_layers, "enc_layers", 4)
    _dec_heads  = _resolve(args.dec_heads,  "dec_heads",  4)
    _dec_layers = _resolve(args.dec_layers, "dec_layers", 2)
    _mlp_ratio  = _resolve(args.mlp_ratio,  "mlp_ratio",  4.0)
    _mask_ratio = _resolve(args.mask_ratio, "mask_ratio", 0.5)

    print(f"  window={window}  max_delta={max_delta}  "
          f"d_model={_d_model}  enc_layers={_enc_layers}  dec_layers={_dec_layers}")
    if exclude_stations:
        print(f"  exclude_stations={exclude_stations}  "
              f"(matched against stations_table index / name / abbr)")

    # ── Data ─────────────────────────────────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset

    cache_dir = args.cache_dir or args.data_root

    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    # Build train_ds only to get obs_stats (do not iterate over it).
    # Exclude the same stations as training so normalisation stats match exactly.
    print("Building train dataset for normalisation statistics …")
    train_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="train",
        num_delta_per_sample=1, max_delta_steps=max_delta,
        cache_dir=cache_dir,
        exclude_stations=exclude_stations,
    )
    obs_stats = train_ds.obs_stats
    print("  obs_stats ready (train split)")

    print("Building test dataset …")
    test_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="test",
        obs_stats=obs_stats,
        num_delta_per_sample=1,          # random delta in [1, max_delta] per sample
        max_delta_steps=max_delta,
        cache_dir=cache_dir,
        exclude_stations=exclude_stations,
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

    # ── Lead-1 (10-min forecast) dataset & loader ─────────────────────────
    print("Building lead-1 test dataset (delta_steps=1, fixed) …")
    lead1_ds = StationMAEDataset(
        ds,
        window_size=window,
        delta_steps=1,
        split="test",
        obs_stats=obs_stats,
        num_delta_per_sample=1,
        max_delta_steps=1,          # clamp: only delta=1 ever sampled
        cache_dir=cache_dir,
        exclude_stations=exclude_stations,
    )
    print(f"  lead-1 test samples: {len(lead1_ds):,}")

    lead1_loader = DataLoader(
        lead1_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_use_persistent,
        prefetch_factor=(4 if _use_persistent else None),
    )

    # ── Persistence baseline (standard loader — all deltas) ──────────────
    print("\nComputing persistence baseline …")
    persist_metrics = compute_persistence_metrics(test_loader, device, obs_stats)

    from model.embeddings import TARGET_VARIABLE_NAMES
    print("  Persistence RMSE (normalised):")
    for var_name in TARGET_VARIABLE_NAMES:
        r = persist_metrics.get(f"persist_{var_name}_rmse_norm", float("nan"))
        print(f"    {var_name:<14}  {r:.5f}")
    print(f"    {'[overall]':<14}  {persist_metrics.get('persist_overall_rmse_norm', float('nan')):.5f}")

    # ── Per-checkpoint evaluation ─────────────────────────────────────────
    all_results:       list[dict] = []
    all_lead1_results: list[dict] = []
    all_gap_results:   list[dict] = []

    for ckpt_path in args.checkpoint:
        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")

        # ── Load checkpoint (Lightning .ckpt or legacy .pt) ─────────────
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if "state_dict" in ckpt:
            # Lightning format — weights stored under "state_dict" with "model." prefix
            ckpt_epoch    = ckpt.get("epoch", "?")
            ckpt_val_loss = float("nan")

            # All arch settings: checkpoint cfg wins, CLI flags override when set
            saved_cfg = ckpt.get("hyper_parameters", {}).get("cfg", {})

            def _arch(cli_val, cfg_key: str, fallback):
                if cli_val is not None:
                    return cli_val
                return saved_cfg.get(cfg_key, fallback)

            c_d_model    = _arch(args.d_model,    "d_model",           _d_model)
            c_enc_heads  = _arch(args.enc_heads,  "enc_heads",         _enc_heads)
            c_enc_layers = _arch(args.enc_layers, "enc_layers",        _enc_layers)
            c_dec_heads  = _arch(args.dec_heads,  "dec_heads",         _dec_heads)
            c_dec_layers = _arch(args.dec_layers, "dec_layers",        _dec_layers)
            c_mlp_ratio  = _arch(args.mlp_ratio,  "mlp_ratio",         _mlp_ratio)
            c_mask_ratio = _arch(args.mask_ratio, "mask_ratio",        _mask_ratio)
            factorised   = _arch(args.factorised_encoder, "factorised_encoder", False)
            no_spatial   = _arch(args.no_spatial_attn,   "no_spatial_attn",    False)
            temp_window  = _arch(args.temporal_window,   "temporal_window",    0)
            cross_attn   = _arch(args.cross_attn_decoder,"cross_attn_decoder", False)

            # Strip "model." prefix (Lightning) and then "_orig_mod." prefix
            # (torch.compile wraps parameters under OptimizedModule).
            state_dict = {}
            for k, v in ckpt["state_dict"].items():
                if not k.startswith("model."):
                    continue
                k = k[len("model."):]
                if k.startswith("_orig_mod."):
                    k = k[len("_orig_mod."):]
                state_dict[k] = v
        else:
            # Legacy format saved by the old engine/train.py
            ckpt_epoch    = ckpt.get("epoch",    "?")
            ckpt_val_loss = ckpt.get("val_loss", float("nan"))
            c_d_model    = _d_model
            c_enc_heads  = _enc_heads
            c_enc_layers = _enc_layers
            c_dec_heads  = _dec_heads
            c_dec_layers = _dec_layers
            c_mlp_ratio  = _mlp_ratio
            c_mask_ratio = _mask_ratio
            factorised   = bool(args.factorised_encoder)
            no_spatial   = bool(args.no_spatial_attn)
            temp_window  = args.temporal_window or 0
            cross_attn   = bool(args.cross_attn_decoder)
            state_dict   = ckpt["model_state_dict"]

        print(f"  Saved at epoch {ckpt_epoch}  "
              + (f"val_loss={ckpt_val_loss:.5f}" if ckpt_val_loss == ckpt_val_loss else ""))
        print(f"  d_model={c_d_model}  enc_layers={c_enc_layers}  dec_layers={c_dec_layers}  "
              f"mlp_ratio={c_mlp_ratio}")
        print(f"  factorised_encoder={factorised}  no_spatial_attn={no_spatial}  "
              f"temporal_window={temp_window}  cross_attn_decoder={cross_attn}")

        # Build model
        from model.mae import StationMAE
        model = StationMAE(
            d_model=c_d_model,
            enc_heads=c_enc_heads,
            enc_layers=c_enc_layers,
            dec_heads=c_dec_heads,
            dec_layers=c_dec_layers,
            mlp_ratio=c_mlp_ratio,
            dropout=0.0,               # no dropout at inference
            mask_ratio=c_mask_ratio,
            factorised_encoder=factorised,
            encoder_spatial_attn=not no_spatial,
            temporal_window=temp_window,
            cross_attention_decoder=cross_attn,
        ).to(device)
        model.load_state_dict(state_dict)
        print(f"  Parameters: {model.count_parameters():,}")

        # ── Standard full evaluation (all deltas, all stations) ─────────
        from engine.evaluate import (
            evaluate_full, print_full_metrics,
            evaluate_gap_filling, print_gap_filling_metrics,
        )
        print("  Running evaluate_full() [all deltas, all stations] …")
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

        # ── Lead-1 evaluation (delta = 1 step = 10 min, all stations) ───
        print(f"\n  Running evaluate_full() [lead-1 / 10-min forecast, all stations] …")
        lead1_metrics = evaluate_full(model, lead1_loader, device, obs_stats=obs_stats)

        print("\n── Lead-1 (10-min forecast) · all stations ──────────────────────────")
        print_full_metrics(lead1_metrics, obs_stats=obs_stats)

        all_lead1_results.append({
            "label":   label,
            "metrics": lead1_metrics,
        })

        # ── Gap-filling evaluation (delta = 0, masked stations only) ────
        print(f"\n  Running evaluate_gap_filling() "
              f"[n_repeats={args.gap_fill_repeats}] …")
        gap_metrics = evaluate_gap_filling(
            model, test_loader, device,
            obs_stats=obs_stats,
            n_repeats=args.gap_fill_repeats,
        )

        print("\n── Spatial gap-filling · masked stations only ───────────────────────")
        print_gap_filling_metrics(gap_metrics, obs_stats=obs_stats)

        all_gap_results.append({
            "label":   label,
            "metrics": gap_metrics,
        })

        # ── WandB logging (optional) ─────────────────────────────────────
        if args.wandb_project:
            try:
                import wandb
                run_name = args.wandb_run_name or f"test-{label}"
                _wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    job_type="evaluation",
                    config={
                        "checkpoint":         ckpt_path,
                        "epoch":              ckpt_epoch,
                        "window":             args.window,
                        "max_delta":          args.max_delta,
                        "mask_ratio":         args.mask_ratio,
                        "gap_fill_repeats":   args.gap_fill_repeats,
                        "factorised_encoder": factorised,
                        "cross_attn_decoder": cross_attn,
                    },
                    reinit=True,
                )

                # -- Standard eval: log per-delta RMSE as a table + summary ──
                delta_keys = sorted(
                    k for k in metrics
                    if k.startswith("delta_") and k.endswith("_overall_rmse_norm")
                )
                for dk in delta_keys:
                    d    = int(dk.split("_")[1])
                    rmse = metrics[dk]
                    mae  = metrics.get(f"delta_{d:02d}_overall_mae_norm", float("nan"))
                    wandb.log({
                        "delta_min":          d * 10,
                        "eval/rmse_norm":     rmse,
                        "eval/mae_norm":      mae,
                    })

                # Summary: all flat metrics prefixed by mode
                summary_flat: dict[str, float] = {}
                for k, v in metrics.items():
                    summary_flat[f"eval/{k}"] = v
                for k, v in skill.items():
                    summary_flat[f"eval/{k}"] = v
                for k, v in lead1_metrics.items():
                    summary_flat[f"lead1/{k}"] = v
                for k, v in gap_metrics.items():
                    summary_flat[f"gap/{k}"] = v

                wandb.summary.update(summary_flat)
                _wandb_run.finish()
                print(f"  WandB run logged: {run_name} "
                      f"(project={args.wandb_project})")
            except ImportError:
                print("  [WandB] wandb not installed — skipping logging")

    if not all_results:
        print("\nNo valid checkpoints evaluated.")
        return

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = save_metrics_csv(all_results, args.save_dir)
    print(f"\nMetrics saved to: {csv_path}")

    if all_lead1_results:
        lead1_csv = save_extra_metrics_csv(
            all_lead1_results, args.save_dir, "lead1_metrics.csv"
        )
        print(f"Lead-1 metrics saved to: {lead1_csv}")

    if all_gap_results:
        gap_csv = save_extra_metrics_csv(
            all_gap_results, args.save_dir, "gap_filling_metrics.csv"
        )
        print(f"Gap-filling metrics saved to: {gap_csv}")

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
