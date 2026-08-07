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

# Renku session containers cap /dev/shm (often 64 MB), which the default sharing
# strategy can exhaust when DataLoader workers hand batches back to the main
# process on the large sliding test set → "No space left on device". Route the
# handoff through temp files instead. Same guard as main.py / train_lstm.py /
# test_lstm.py.
torch.multiprocessing.set_sharing_strategy("file_system")


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
    p.add_argument("--temporal_patch", type=int, default=None,
                   help="Auto-read from the checkpoint; override only to debug.")
    p.add_argument("--residual_head", action="store_true", default=None,
                   help="v15 residual head; auto-read from the checkpoint cfg.")
    p.add_argument("--temporal_window", type=int, default=None,
                   help="Local temporal attention window (auto-detected from Lightning checkpoints)")
    p.add_argument("--cross_attn_decoder", action="store_true", default=None,
                   help="Cross-attention decoder (auto-detected from Lightning checkpoints)")
    p.add_argument("--device",      type=str, default=None)

    # Output
    p.add_argument("--exclude_stations",   type=str, nargs="+", default=None,
                   help="Override checkpoint exclude_stations. Use abbreviation e.g. PFA.")
    p.add_argument("--global_norm",       action="store_true",
                   help="Use global per-variable normalisation (one mean/std per variable). "
                        "Required when testing checkpoints trained before per-station "
                        "normalisation was introduced. Without this flag, per-station "
                        "normalisation is used, which is inconsistent with old checkpoints "
                        "and produces astronomical RMSE values.")
    p.add_argument("--test_mask_ratios",  type=float, nargs="+", default=None,
                   help="One or more encoder mask ratios to sweep at inference time. "
                        "Default: use the trained mask_ratio from the checkpoint. "
                        "Example: --test_mask_ratios 0.0 0.5 to compare no-masking vs trained. "
                        "  0.0 → all stations visible (pure temporal forecasting). "
                        "  0.5 → 50%% masked (trained setting, gap-filling + forecasting).")
    p.add_argument("--index_mode",        type=str, default="sliding",
                   choices=["sliding", "blocks"],
                   help="Window selection for the test split.\n"
                        "  sliding (default): all contiguous windows, stride=1 (~105k samples).\n"
                        "    Most stable metrics; slow.\n"
                        "  blocks: non-overlapping windows only (~1,460 samples).\n"
                        "    Fast evaluation; use for paper metrics and quick checks.")
    p.add_argument("--stride",            type=int, default=1,
                   help="Sliding forecast-origin spacing in 10-min steps (9 = 90 min = "
                        "1h30 → rolling-origin evaluation). Only used with --index_mode "
                        "sliding; ignored for blocks. Default 1 = every window.")
    p.add_argument("--save_dir",          type=str, default="test_results")
    p.add_argument("--no_plots",          action="store_true")
    p.add_argument("--save_predictions",  type=int, default=0,
                   help="Save raw predictions for N test windows to a .pt file "
                        "(0 = disabled). Enables time-series and spatial plots in "
                        "the notebook. Each window includes preds, targets, masks, "
                        "delta_steps, timestamps, and station spatial features. "
                        "~40 MB per 100 windows at d_model=1024.")
    p.add_argument("--predictions_only",  action="store_true",
                   help="Dump predictions.pt per mask ratio and nothing else: no "
                        "persistence baseline, no lead-30min pass, no metrics CSVs, "
                        "no plots. One forward pass over the test windows per mask "
                        "ratio; all metrics are computed downstream from the saved "
                        "predictions (Test_Results_Exploration.ipynb). "
                        "--save_predictions caps the window count (0 = ALL).")
    p.add_argument("--seasonal",          action="store_true",
                   help="Compute and save per-season metrics (DJF/MAM/JJA/SON). "
                        "Useful for analysing model behaviour across weather regimes.")
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

    _std = obs_stats["std"].cpu()
    # Handle (N, V) per-station stats — use station-mean for persistence display
    std_t = _std.mean(dim=0) if _std.dim() == 2 else _std   # (V,)

    # Batch-level records matching evaluate_full() structure
    records: list[dict] = []

    for batch in loader:
        x           = batch["x"]            # (B, W, N, V)
        y_raw       = batch["y"]            # (B, N, V) or (B, K, N, V)
        y_mask_raw  = batch["y_mask"]
        delta_steps = batch["delta_steps"]  # (B,) or (B, K)

        B, W, N, V = x.shape
        persist_pred = x[:, -1, :, :NUM_TARGET_VARIABLES]   # (B, N, 5)

        # Iterate over all K deltas (fixed-grid) or just the single delta
        if y_raw.dim() == 4:
            K = y_raw.shape[1]
        else:
            K = 1
            y_raw       = y_raw.unsqueeze(1)
            y_mask_raw  = y_mask_raw.unsqueeze(1)
            delta_steps = delta_steps.unsqueeze(1)

        for k in range(K):
            y_k  = y_raw[:, k];   ym_k = y_mask_raw[:, k];   ds_k = delta_steps[:, k]
            y_target      = y_k[:, :, :NUM_TARGET_VARIABLES]
            y_mask_target = ym_k[:, :, :NUM_TARGET_VARIABLES]
            records.append({
                "pred":   persist_pred.reshape(B * N, NUM_TARGET_VARIABLES),
                "target": y_target.reshape(B * N, NUM_TARGET_VARIABLES),
                "mask":   y_mask_target.reshape(B * N, NUM_TARGET_VARIABLES),
                "deltas": ds_k.tolist(),
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

def _prediction_sanity_check(res: dict, label: str = "") -> None:
    """
    Fast post-dump sanity check on the freshly collected predictions.

    Purpose: catch a broken model rebuild AT DUMP TIME, not days later in the
    notebook. The v13 incident (input-context block silently dropped) would
    have tripped three of the four checks below.

    Checks (on a ≤500-window subsample, normalised space, ~1 s):
      1. finiteness            — NaN/Inf in predictions
      2. dispersion            — pred std vs target std per variable
                                 (conditional-mean collapse → ratio ≪ 1)
      3. persistence collapse  — Δ>0 overall RMSE vs the last-observation
                                 baseline (same definition as test_lstm.py)
      4. context pathway       — at Δ=0, VISIBLE stations must beat MASKED
                                 ones: the decoder can read visible stations
                                 directly from input_context. visible ≈ masked
                                 is the signature of a severed context path.
    Warnings are printed, never raised — the dump is already on disk and may
    still be useful for diagnosis.
    """
    import numpy as np
    Vt   = res["preds"].shape[-1]
    Mw   = res["preds"].shape[0]
    sub  = torch.arange(0, Mw, max(1, Mw // 500))
    P    = res["preds"][sub].numpy()
    T    = res["targets"][sub][..., :Vt].numpy()
    M    = res["masks"][sub][..., :Vt].numpy() > 0.5
    MI   = res.get("masked_idx")
    vnames = list(res.get("var_names", [f"var{v}" for v in range(Vt)]))
    print(f"\n  ── Sanity check ({label}, {len(sub)} windows) ──")

    # 1. finiteness
    n_bad = int(np.count_nonzero(~np.isfinite(P)))
    print(f"  {'✓' if n_bad == 0 else '⚠'} finiteness: "
          f"{'all finite' if n_bad == 0 else f'{n_bad} NaN/Inf values!'}")

    # 2. dispersion (conditional-mean collapse)
    ratios = []
    for v in range(Vt):
        m = M[..., v]
        r = float(P[..., v][m].std() / max(T[..., v][m].std(), 1e-6))
        ratios.append(r)
    worst = min(ratios)
    flag  = "✓" if worst > 0.5 else "⚠"
    print(f"  {flag} dispersion (pred std / target std): "
          + "  ".join(f"{vnames[v][:4]}={ratios[v]:.2f}" for v in range(Vt))
          + ("" if worst > 0.5 else "   ← ≪1 = mean-collapse"))

    # 3. persistence collapse on Δ>0 (identical definition to test_lstm.py)
    K  = P.shape[1]
    fc = slice(1, K)
    persist = np.repeat(T[:, :1], K, axis=1)
    both    = M & M[:, :1]
    e  = (P[:, fc] - T[:, fc])[both[:, fc]]
    ep = (persist[:, fc] - T[:, fc])[both[:, fc]]
    rm, rp = float(np.sqrt(np.mean(e**2))), float(np.sqrt(np.mean(ep**2)))
    skill  = 1.0 - rm / rp if rp > 0 else float("nan")
    flag   = "✓" if skill > -0.5 else "⚠"
    print(f"  {flag} vs persistence (Δ>0 norm RMSE): model {rm:.4f} "
          f"vs persist {rp:.4f} → skill {skill:+.3f}"
          + ("" if skill > -0.5 else "   ← far below persistence: check the rebuild"))

    # 4. context pathway: visible vs masked stations at Δ=0
    if MI is not None and MI.shape[1] > 0:
        mi  = MI[sub].numpy()
        N   = P.shape[2]
        selm = np.zeros((len(sub), N), bool)
        np.put_along_axis(selm, mi, True, axis=1)
        m0   = M[:, 0]                                    # (m, N, Vt) at Δ=0
        e0   = np.abs(P[:, 0] - T[:, 0])
        mae_mask = float(e0[m0 & selm[:, :, None]].mean())
        mae_vis  = float(e0[m0 & ~selm[:, :, None]].mean())
        ratio    = mae_vis / max(mae_mask, 1e-6)
        flag     = "✓" if ratio < 0.9 else "⚠"
        print(f"  {flag} Δ=0 context path: visible MAE {mae_vis:.4f} vs "
              f"masked {mae_mask:.4f} (ratio {ratio:.2f})"
              + ("" if ratio < 0.9 else
                 "   ← visible ≈ masked: input_context may be severed"))
    print()


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

    # ── Fail fast on missing checkpoints ──────────────────────────────────
    # Building the datasets takes minutes. Validate first, and refuse to run at
    # all if NOTHING can be evaluated — otherwise the config below silently
    # falls back to library defaults (d_model=128, window=288, max_delta=18),
    # the run "succeeds", and writes nothing. That failure mode cost a full
    # evaluation cycle once; it must be loud.
    _missing = [p for p in args.checkpoint if not os.path.exists(p)]
    _present = [p for p in args.checkpoint if os.path.exists(p)]
    if _missing:
        print("\n[ERROR] checkpoint(s) not found:")
        for p in _missing:
            print(f"    {p}")
            _d = os.path.dirname(p)
            if os.path.isdir(_d):
                _sib = sorted(f for f in os.listdir(_d) if f.endswith(".ckpt"))
                print(f"      dir exists; .ckpt files here: {_sib or '(none)'}")
            elif os.path.isdir(os.path.dirname(_d)):
                _runs = sorted(os.listdir(os.path.dirname(_d)))
                print(f"      no such dir. Available runs in "
                      f"{os.path.dirname(_d)}: {_runs or '(none)'}")
            else:
                print(f"      parent directory does not exist either: {_d}")
    if not _present:
        raise SystemExit(
            "\n[ABORT] none of the requested checkpoints exist — nothing to evaluate.\n"
            "        Check CHECKPOINT in src/scripts/run_test_cloud.sh, and that it\n"
            "        matches RUN_NAME (the run being written to test_results/)."
        )
    if _missing:
        print(f"\n  continuing with {len(_present)} of {len(args.checkpoint)} checkpoint(s)")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Auto-read settings from first Lightning checkpoint ────────────────
    # Loads only the cfg sub-dict (no weights); fast even for large checkpoints.
    # CLI flags always win when explicitly provided (non-None).
    _ckpt_cfg: dict = {}
    for _p in _present:
        _ckpt_cfg = _read_lightning_cfg(_p)
        if _ckpt_cfg:
            print(f"Auto-detected settings from checkpoint: {_p}")
            break
    if not _ckpt_cfg:
        # The checkpoint exists but carries no hyper_parameters. Every setting
        # below then falls back to a library default that almost certainly does
        # not match how the model was trained — say so rather than printing a
        # confident banner full of wrong numbers.
        print("\n[WARN] no cfg found inside the checkpoint(s).")
        print("       Architecture and window settings will fall back to DEFAULTS")
        print("       (window=288, max_delta=18, d_model=128) unless passed on the CLI.")
        print("       If the banner below does not match your training run, stop now.\n")

    def _resolve(cli_val, cfg_key: str, fallback):
        """Return CLI value if set (not None), else checkpoint cfg, else fallback."""
        if cli_val is not None:
            return cli_val
        return _ckpt_cfg.get(cfg_key, fallback)

    # Data settings — resolved before datasets are built
    window             = _resolve(args.window,    "window",    288)
    max_delta          = _resolve(args.max_delta, "max_delta", 18)
    # CLI --exclude_stations overrides checkpoint value; use abbreviation e.g. PFA
    exclude_stations   = args.exclude_stations or _ckpt_cfg.get("exclude_stations", None) or None
    delta_mode         = _ckpt_cfg.get("delta_mode",         "fixed_grid")
    delta_grid_stride  = _ckpt_cfg.get("delta_grid_stride",  3)

    # Arch settings — resolved per-checkpoint below, but define defaults here
    _d_model    = _resolve(args.d_model,    "d_model",    128)
    _enc_heads  = _resolve(args.enc_heads,  "enc_heads",  4)
    _enc_layers = _resolve(args.enc_layers, "enc_layers", 4)
    _dec_heads  = _resolve(args.dec_heads,  "dec_heads",  4)
    _dec_layers = _resolve(args.dec_layers, "dec_layers", 2)
    _mlp_ratio  = _resolve(args.mlp_ratio,  "mlp_ratio",  4.0)
    _mask_ratio = _resolve(args.mask_ratio, "mask_ratio", 0.5)

    # ── Evaluation mode ───────────────────────────────────────────────────
    # max_delta == 0 → pure spatial inpainting (gap-filling):
    #   • no lead-times to sweep, no persistence baseline, no skill scores
    #   • gap-filling (masked stations only) is the primary metric
    # max_delta  > 0 → temporal forecasting:
    #   • full delta sweep, lead-1, persistence baseline + skill scores
    #   • gap-filling is a secondary metric
    _is_inpainting = (max_delta == 0)
    _mode_str = "inpainting (max_delta=0)" if _is_inpainting else f"forecasting (max_delta={max_delta})"
    print(f"  window={window}  max_delta={max_delta}  mode={_mode_str}  "
          f"d_model={_d_model}  enc_layers={_enc_layers}  dec_layers={_dec_layers}")

    # ── WandB — initialise early so the run appears immediately ──────────
    _wandb_run = None
    if args.wandb_project:
        try:
            import wandb as _wandb
            _run_name = args.wandb_run_name or f"test-{os.path.splitext(os.path.basename(args.checkpoint[0]))[0]}"
            _wandb_run = _wandb.init(
                project  = args.wandb_project,
                name     = _run_name,
                job_type = "evaluation",
                config   = {
                    "checkpoints":    args.checkpoint,
                    "index_mode":     args.index_mode,
                    "test_mask_ratios": args.test_mask_ratios,
                    "global_norm":    args.global_norm,
                    "gap_fill_repeats": args.gap_fill_repeats,
                    "window":         window,
                    "max_delta":      max_delta,
                },
            )
            print(f"WandB run                : {_run_name}  (project={args.wandb_project})")
        except ImportError:
            print("  [WandB] wandb not installed — skipping")

    # ── Data ─────────────────────────────────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset

    cache_dir = args.cache_dir or args.data_root

    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    # ── Station table diagnostic ──────────────────────────────────────────────
    # Printed once so you can verify station IDs and find the correct exclusion key.
    _stns = ds.stations_table
    _idx_sample = list(_stns.index[:5])
    _name_col   = next((c for c in ["name", "abbr", "station_name"] if c in _stns.columns), None)
    print(f"  stations_table: {len(_stns)} stations  "
          f"index dtype={_stns.index.dtype}  "
          f"sample indices={_idx_sample}")
    if _name_col:
        print(f"  sample {_name_col}s: {list(_stns[_name_col].head())}")
    if exclude_stations:
        _excl_upper = {str(s).upper() for s in exclude_stations}
        print(f"  looking for: {exclude_stations}")
        _matched = []
        for _idx in _stns.index:
            _candidates = {str(_idx).upper()}
            try:
                _candidates.add(str(int(float(str(_idx)))).upper())
            except (ValueError, TypeError):
                pass
            if _candidates & _excl_upper:
                _matched.append(str(_idx))
        print(f"  matched indices: {_matched if _matched else 'NONE — use abbreviation, e.g. PFA not 110'}")

    # Build train_ds only to get obs_stats (do not iterate over it).
    # Exclude the same stations as training so normalisation stats match exactly.
    print("Building train dataset for normalisation statistics …")
    if args.global_norm:
        print("  Normalisation: GLOBAL (per-variable) — use for old checkpoints")
    else:
        print("  Normalisation: per-station (per-station × per-variable)")

    train_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="train",
        num_delta_per_sample=1, max_delta_steps=max_delta,
        cache_dir=cache_dir,
        exclude_stations=exclude_stations,
        delta_mode=delta_mode,
        delta_grid_stride=delta_grid_stride,
        global_norm=args.global_norm,
    )
    obs_stats = train_ds.obs_stats
    print("  obs_stats ready (train split)")

    print("Building test dataset …")
    test_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="test",
        obs_stats=obs_stats,
        num_delta_per_sample=1,
        max_delta_steps=max_delta,
        cache_dir=cache_dir,
        exclude_stations=exclude_stations,
        delta_mode=delta_mode,
        delta_grid_stride=delta_grid_stride,
        index_mode=args.index_mode,
        train_stride=args.stride,
        global_norm=args.global_norm,
    )
    _window_mode_str = (
        f"non-overlapping blocks (~{len(test_ds):,} windows)"
        if args.index_mode == "blocks"
        else f"sliding stride={args.stride} ({len(test_ds):,} windows)"
    )
    print(f"  test samples: {_window_mode_str}  "
          f"(delta_mode={delta_mode}"
          + (f", grid=[0..{max_delta} step {delta_grid_stride}] K={len(test_ds.delta_grid)}"
             if delta_mode == "fixed_grid" else "") + ")")

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

    from model.embeddings import TARGET_VARIABLE_NAMES

    # ── Lead-30min dataset + persistence baseline (forecasting mode only) ─
    # Lead times follow the same 30-min fixed grid as training.
    # delta=3 steps = 30 min is the first non-zero horizon in the grid.
    # The old lead-1 (delta=1 = 10 min) has been removed — it is not part
    # of the training grid and produces inconsistent comparisons.
    lead1_loader   = None
    persist_metrics: dict = {}

    if not _is_inpainting and not args.predictions_only:
        _lead_delta = delta_grid_stride   # 3 steps = 30 min (matches training grid)
        print(f"Building lead-{_lead_delta*10}min test dataset (delta={_lead_delta} steps) …")
        lead1_ds = StationMAEDataset(
            ds,
            window_size=window,
            delta_steps=_lead_delta,
            split="test",
            obs_stats=obs_stats,
            num_delta_per_sample=1,
            max_delta_steps=_lead_delta,
            cache_dir=cache_dir,
            exclude_stations=exclude_stations,
            delta_mode="random",
            delta_grid_stride=_lead_delta,
            index_mode=args.index_mode,
            global_norm=args.global_norm,
        )
        print(f"  lead-{_lead_delta*10}min test samples: {len(lead1_ds):,}")

        lead1_loader = DataLoader(
            lead1_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=_use_persistent,
            prefetch_factor=(4 if _use_persistent else None),
        )

        print("\nComputing persistence baseline …")
        persist_metrics = compute_persistence_metrics(test_loader, device, obs_stats)
        print("  Persistence RMSE (normalised):")
        for var_name in TARGET_VARIABLE_NAMES:
            r = persist_metrics.get(f"persist_{var_name}_rmse_norm", float("nan"))
            print(f"    {var_name:<14}  {r:.5f}")
        print(f"    {'[overall]':<14}  "
              f"{persist_metrics.get('persist_overall_rmse_norm', float('nan')):.5f}")
    elif args.predictions_only:
        print("  [predictions-only] Skipping lead-30min dataset and persistence baseline.")
    else:
        print("  [Inpainting mode] Skipping lead-30min dataset and persistence baseline.")

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
        # map_location="cpu", NOT `device`. torch.load's deserializer restores
        # each tensor's storage on `map_location` DIRECTLY during unpickling
        # (serialization.py: default_restore_location -> obj.to(device=...)),
        # which is a different code path from an ordinary `.to(device)` call
        # on an already-constructed tensor. On some virtualised CUDA profiles
        # (observed here: NVIDIA A10-8Q, a GRID vGPU slice) that direct-to-CUDA
        # storage path raises "CUDA driver error: operation not supported"
        # even though CUDA otherwise works fine in the same process — the model
        # is moved to `device` two lines below via `.to(device)`, an ordinary
        # tensor op, which does not hit this restriction.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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
            temp_window  = _arch(args.temporal_window,   "temporal_window",    0)
            temp_patch   = _arch(args.temporal_patch,    "temporal_patch",     1)
            cross_attn   = _arch(args.cross_attn_decoder,"cross_attn_decoder", False)
            # v15 structural settings — recorded in cfg by main.py
            residual     = bool(saved_cfg.get("residual_head", False))

            # ── v18–v22 structural settings ──────────────────────────────
            # These were recorded by main.py from the start but never READ
            # here, so every one of them silently reverted to its default
            # when rebuilding the model. That is the same defect class as
            # input_cross_attn below, and it is not survivable:
            #
            #   value_embedding  "mlp"/"fourier" -> "linear" swaps the whole
            #                    per-variable map (mlp_w1/w2 vs var_weights)
            #   static_in_token  True -> False drops var_proj.static_maps AND
            #                    re-adds pos_emb/station_emb to the token
            #   direct_head      True -> False rebuilds a decoder that the
            #                    trained model does not have
            #
            # The defaults below are the pre-v18 behaviour, so v15 and
            # earlier checkpoints keep loading exactly as they did.
            value_emb    = saved_cfg.get("value_embedding", "linear") or "linear"
            wind_enc     = bool(saved_cfg.get("wind_encoder", False))
            static_tok   = bool(saved_cfg.get("static_in_token", False))
            direct_head  = bool(saved_cfg.get("direct_head", False))
            readout      = saved_cfg.get("readout", "last") or "last"
            # num_horizons must match the trained direct_proj output width, or
            # the head is the wrong shape. main.py derives it the same way.
            _max_delta   = int(saved_cfg.get("max_delta", 36))
            _grid_stride = int(saved_cfg.get("delta_grid_stride", 3) or 3)
            num_horizons = _max_delta // _grid_stride + 1

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
            temp_window  = args.temporal_window or 0
            temp_patch   = args.temporal_patch or 1
            cross_attn   = bool(args.cross_attn_decoder)
            residual     = bool(args.residual_head)          # v15
            # Legacy .pt checkpoints all predate v18, so the pre-v18 defaults
            # are the correct reconstruction here.
            value_emb    = "linear"
            wind_enc     = False
            static_tok   = False
            direct_head  = False
            readout      = "last"
            num_horizons = 13
            state_dict   = ckpt["model_state_dict"]
            saved_cfg    = {}

        print(f"  Saved at epoch {ckpt_epoch}  "
              + (f"val_loss={ckpt_val_loss:.5f}" if ckpt_val_loss == ckpt_val_loss else ""))
        print(f"  d_model={c_d_model}  enc_layers={c_enc_layers}  dec_layers={c_dec_layers}  "
              f"mlp_ratio={c_mlp_ratio}")
        print(f"  factorised_encoder={factorised}  "
              f"temporal_window={temp_window}  temporal_patch={temp_patch}  "
              f"cross_attn_decoder={cross_attn}")
        print(f"  [v15] residual_head={residual}")
        print(f"  [v18+] value_embedding={value_emb}  wind_encoder={wind_enc}  "
              f"static_in_token={static_tok}")
        print(f"  [v22]  direct_head={direct_head}"
              + (f"  readout={readout}  num_horizons={num_horizons}"
                 if direct_head else ""))

        # ── Detect an NLL (heteroscedastic) checkpoint ────────────────────────
        # The cfg key "nll_loss" is unreliable (absent/None on older runs such as
        # v9), so infer it from the weights: an NLL model has a second decoder
        # head predicting log σ².  Without this the head would be silently
        # dropped by strict=False and the predicted uncertainty lost.
        # ── Structural flags inferred from the WEIGHTS, not the cfg ───────
        # The cfg dict is not a reliable record: `input_context_cross_attn` was
        # never written into it, and test.py never passed it, so every
        # evaluation ever run rebuilt the decoder WITHOUT its input-context
        # cross-attention block. strict=False then dropped
        # decoder.input_cross_attn.* silently, severing the direct path from the
        # stations' current observations to the decoder. Symptom: the model
        # predicts a hidden station exactly as well as one it was shown.
        #
        # The state_dict cannot lie about which modules were trained, so infer
        # from it. This also fixes v9/v11/v12/v13 retroactively, whose cfgs
        # predate the key.
        _has_input_ctx = any("decoder.input_cross_attn." in k for k in state_dict)
        if _has_input_ctx:
            raise SystemExit(
                "\n[ABORT] This checkpoint contains decoder.input_cross_attn.* "
                "weights — it predates v15, where the input-context pathway was "
                "removed. Evaluate pre-v15 checkpoints with the code that "
                "trained them (git checkout main / the training-time commit). "
                "Rebuilding it here would silently drop those trained weights — "
                "the exact class of bug the v13 post-mortem uncovered."
            )

        # ── Shared pos_emb / station_emb / temporal_emb guard ───────────────
        # mae.py now builds ONE PositionalEmbedding/StationEmbedding/
        # TemporalEmbedding instance and registers it under BOTH
        # `encoder.<name>` and `decoder.<name>`, so a station's query and its
        # matching key carry a bit-identical positional fingerprint (see
        # tests/test_shared_embeddings.py). A checkpoint trained BEFORE that
        # change has two INDEPENDENTLY-trained modules at those two paths,
        # with different weight values.
        #
        # This is silent, unlike a missing/unexpected key: both key names
        # exist in the checkpoint AND in the current model, so
        # load_state_dict(strict=False) reports nothing wrong. What actually
        # happens is a double write to the SAME underlying nn.Parameter:
        # Module.load_state_dict recurses through the model's OWN module tree
        # in registration order (`self.encoder` is assigned before
        # `self.decoder` in StationMAE.__init__), so `encoder.<name>.*` is
        # copied in first and `decoder.<name>.*` is copied in second,
        # SILENTLY OVERWRITING it. The encoder's trained position/topography/
        # time understanding is discarded and replaced with the decoder's —
        # for every window, at every station — with no error, no warning, and
        # shapes that match perfectly.
        #
        # Detected by comparing the CHECKPOINT's own two copies: a
        # current-code checkpoint has bit-identical values at both paths (one
        # shared tensor was saved via two attribute names); a pre-fix
        # checkpoint does not, to the precision of two independently
        # optimised parameter sets.
        _shared_mismatch = sorted({
            _name
            for _name in ("pos_emb", "station_emb", "temporal_emb")
            for _ek in state_dict
            if _ek.startswith(f"encoder.{_name}.")
            and (_dk := "decoder." + _ek[len("encoder."):]) in state_dict
            and not torch.equal(state_dict[_ek], state_dict[_dk])
        })

        saved_cfg_for_build = saved_cfg

        _use_nll = any(k.endswith("decoder.log_var_head.weight") or
                       k.endswith("decoder.log_var_head.bias")
                       for k in state_dict)
        print(f"  NLL uncertainty head: {'FOUND — σ will be predicted and saved' if _use_nll else 'absent (point predictions only)'}")

        # Build model
        from model.mae import StationMAE
        # ── Build from the checkpoint's own cfg ──────────────────────────
        # StationMAE._CFG_TO_ARG is the single table mapping saved cfg keys to
        # constructor arguments, shared with main.py. This file used to carry a
        # second, independently maintained list; it drifted, and every v18+
        # checkpoint silently rebuilt with pre-v18 defaults as a result.
        #
        # Overrides win over the checkpoint: dropout is always 0 at inference,
        # use_nll_loss is inferred from the WEIGHTS (the cfg key is unreliable
        # on v9-era runs), and any CLI flag the user set explicitly wins over
        # what the checkpoint recorded.
        _overrides = {"dropout": 0.0, "use_nll_loss": _use_nll}
        for _cli, _arg in (
            (args.d_model,            "d_model"),
            (args.enc_heads,          "enc_heads"),
            (args.enc_layers,         "enc_layers"),
            (args.dec_heads,          "dec_heads"),
            (args.dec_layers,         "dec_layers"),
            (args.mlp_ratio,          "mlp_ratio"),
            (args.mask_ratio,         "mask_ratio"),
            (args.factorised_encoder, "factorised_encoder"),
            (args.temporal_window,    "temporal_window"),
            (args.temporal_patch,     "temporal_patch"),
            (args.cross_attn_decoder, "cross_attn_decoder"),
        ):
            if _cli is not None:
                _overrides[StationMAE._CFG_TO_ARG.get(_arg, _arg)] = _cli
        model = StationMAE.from_cfg(saved_cfg_for_build, **_overrides).to(device)

        # ── Legacy-checkpoint compatibility: un-share what this checkpoint
        # never shared ────────────────────────────────────────────────────
        # Rather than refusing to load a pre-shared-embedding checkpoint,
        # give the DECODER its own fresh, separate copy of each mismatched
        # module. Both `encoder.<name>.*` and `decoder.<name>.*` keys then
        # map to genuinely distinct nn.Parameters, so load_state_dict below
        # populates each from the checkpoint's own two independently-trained
        # copies instead of colliding. Exact shapes are guaranteed to match:
        # d_model comes from this same checkpoint's cfg, and the Fourier
        # dims (position/station/temporal) are architecture constants that
        # have not changed — only TemporalEmbedding's WAVELENGTH VALUES have,
        # and those live in its `lambdas` buffer, which load_state_dict
        # overwrites from the checkpoint just like any other tensor, so the
        # freshly-built instance ends up with the checkpoint's original
        # wavelengths, not today's defaults.
        if _shared_mismatch:
            from model.embeddings import (
                PositionalEmbedding, StationEmbedding, TemporalEmbedding,
                POSITION_FOURIER_DIM, STATION_CHAR_DIM, TEMPORAL_FOURIER_DIM,
            )
            print(f"  [compat] legacy checkpoint predates shared embeddings "
                  f"{_shared_mismatch} — rebuilding the decoder's copies as "
                  f"UNSHARED so both trained versions load intact.")
            if "pos_emb" in _shared_mismatch:
                model.decoder.pos_emb = PositionalEmbedding(
                    d_model=c_d_model, fourier_dim=POSITION_FOURIER_DIM).to(device)
            if "station_emb" in _shared_mismatch:
                model.decoder.station_emb = StationEmbedding(
                    d_model=c_d_model, input_dim=STATION_CHAR_DIM).to(device)
            if "temporal_emb" in _shared_mismatch:
                model.decoder.temporal_emb = TemporalEmbedding(
                    d_model=c_d_model, fourier_dim=TEMPORAL_FOURIER_DIM).to(device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (using default init): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")

        # ── Architecture mismatch guard ───────────────────────────────────
        # strict=False is deliberate (it lets an NLL checkpoint load into a
        # point-prediction model and vice versa), but it will just as happily
        # drop a whole STRUCTURAL module. If the checkpoint was trained with
        # temporal patching and cfg failed to carry temporal_patch through,
        # the rebuilt model has no patch_merge: every patch_merge weight lands
        # in `unexpected`, the encoder silently runs unpatched on 6x the
        # tokens, and the predictions are garbage produced very slowly.
        # That is a wrong result, not just a slow one, so refuse to continue.
        _structural = ("patch_merge", "patch_norm", "var_proj",
                       "input_cross_attn",   # <-- the one that was silently dropped
                       "encoder.blocks", "decoder.blocks")
        _bad = [k for k in list(missing) + list(unexpected)
                if any(s in k for s in _structural)]
        if _bad:
            raise SystemExit(
                "\n[ABORT] structural weights did not match the checkpoint:\n"
                + "".join(f"    {k}\n" for k in _bad[:12])
                + (f"    ... and {len(_bad)-12} more\n" if len(_bad) > 12 else "")
                + "\n  The rebuilt architecture differs from the trained one — most\n"
                  "  likely temporal_patch / temporal_window / d_model / layer counts\n"
                  "  were not read from the checkpoint cfg. Check the banner above\n"
                  "  against the training run, or pass the values explicitly.\n"
                  "  Evaluating anyway would produce meaningless predictions."
            )
        print(f"  Parameters: {model.count_parameters():,}")

        from engine.evaluate import (
            evaluate_full, print_full_metrics,
            evaluate_gap_filling, print_gap_filling_metrics,
        )
        import time as _time

        base_label = os.path.splitext(os.path.basename(ckpt_path))[0]

        # ── Determine which mask ratios to sweep ─────────────────────────
        # Default: use the trained mask_ratio from the checkpoint.
        # CLI --test_mask_ratios overrides, enabling e.g. 0.0 vs 0.5 comparison.
        _test_mrs = (args.test_mask_ratios if args.test_mask_ratios
                     else [model.mask_ratio])

        # A direct-head model reads each station's prediction from that
        # station's OWN tokens. A masked station has no tokens in the sequence,
        # so there is nothing to read from and the readout would silently take
        # a different station's row. Drop any non-zero ratio rather than dump
        # predictions that look plausible and are misaligned.
        if model.direct_head:
            _dropped = [mr for mr in _test_mrs if mr > 0.0]
            _test_mrs = [mr for mr in _test_mrs if mr == 0.0] or [0.0]
            if _dropped:
                print(f"  [direct_head] skipping mask ratios {_dropped} — the "
                      f"readout needs every station present. Evaluating "
                      f"{_test_mrs} only.")

        # Summary table across mask ratios (printed at the end of this checkpoint)
        _mr_summary: list[dict] = []

        for _test_mr in _test_mrs:
            # Override the encoder mask ratio for this evaluation pass
            model.encoder.mask_ratio = _test_mr
            model.eval()

            _mr_tag   = f"mr{_test_mr:.2f}"
            label     = f"{base_label}_{_mr_tag}"
            mr_save   = os.path.join(args.save_dir, label)
            os.makedirs(mr_save, exist_ok=True)

            print(f"\n{'─'*60}")
            print(f"  mask_ratio={_test_mr:.2f}  "
                  f"({'trained setting' if _test_mr == c_mask_ratio else 'override'})  "
                  f"→ save_dir: {mr_save}")

            # ── Predictions-only mode: one forward pass, dump, next ratio ──
            if args.predictions_only:
                from engine.evaluate import collect_predictions
                t0 = _time.time()
                _n = args.save_predictions if args.save_predictions > 0 else len(test_ds)
                print(f"  [predictions-only] Dumping {_n:,} windows "
                      f"(index_mode={args.index_mode}, stride={args.stride}) …")
                _pred_path = os.path.join(mr_save, "predictions.pt")
                _res = collect_predictions(
                    model, test_loader, device,
                    n_windows=_n,
                    save_path=_pred_path,
                )
                print(f"  ⏱  {_time.time()-t0:.0f}s")
                _prediction_sanity_check(_res, label)
                del _res
                continue

            skill:         dict = {}
            lead1_metrics: dict = {}
            gap_metrics:   dict = {}

            if _is_inpainting:
                t0 = _time.time()
                print("  Running evaluate_full() [all stations, delta=0] …")
                metrics = evaluate_full(model, test_loader, device, obs_stats=obs_stats)
                print_full_metrics(metrics, obs_stats=obs_stats)

                if _test_mr > 0:
                    print(f"\n  Running evaluate_gap_filling() "
                          f"[masked only · n_repeats={args.gap_fill_repeats}] …")
                    gap_metrics = evaluate_gap_filling(
                        model, test_loader, device,
                        obs_stats=obs_stats,
                        n_repeats=args.gap_fill_repeats,
                    )
                    print("\n── Spatial inpainting · masked stations only (PRIMARY) ──────────")
                    print_gap_filling_metrics(gap_metrics, obs_stats=obs_stats)
                else:
                    print("  [skip] gap-filling — mask_ratio=0.0, no hidden stations")
                print(f"  ⏱  {_time.time()-t0:.0f}s")

            else:
                # ── Full delta sweep ────────────────────────────────────
                t0 = _time.time()
                print("  Running evaluate_full() [all deltas, all stations] …")
                metrics = evaluate_full(model, test_loader, device, obs_stats=obs_stats)
                skill   = compute_skill_scores(metrics, persist_metrics)
                print_full_metrics(metrics, obs_stats=obs_stats)
                print(f"  ⏱  evaluate_full: {_time.time()-t0:.0f}s")

                print("── Skill scores vs persistence ──────────────────────────────────")
                for var_name in TARGET_VARIABLE_NAMES:
                    s = skill.get(f"skill_{var_name}", float("nan"))
                    bar_len = max(0, int(s * 20)) if not (s != s) else 0
                    print(f"  {var_name:<14}  {s:+.3f}  {'█' * bar_len}")
                print(f"  {'[overall]':<14}  {skill.get('skill_overall', float('nan')):+.3f}")

                # ── Lead-1 ─────────────────────────────────────────────
                t0 = _time.time()
                _lead_min = delta_grid_stride * 10
                print(f"\n  Running evaluate_full() [lead-{_lead_min}min forecast] …")
                lead1_metrics = evaluate_full(model, lead1_loader, device, obs_stats=obs_stats)
                print(f"\n── Lead-{_lead_min}min forecast · all stations ─────────────────────────")
                print_full_metrics(lead1_metrics, obs_stats=obs_stats)
                print(f"  ⏱  lead-{_lead_min}min: {_time.time()-t0:.0f}s")

                # ── Gap-filling (skip when mask_ratio=0) ───────────────
                if _test_mr > 0:
                    t0 = _time.time()
                    print(f"\n  Running evaluate_gap_filling() "
                          f"[delta=0, masked only · n_repeats={args.gap_fill_repeats}] …")
                    gap_metrics = evaluate_gap_filling(
                        model, test_loader, device,
                        obs_stats=obs_stats,
                        n_repeats=args.gap_fill_repeats,
                    )
                    print("\n── Spatial gap-filling · masked stations only ───────────────────")
                    print_gap_filling_metrics(gap_metrics, obs_stats=obs_stats)
                    print(f"  ⏱  gap-filling: {_time.time()-t0:.0f}s")
                else:
                    print("\n  [skip] gap-filling — mask_ratio=0.0, all stations visible")

            from engine.evaluate import (
                build_delta_variable_matrix,
                collect_predictions,
                compute_seasonal_metrics,
                evaluate_per_station,
            )

            # ── Variable × delta RMSE matrix ────────────────────────────
            _mat = build_delta_variable_matrix(metrics)
            if _mat:
                import pandas as pd
                _mat_df = pd.DataFrame(_mat).T   # rows=variables, cols=lead times
                _mat_path = os.path.join(mr_save, "delta_variable_rmse_matrix.csv")
                _mat_df.to_csv(_mat_path)
                print(f"\n── Variable × lead-time RMSE matrix ──────────────────────────")
                print(_mat_df.to_string(float_format=lambda x: f"{x:.4f}"))
                print(f"  Saved: {_mat_path}")

            # ── Per-station metrics (for geographic map plots) ──────────
            t0 = _time.time()
            print("\n  Computing per-station metrics (for map plots) ...")
            _per_station = evaluate_per_station(
                model, test_loader, device,
                test_ds.spatial.cpu(),
                obs_stats=obs_stats,
            )
            _ps_df = pd.DataFrame(_per_station)
            _ps_path = os.path.join(mr_save, "per_station_metrics.csv")
            _ps_df.to_csv(_ps_path, index=False)
            print(f"  Saved {len(_per_station)} stations \u2192 {_ps_path}  \u23f1  {_time.time()-t0:.0f}s")

            # ── Save raw predictions for notebook ────────────────────────
            if args.save_predictions > 0:
                t0 = _time.time()
                print(f"\n  Saving {args.save_predictions} raw prediction windows …")
                _pred_path = os.path.join(mr_save, "predictions.pt")
                collect_predictions(
                    model, test_loader, device,
                    n_windows=args.save_predictions,
                    save_path=_pred_path,
                )
                print(f"  ⏱  predictions: {_time.time()-t0:.0f}s")

            # ── Seasonal metrics ─────────────────────────────────────────
            if args.seasonal:
                t0 = _time.time()
                print("\n  Computing seasonal metrics (DJF / MAM / JJA / SON) …")
                _seas = compute_seasonal_metrics(model, test_loader, device,
                                                 obs_stats=obs_stats)
                print(f"\n── Seasonal RMSE (normalised) ──────────────────────────────────")
                print(f"  {'Season':<6}  {'N':>6}  " +
                      "  ".join(f"{v[:4]:>6}" for v in TARGET_VARIABLE_NAMES) +
                      "  overall")
                print("  " + "-" * 62)
                for s in ["DJF", "MAM", "JJA", "SON"]:
                    sm = _seas.get(s, {})
                    n  = int(sm.get("n_samples", 0))
                    vals = "  ".join(
                        f"{sm.get(f'{v}_rmse_norm', float('nan')):>6.4f}"
                        for v in TARGET_VARIABLE_NAMES
                    )
                    ov = sm.get("overall_rmse_norm", float("nan"))
                    print(f"  {s:<6}  {n:>6,}  {vals}  {ov:.4f}")

                # Save seasonal CSV
                import pandas as pd
                _seas_df = pd.DataFrame(_seas).T
                _seas_path = os.path.join(mr_save, "seasonal_metrics.csv")
                _seas_df.to_csv(_seas_path)
                print(f"  Saved: {_seas_path}  ⏱  {_time.time()-t0:.0f}s")

            # Accumulate for summary table and CSV
            all_results.append({
                "label":   label,
                "path":    ckpt_path,
                "epoch":   ckpt_epoch,
                "metrics": metrics,
                "skill":   skill,
            })
            if lead1_metrics:
                all_lead1_results.append({"label": label, "metrics": lead1_metrics})
            if gap_metrics:
                all_gap_results.append({"label": label, "metrics": gap_metrics})

            _mr_summary.append({
                "mask_ratio":       _test_mr,
                "overall_rmse":     metrics.get("overall_rmse_norm", float("nan")),
                "temperature_rmse": metrics.get("temperature_norm_rmse", float("nan")),
                "wind_u_rmse":      metrics.get("wind_u_norm_rmse", float("nan")),
                "wind_v_rmse":      metrics.get("wind_v_norm_rmse", float("nan")),
                "skill_overall":    skill.get("skill_overall", float("nan")),
                "gap_overall":      gap_metrics.get("gap_overall_rmse_norm", float("nan")),
            })

        # ── Mask-ratio comparison summary ────────────────────────────────
        if len(_mr_summary) > 1:
            print(f"\n{'='*60}")
            print(f"  Mask-ratio sweep summary — {base_label}")
            print(f"  {'mask_ratio':>10}  {'overall_rmse':>12}  {'skill':>7}  "
                  f"{'temp_rmse':>10}  {'gap_rmse':>10}")
            print("  " + "-" * 56)
            for r in _mr_summary:
                gap_str = f"{r['gap_overall']:>10.5f}" if r['gap_overall'] == r['gap_overall'] else f"{'—':>10}"
                print(f"  {r['mask_ratio']:>10.2f}  {r['overall_rmse']:>12.5f}  "
                      f"{r['skill_overall']:>+7.3f}  {r['temperature_rmse']:>10.5f}  {gap_str}")
            print()

        # ── WandB logging (optional; no metrics exist in predictions-only) ─
        if args.wandb_project and not args.predictions_only:
            try:
                import wandb
                # Use last evaluated mask ratio for WandB label
                _last_mr = _mr_summary[-1]["mask_ratio"] if _mr_summary else c_mask_ratio
                run_name = args.wandb_run_name or f"test-{base_label}"
                if len(_test_mrs) > 1:
                    run_name = f"{run_name}-sweep"
                _wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    job_type="evaluation",
                    config={
                        "checkpoint":         ckpt_path,
                        "epoch":              ckpt_epoch,
                        "mode":               "inpainting" if _is_inpainting else "forecasting",
                        "window":             window,
                        "max_delta":          max_delta,
                        "trained_mask_ratio": c_mask_ratio,
                        "test_mask_ratios":   _test_mrs,
                        "index_mode":         args.index_mode,
                        "gap_fill_repeats":   args.gap_fill_repeats,
                        "factorised_encoder": factorised,
                        "cross_attn_decoder": cross_attn,
                    },
                    reinit=True,
                )

                summary_flat: dict[str, float] = {}

                if _is_inpainting:
                    # Primary metric: gap-filling on masked stations
                    for k, v in gap_metrics.items():
                        summary_flat[f"gap/{k}"] = v
                    for k, v in metrics.items():
                        summary_flat[f"eval/{k}"] = v
                else:
                    # Per-delta RMSE curve
                    delta_keys = sorted(
                        k for k in metrics
                        if k.startswith("delta_") and k.endswith("_overall_rmse_norm")
                    )
                    for dk in delta_keys:
                        d    = int(dk.split("_")[1])
                        rmse = metrics[dk]
                        mae  = metrics.get(f"delta_{d:02d}_overall_mae_norm", float("nan"))
                        wandb.log({"delta_min": d * 10,
                                   "eval/rmse_norm": rmse,
                                   "eval/mae_norm":  mae})
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

    if args.predictions_only:
        print(f"\n[predictions-only] Done — predictions.pt written per mask ratio "
              f"under: {os.path.abspath(args.save_dir)}")
        print("Compute metrics downstream in Test_Results_Exploration.ipynb.")
        return

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
        gap_filename = ("inpainting_metrics.csv" if _is_inpainting
                        else "gap_filling_metrics.csv")
        gap_csv = save_extra_metrics_csv(
            all_gap_results, args.save_dir, gap_filename
        )
        print(f"{'Inpainting' if _is_inpainting else 'Gap-filling'} metrics saved to: {gap_csv}")

    if not _is_inpainting and persist_metrics:
        persist_csv = os.path.join(args.save_dir, "persistence_metrics.csv")
        with open(persist_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in sorted(persist_metrics.items()):
                writer.writerow([k, f"{v:.6f}"])
        print(f"Persistence metrics saved to: {persist_csv}")

    # ── Plots (forecasting mode only — delta RMSE curve needs lead-times) ─
    if not args.no_plots and not _is_inpainting:
        print("\nGenerating plots …")
        plot_delta_rmse(all_results, persist_metrics, args.save_dir)

    print(f"\nAll results written to: {os.path.abspath(args.save_dir)}")
    print("Done.")


if __name__ == "__main__":
    main()
