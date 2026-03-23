"""
engine/evaluate.py

Evaluation loop for Station-MAE.

Public API
----------
    evaluate(model, loader, device)
        → metrics: dict[str, float]

Returned metrics
----------------
    For each variable v in VARIABLE_NAMES:
        {v}_rmse   — root-mean-square error on masked stations
        {v}_mae    — mean absolute error on masked stations

    Aggregated across all variables:
        overall_rmse
        overall_mae

All metrics are computed in NORMALISED space (same space in which the model
is trained).  Denormalise with `obs_stats["mean"]` and `obs_stats["std"]`
from the dataset if you want physical units.

Only masked stations are evaluated (to match the training objective and avoid
the model trivially predicting its own inputs).  Within each masked station,
only sensors that were present at the target timestep (y_mask == 1) count.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.embeddings import VARIABLE_NAMES


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on one pass through `loader`.

    Args:
        model:   StationMAE (or any nn.Module with the same forward signature).
                 Should already be moved to `device`.
        loader:  DataLoader yielding batches from StationMAEDataset.
        device:  torch.device.

    Returns:
        metrics: dict mapping metric names → float values.
                 Keys: "{var}_rmse", "{var}_mae", "overall_rmse", "overall_mae".
    """
    model.eval()

    # Accumulate (masked_prediction, masked_target, sensor_mask) per variable
    preds_list   = []   # list of (n_masked, V) tensors
    targets_list = []
    masks_list   = []   # sensor presence at target step

    for batch in loader:
        # ---- Move to device ----------------------------------------
        x           = batch["x"].to(device)
        x_mask      = batch["x_mask"].to(device)
        spatial     = batch["spatial"].to(device)
        x_hours     = batch["x_hours"].to(device)
        y           = batch["y"].to(device)
        y_mask      = batch["y_mask"].to(device)
        y_hours     = batch["y_hours"].to(device)
        delta_steps = batch["delta_steps"].to(device)

        if spatial.dim() == 3 and spatial.size(0) == x.size(0):
            spatial = spatial[0]   # (N, 14)

        # ---- Forward (includes random masking in encoder) ----------
        loss, preds, masked_idx = model(
            x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
        )
        # preds:      (B, N, V)
        # masked_idx: (B, N_masked)

        B = preds.size(0)

        # ---- Gather masked-station predictions per sample ----------
        for b in range(B):
            m_idx = masked_idx[b]                           # (N_masked,)
            preds_list.append(preds[b, m_idx].cpu())        # (N_masked, V)
            targets_list.append(y[b, m_idx].cpu())          # (N_masked, V)
            masks_list.append(y_mask[b, m_idx].cpu())       # (N_masked, V)

    # ---- Concatenate across all batches ----------------------------
    preds_all   = torch.cat(preds_list,   dim=0)    # (total_masked, V)
    targets_all = torch.cat(targets_list, dim=0)
    masks_all   = torch.cat(masks_list,   dim=0).bool()

    # ---- Per-variable metrics (present sensors only) ---------------
    metrics: dict[str, float] = {}

    for v, var_name in enumerate(VARIABLE_NAMES):
        m = masks_all[:, v]              # (total_masked,) boolean
        if m.sum() == 0:
            metrics[f"{var_name}_rmse"] = float("nan")
            metrics[f"{var_name}_mae"]  = float("nan")
            continue

        p = preds_all[m, v]
        t = targets_all[m, v]
        metrics[f"{var_name}_rmse"] = float((p - t).pow(2).mean().sqrt().item())
        metrics[f"{var_name}_mae"]  = float((p - t).abs().mean().item())

    # ---- Overall metrics across all variables ----------------------
    valid = masks_all                                       # (total_masked, V)
    if valid.sum() > 0:
        p_flat = preds_all[valid]
        t_flat = targets_all[valid]
        metrics["overall_rmse"] = float((p_flat - t_flat).pow(2).mean().sqrt().item())
        metrics["overall_mae"]  = float((p_flat - t_flat).abs().mean().item())
    else:
        metrics["overall_rmse"] = float("nan")
        metrics["overall_mae"]  = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict[str, float]) -> None:
    """Print a formatted table of evaluation metrics."""
    col_w = max(len(k) for k in metrics) + 2
    print(f"\n{'Metric':<{col_w}}  {'Value':>10}")
    print("-" * (col_w + 14))
    for k, v in sorted(metrics.items()):
        print(f"{k:<{col_w}}  {v:>10.5f}")
    print()
