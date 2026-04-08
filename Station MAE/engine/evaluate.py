"""
engine/evaluate.py

Evaluation loop for Station-MAE.

Public API
----------
    evaluate(model, loader, device)
        → metrics: dict[str, float]
        Fast loop used during training — normalised RMSE/MAE only.

    evaluate_full(model, loader, device, obs_stats=None)
        → metrics: dict[str, float]
        Full test-set evaluation with physical units, bias (MBE), R²,
        wind-speed/direction metrics, and per-lead-time breakdown.

evaluate() — metrics
--------------------
    avg_loss              — MSE (same scale as training loss)
    {var}_rmse / {var}_mae — per variable, normalised space

evaluate_full() — additional metrics
-------------------------------------
    Normalised space  (suffix _norm)
        {var}_rmse_norm, {var}_mae_norm, {var}_bias_norm, {var}_r2

    Physical space  (only if obs_stats provided; suffix = physical unit)
        {var}_rmse, {var}_mae, {var}_bias   — in °C / hPa / % / m/s
        wind_speed_rmse, wind_speed_mae, wind_speed_bias   — m/s
        wind_dir_mae                                       — degrees

    Per lead-time  (keys "delta_{d}_overall_rmse_norm" for d=1..max_delta)
        Groups predictions by the delta_steps value in each batch item.

    Overall
        overall_rmse_norm, overall_mae_norm
        overall_rmse, overall_mae  (physical, if obs_stats provided)

Units per predicted variable
----------------------------
    temperature  → °C
    pressure     → hPa
    humidity     → %
    wind_u       → m/s  (eastward component)
    wind_v       → m/s  (northward component)
    wind_speed   → m/s  (derived: √(u² + v²), denormalised)
    wind_dir     → °    (derived: atan2(u, v), circular MAE)

Only masked stations are evaluated to match the training objective and avoid
the model trivially repeating its own inputs.  Within each masked station,
only sensors present at the target timestep (y_mask == 1) count.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.embeddings import TARGET_VARIABLE_NAMES, NUM_TARGET_VARIABLES


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
                 Keys: "avg_loss" (MSE, same scale as train loss),
                       "{var}_rmse", "{var}_mae", "overall_rmse", "overall_mae".
    """
    model.eval()

    # Accumulate (masked_prediction, masked_target, sensor_mask) per variable
    preds_list   = []   # list of (n_masked, V) tensors
    targets_list = []
    masks_list   = []   # sensor presence at target step

    total_loss = 0.0   # sum of per-batch MSE losses (same objective as training)
    n_batches  = 0

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
            spatial = spatial[0]   # (N, 15)

        # ---- Forward (includes random masking in encoder) ----------
        loss, preds, masked_idx = model(
            x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
        )
        # loss:       scalar MSE (same objective as training)
        # preds:      (B, N, V)
        # masked_idx: (B, N_masked)

        total_loss += loss.item()
        n_batches  += 1

        B = preds.size(0)

        # Slice y and y_mask to target variables only — drops precipitation (last col).
        # preds is already (B, N, NUM_TARGET_VARIABLES) from the decoder head;
        # y and y_mask from the batch are still (B, N, 6) so they must match.
        y_target      = y[:, :, :NUM_TARGET_VARIABLES]       # (B, N, 5)
        y_mask_target = y_mask[:, :, :NUM_TARGET_VARIABLES]  # (B, N, 5)

        # ---- Gather masked-station predictions per sample ----------
        for b in range(B):
            m_idx = masked_idx[b]                                    # (N_masked,)
            preds_list.append(preds[b, m_idx].cpu())                 # (N_masked, 5)
            targets_list.append(y_target[b, m_idx].cpu())            # (N_masked, 5)
            masks_list.append(y_mask_target[b, m_idx].cpu())         # (N_masked, 5)

    # ---- Concatenate across all batches ----------------------------
    preds_all   = torch.cat(preds_list,   dim=0)    # (total_masked, 5)
    targets_all = torch.cat(targets_list, dim=0)    # (total_masked, 5)
    masks_all   = torch.cat(masks_list,   dim=0).bool()  # (total_masked, 5)

    # ---- Per-variable metrics (present sensors only) ---------------
    metrics: dict[str, float] = {
        "avg_loss": total_loss / max(n_batches, 1),   # MSE, same scale as train_loss
    }

    for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
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
# Full test-set evaluation
# ---------------------------------------------------------------------------

# Physical units for display and CSV headers
_VAR_UNITS = {
    "temperature": "°C",
    "pressure":    "hPa",
    "humidity":    "%",
    "wind_u":      "m/s",
    "wind_v":      "m/s",
}
_IDX = {v: i for i, v in enumerate(TARGET_VARIABLE_NAMES)}


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination R²."""
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum()
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _wind_dir_deg(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Meteorological wind direction in [0, 360): direction FROM which wind blows."""
    # Meteorological convention: 0° = N (wind from north), 90° = E, etc.
    # dir_from = atan2(-u, -v) in radians, shifted to [0, 2π)
    rad = torch.atan2(-u, -v)
    deg = torch.rad2deg(rad)
    return deg % 360.0


def _circular_mae_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> float:
    """Mean absolute error on a circular quantity (degrees)."""
    diff = pred_deg - true_deg
    # Wrap to [-180, 180]
    diff = (diff + 180.0) % 360.0 - 180.0
    return float(diff.abs().mean().item())


@torch.no_grad()
def evaluate_full(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    obs_stats: "dict | None" = None,
) -> dict[str, float]:
    """
    Full test-set evaluation with physical units, bias, R², wind speed/direction
    metrics, and per-lead-time breakdown.

    Args:
        model:      StationMAE moved to ``device``.
        loader:     DataLoader from StationMAEDataset (num_delta_per_sample=1
                    recommended; multi-delta is also handled — first delta used).
        device:     torch.device.
        obs_stats:  dict with keys ``"mean"`` and ``"std"`` (both tensors of
                    shape (NUM_VARIABLES,) = (6,)).  Pass ``train_ds.obs_stats``
                    to get physical-unit metrics.  If None, only normalised
                    metrics are returned.

    Returns:
        metrics: dict[str, float] — see module docstring for key listing.
    """
    model.eval()

    # Storage per sample: list of dicts so we can group by delta later
    records = []    # each entry: {"pred": (n_m,5), "target": (n_m,5),
                    #              "mask": (n_m,5), "delta": int}

    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        x           = batch["x"].to(device)
        x_mask      = batch["x_mask"].to(device)
        spatial     = batch["spatial"].to(device)
        x_hours     = batch["x_hours"].to(device)
        y_raw       = batch["y"].to(device)
        y_mask_raw  = batch["y_mask"].to(device)
        y_hours     = batch["y_hours"].to(device)
        delta_steps = batch["delta_steps"].to(device)

        if spatial.dim() == 3 and spatial.size(0) == x.size(0):
            spatial = spatial[0]

        # Multi-delta batches: take only the first lead-time slice for evaluation
        # (consistent with single-delta val_loader; avoids double-counting)
        if y_raw.dim() == 4:          # (B, K, N, V) → use k=0
            y_raw      = y_raw[:, 0]
            y_mask_raw = y_mask_raw[:, 0]
            y_hours    = y_hours[:, 0]
            delta_steps = delta_steps[:, 0]

        loss, preds, masked_idx = model(
            x, x_mask, spatial, x_hours,
            y_raw, y_mask_raw, y_hours, delta_steps,
        )
        total_loss += loss.item()
        n_batches  += 1

        y_target      = y_raw[:, :, :NUM_TARGET_VARIABLES]       # (B, N, 5)
        y_mask_target = y_mask_raw[:, :, :NUM_TARGET_VARIABLES]  # (B, N, 5)

        B = preds.size(0)
        for b in range(B):
            m_idx = masked_idx[b]
            delta = int(delta_steps[b].item())
            records.append({
                "pred":   preds[b, m_idx].cpu(),             # (n_m, 5)
                "target": y_target[b, m_idx].cpu(),          # (n_m, 5)
                "mask":   y_mask_target[b, m_idx].cpu(),     # (n_m, 5) bool
                "delta":  delta,
            })

    # ── Concatenate all samples ─────────────────────────────────────────
    preds_all   = torch.cat([r["pred"]   for r in records], dim=0)   # (M, 5)
    targets_all = torch.cat([r["target"] for r in records], dim=0)
    masks_all   = torch.cat([r["mask"]   for r in records], dim=0).bool()

    metrics: dict[str, float] = {
        "avg_loss": total_loss / max(n_batches, 1),
    }

    # ── Helper: compute scalar metrics for one (pred, target, mask) slice ──
    def _scalars(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor,
                 prefix: str, std_scale: float = 1.0) -> None:
        """Write rmse, mae, bias, r2 into metrics dict under `prefix`."""
        if m.sum() == 0:
            for k in ("rmse", "mae", "bias", "r2"):
                metrics[f"{prefix}_{k}"] = float("nan")
            return
        pv = p[m] * std_scale
        tv = t[m] * std_scale
        err = pv - tv
        metrics[f"{prefix}_rmse"] = float(err.pow(2).mean().sqrt().item())
        metrics[f"{prefix}_mae"]  = float(err.abs().mean().item())
        metrics[f"{prefix}_bias"] = float(err.mean().item())           # MBE
        metrics[f"{prefix}_r2"]   = _r2(pv, tv)

    # ── Normalised per-variable metrics ────────────────────────────────
    for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
        _scalars(
            preds_all[:, v], targets_all[:, v], masks_all[:, v],
            prefix=f"{var_name}_norm",
        )

    # ── Overall normalised ─────────────────────────────────────────────
    if masks_all.sum() > 0:
        p_flat = preds_all[masks_all]
        t_flat = targets_all[masks_all]
        err_flat = p_flat - t_flat
        metrics["overall_rmse_norm"] = float(err_flat.pow(2).mean().sqrt().item())
        metrics["overall_mae_norm"]  = float(err_flat.abs().mean().item())
        metrics["overall_bias_norm"] = float(err_flat.mean().item())
    else:
        metrics["overall_rmse_norm"] = float("nan")
        metrics["overall_mae_norm"]  = float("nan")
        metrics["overall_bias_norm"] = float("nan")

    # ── Physical-unit metrics (requires obs_stats) ─────────────────────
    if obs_stats is not None:
        mean_t = obs_stats["mean"].cpu()   # (6,) all variables
        std_t  = obs_stats["std"].cpu()    # (6,) all variables

        # Per-variable in physical units
        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            std_v = float(std_t[v].item())
            _scalars(
                preds_all[:, v], targets_all[:, v], masks_all[:, v],
                prefix=var_name,
                std_scale=std_v,   # error in physical units
            )

        # Overall physical
        if masks_all.sum() > 0:
            # Rescale each variable by its std before pooling (mix of units → skip)
            # Instead report in normalised space for overall — already done above.
            # Repeat with physical for display convenience (mean over std-scaled errs)
            errs_phys = []
            for v in range(NUM_TARGET_VARIABLES):
                std_v = float(std_t[v].item())
                m = masks_all[:, v]
                if m.sum() > 0:
                    errs_phys.append(
                        ((preds_all[m, v] - targets_all[m, v]) * std_v)
                    )
            if errs_phys:
                e = torch.cat(errs_phys)
                metrics["overall_rmse"] = float(e.pow(2).mean().sqrt().item())
                metrics["overall_mae"]  = float(e.abs().mean().item())
                metrics["overall_bias"] = float(e.mean().item())

        # Wind-speed RMSE/MAE (denormalise u and v first)
        ui = _IDX.get("wind_u")
        vi = _IDX.get("wind_v")
        if ui is not None and vi is not None:
            mean_u = float(mean_t[ui].item()); std_u = float(std_t[ui].item())
            mean_v = float(mean_t[vi].item()); std_v_w = float(std_t[vi].item())
            m_uv = masks_all[:, ui] & masks_all[:, vi]
            if m_uv.sum() > 0:
                u_pred = preds_all[m_uv, ui]   * std_u + mean_u
                u_true = targets_all[m_uv, ui] * std_u + mean_u
                v_pred = preds_all[m_uv, vi]   * std_v_w + mean_v
                v_true = targets_all[m_uv, vi] * std_v_w + mean_v

                ws_pred = (u_pred.pow(2) + v_pred.pow(2)).sqrt()
                ws_true = (u_true.pow(2) + v_true.pow(2)).sqrt()
                ws_err  = ws_pred - ws_true

                metrics["wind_speed_rmse"] = float(ws_err.pow(2).mean().sqrt().item())
                metrics["wind_speed_mae"]  = float(ws_err.abs().mean().item())
                metrics["wind_speed_bias"] = float(ws_err.mean().item())

                # Wind direction MAE (circular)
                wd_pred = _wind_dir_deg(u_pred, v_pred)
                wd_true = _wind_dir_deg(u_true, v_true)
                metrics["wind_dir_mae_deg"] = _circular_mae_deg(wd_pred, wd_true)
            else:
                for k in ("wind_speed_rmse", "wind_speed_mae",
                          "wind_speed_bias", "wind_dir_mae_deg"):
                    metrics[k] = float("nan")

    # ── Per-delta-step breakdown ────────────────────────────────────────
    unique_deltas = sorted({r["delta"] for r in records})
    for d in unique_deltas:
        recs_d = [r for r in records if r["delta"] == d]
        if not recs_d:
            continue
        p_d = torch.cat([r["pred"]   for r in recs_d], dim=0)
        t_d = torch.cat([r["target"] for r in recs_d], dim=0)
        m_d = torch.cat([r["mask"]   for r in recs_d], dim=0).bool()

        if m_d.sum() == 0:
            metrics[f"delta_{d:02d}_overall_rmse_norm"] = float("nan")
            continue

        err_d = p_d[m_d] - t_d[m_d]
        metrics[f"delta_{d:02d}_overall_rmse_norm"] = float(
            err_d.pow(2).mean().sqrt().item()
        )
        metrics[f"delta_{d:02d}_overall_mae_norm"] = float(
            err_d.abs().mean().item()
        )
        metrics[f"delta_{d:02d}_n_samples"] = len(recs_d)

        # Per-variable RMSE at this delta
        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            mv = m_d[:, v] if m_d.dim() == 2 else m_d
            # m_d is (M_d, 5) after masking — already bool tensor of (total_masked_d, 5)
            # Need per-column mask
            m_d_v = torch.cat([r["mask"][:, v] for r in recs_d], dim=0).bool()
            if m_d_v.sum() == 0:
                metrics[f"delta_{d:02d}_{var_name}_rmse_norm"] = float("nan")
                continue
            p_dv = torch.cat([r["pred"][:, v]   for r in recs_d], dim=0)[m_d_v]
            t_dv = torch.cat([r["target"][:, v] for r in recs_d], dim=0)[m_d_v]
            metrics[f"delta_{d:02d}_{var_name}_rmse_norm"] = float(
                (p_dv - t_dv).pow(2).mean().sqrt().item()
            )

    return metrics


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict[str, float]) -> None:
    """Print a formatted table of evaluation metrics (used during training)."""
    col_w = max(len(k) for k in metrics) + 2
    print(f"\n{'Metric':<{col_w}}  {'Value':>10}")
    print("-" * (col_w + 14))
    for k, v in sorted(metrics.items()):
        print(f"{k:<{col_w}}  {v:>10.5f}")
    print()


def print_full_metrics(metrics: dict[str, float], obs_stats: "dict | None" = None) -> None:
    """
    Pretty-print evaluate_full() results in three sections:
      1. Per-variable summary (physical units if obs_stats provided, else normalised)
      2. Wind speed + direction
      3. Per-delta-step RMSE table
    """
    has_phys = obs_stats is not None
    unit_map = _VAR_UNITS if has_phys else {v: "norm" for v in TARGET_VARIABLE_NAMES}

    # ── Section 1: per-variable ─────────────────────────────────────────
    print("\n── Per-variable metrics (masked stations) " + "─" * 30)
    hdr = f"  {'Variable':<14}  {'RMSE':>8}  {'MAE':>8}  {'Bias':>8}  {'R²':>6}  Unit"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for var_name in TARGET_VARIABLE_NAMES:
        pfx  = var_name if (has_phys and f"{var_name}_rmse" in metrics) else f"{var_name}_norm"
        rmse = metrics.get(f"{pfx}_rmse", float("nan"))
        mae  = metrics.get(f"{pfx}_mae",  float("nan"))
        bias = metrics.get(f"{pfx}_bias", float("nan"))
        r2   = metrics.get(f"{pfx}_r2",   float("nan"))
        unit = unit_map.get(var_name, "norm")
        print(f"  {var_name:<14}  {rmse:>8.4f}  {mae:>8.4f}  {bias:>+8.4f}  {r2:>6.3f}  {unit}")

    # Overall
    pfx_ov = "overall" if (has_phys and "overall_rmse" in metrics) else "overall"
    rmse_ov = metrics.get(f"{pfx_ov}_rmse", metrics.get("overall_rmse_norm", float("nan")))
    mae_ov  = metrics.get(f"{pfx_ov}_mae",  metrics.get("overall_mae_norm",  float("nan")))
    bias_ov = metrics.get(f"{pfx_ov}_bias", metrics.get("overall_bias_norm", float("nan")))
    print(f"  {'[overall]':<14}  {rmse_ov:>8.4f}  {mae_ov:>8.4f}  {bias_ov:>+8.4f}  {'':>6}")

    # ── Section 2: wind speed / direction ───────────────────────────────
    if has_phys and "wind_speed_rmse" in metrics:
        print("\n── Wind speed & direction " + "─" * 45)
        print(f"  {'wind_speed':<14}  RMSE={metrics['wind_speed_rmse']:.4f} m/s  "
              f"MAE={metrics['wind_speed_mae']:.4f} m/s  "
              f"Bias={metrics['wind_speed_bias']:+.4f} m/s")
        print(f"  {'wind_dir':<14}  MAE={metrics.get('wind_dir_mae_deg', float('nan')):.2f}°")

    # ── Section 3: per-delta ────────────────────────────────────────────
    delta_keys = sorted(k for k in metrics if k.startswith("delta_") and k.endswith("_overall_rmse_norm"))
    if delta_keys:
        print("\n── RMSE by lead-time (normalised) " + "─" * 37)
        print(f"  {'Lead-time':>10}  {'RMSE':>8}  {'MAE':>8}  {'N samples':>10}")
        print("  " + "-" * 44)
        for dk in delta_keys:
            d    = int(dk.split("_")[1])
            rmse = metrics[dk]
            mae  = metrics.get(f"delta_{d:02d}_overall_mae_norm", float("nan"))
            n    = int(metrics.get(f"delta_{d:02d}_n_samples", 0))
            mins = d * 10
            print(f"  {mins:>7} min  {rmse:>8.5f}  {mae:>8.5f}  {n:>10,}")

    print(f"\n  avg_loss (train objective MSE) = {metrics.get('avg_loss', float('nan')):.6f}")
    if not has_phys:
        print("  (all metrics in normalised space — pass obs_stats for physical units)")
    print()
