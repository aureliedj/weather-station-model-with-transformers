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

Metrics are computed over all N stations wherever sensors are present
(y_mask == 1).  See per-function docstrings for details.
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

    Metrics are computed over **all N stations** wherever sensors are present
    at the target timestep (y_mask == 1), consistent with the training loss
    which also supervises all stations.

    Args:
        model:   StationMAE (or any nn.Module with the same forward signature).
                 Should already be moved to `device`.
        loader:  DataLoader yielding batches from StationMAEDataset.
        device:  torch.device.

    Returns:
        metrics: dict — "avg_loss", "{var}_rmse", "{var}_mae",
                        "overall_rmse", "overall_mae".
    """
    model.eval()

    preds_list   = []
    targets_list = []
    masks_list   = []

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

        # Handle both single-delta (B, N, V) and fixed-grid (B, K, N, V)
        if y_raw.dim() == 4:
            K = y_raw.shape[1]
        else:
            K = 1
            y_raw       = y_raw.unsqueeze(1)
            y_mask_raw  = y_mask_raw.unsqueeze(1)
            y_hours     = y_hours.unsqueeze(1)
            delta_steps = delta_steps.unsqueeze(1)

        for k in range(K):
            y_k  = y_raw[:, k];       ym_k = y_mask_raw[:, k]
            yh_k = y_hours[:, k];     ds_k = delta_steps[:, k]

            loss, preds, _ = model(
                x, x_mask, spatial, x_hours, y_k, ym_k, yh_k, ds_k
            )
            total_loss += loss.item()
            n_batches  += 1

            B, N = preds.shape[:2]
            y_target      = y_k[:, :, :NUM_TARGET_VARIABLES]
            y_mask_target = ym_k[:, :, :NUM_TARGET_VARIABLES]

            preds_list.append(preds.reshape(B * N, NUM_TARGET_VARIABLES).cpu())
            targets_list.append(y_target.reshape(B * N, NUM_TARGET_VARIABLES).cpu())
            masks_list.append(y_mask_target.reshape(B * N, NUM_TARGET_VARIABLES).cpu())

    preds_all   = torch.cat(preds_list,   dim=0)            # (total, 5)
    targets_all = torch.cat(targets_list, dim=0)
    masks_all   = torch.cat(masks_list,   dim=0).bool()

    metrics: dict[str, float] = {
        "avg_loss": total_loss / max(n_batches, 1),
    }

    for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
        m = masks_all[:, v]
        if m.sum() == 0:
            metrics[f"{var_name}_rmse"] = float("nan")
            metrics[f"{var_name}_mae"]  = float("nan")
            continue
        p, t = preds_all[m, v], targets_all[m, v]
        metrics[f"{var_name}_rmse"] = float((p - t).pow(2).mean().sqrt().item())
        metrics[f"{var_name}_mae"]  = float((p - t).abs().mean().item())

    if masks_all.sum() > 0:
        p_flat = preds_all[masks_all]
        t_flat = targets_all[masks_all]
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

    # Records are stored per batch for the per-delta breakdown.
    # Each entry covers ALL N stations in the batch (not just masked ones),
    # consistent with the all-station training objective.
    records = []    # each entry: {"pred": (B*N, 5), "target": (B*N, 5),
                    #              "mask": (B*N, 5), "deltas": (B,) int list}

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

        # Use forward_multi_delta when y has K lead-times (B, K, N, V):
        # encoder runs ONCE, decoder runs once for all K → much faster than
        # calling forward() K times separately.
        # Falls back to single-delta forward() for (B, N, V) batches.
        if y_raw.dim() == 4:
            # Multi-delta path — encoder once, decoder once
            loss_k, preds_k, _ = model.forward_multi_delta(
                x, x_mask, spatial, x_hours,
                y_raw, y_mask_raw, y_hours, delta_steps,
            )
            # preds_k: (B, K, N, num_target_vars)
            total_loss += loss_k.item()
            n_batches  += 1

            B, K, N = preds_k.shape[:3]
            for k in range(K):
                y_target      = y_raw[:, k, :, :NUM_TARGET_VARIABLES]
                y_mask_target = y_mask_raw[:, k, :, :NUM_TARGET_VARIABLES]
                records.append({
                    "pred":   preds_k[:, k].reshape(B * N, NUM_TARGET_VARIABLES).cpu(),
                    "target": y_target.reshape(B * N, NUM_TARGET_VARIABLES).cpu(),
                    "mask":   y_mask_target.reshape(B * N, NUM_TARGET_VARIABLES).cpu(),
                    "deltas": delta_steps[:, k].tolist(),
                    "N":      N,
                })
        else:
            # Single-delta path
            loss_k, preds_k, _ = model(
                x, x_mask, spatial, x_hours,
                y_raw, y_mask_raw, y_hours, delta_steps,
            )
            total_loss += loss_k.item()
            n_batches  += 1

            B, N = preds_k.shape[:2]
            y_target      = y_raw[:, :, :NUM_TARGET_VARIABLES]
            y_mask_target = y_mask_raw[:, :, :NUM_TARGET_VARIABLES]
            records.append({
                "pred":   preds_k.reshape(B * N, NUM_TARGET_VARIABLES).cpu(),
                "target": y_target.reshape(B * N, NUM_TARGET_VARIABLES).cpu(),
                "mask":   y_mask_target.reshape(B * N, NUM_TARGET_VARIABLES).cpu(),
                "deltas": delta_steps.tolist(),
                "N":      N,
            })

    # ── Concatenate all batches ─────────────────────────────────────────
    preds_all   = torch.cat([r["pred"]   for r in records], dim=0)
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
        mean_t = obs_stats["mean"].cpu()   # (V,) or (N, V)
        std_t  = obs_stats["std"].cpu()    # (V,) or (N, V)

        # Per-station normalization produces (N, V) stats.
        # Average over stations for a single representative std per variable
        # used in physical-unit display (RMSE_phys ≈ RMSE_norm × mean_std).
        if std_t.dim() == 2:
            std_per_var  = std_t.mean(dim=0)   # (V,)
            mean_per_var = mean_t.mean(dim=0)  # (V,)
        else:
            std_per_var  = std_t
            mean_per_var = mean_t

        # Per-variable in physical units
        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            std_v = float(std_per_var[v].item())
            _scalars(
                preds_all[:, v], targets_all[:, v], masks_all[:, v],
                prefix=var_name,
                std_scale=std_v,
            )

        # Overall physical
        if masks_all.sum() > 0:
            errs_phys = []
            for v in range(NUM_TARGET_VARIABLES):
                std_v = float(std_per_var[v].item())
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
            mean_u  = float(mean_per_var[ui].item())
            std_u   = float(std_per_var[ui].item())
            mean_v  = float(mean_per_var[vi].item())
            std_v_w = float(std_per_var[vi].item())
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
    # Re-expand batch records to per-sample entries for delta grouping.
    # Each batch record covers B samples × N stations; we split by sample
    # so we can filter on the per-sample delta value.
    sample_records: list[dict] = []
    for r in records:
        N_r = r["N"]
        for b, d in enumerate(r["deltas"]):
            sample_records.append({
                "pred":   r["pred"]  [b * N_r : (b + 1) * N_r],    # (N, 5)
                "target": r["target"][b * N_r : (b + 1) * N_r],
                "mask":   r["mask"]  [b * N_r : (b + 1) * N_r],
                "delta":  d,
            })

    unique_deltas = sorted({sr["delta"] for sr in sample_records})
    for d in unique_deltas:
        recs_d = [sr for sr in sample_records if sr["delta"] == d]
        if not recs_d:
            continue

        p_d = torch.cat([sr["pred"]   for sr in recs_d], dim=0)
        t_d = torch.cat([sr["target"] for sr in recs_d], dim=0)
        m_d = torch.cat([sr["mask"]   for sr in recs_d], dim=0).bool()

        if m_d.sum() == 0:
            metrics[f"delta_{d:02d}_overall_rmse_norm"] = float("nan")
            continue

        err_d = p_d[m_d] - t_d[m_d]
        metrics[f"delta_{d:02d}_overall_rmse_norm"] = float(
            err_d.pow(2).mean().sqrt().item()
        )
        metrics[f"delta_{d:02d}_overall_mae_norm"]  = float(err_d.abs().mean().item())
        metrics[f"delta_{d:02d}_n_samples"]         = len(recs_d)

        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            m_d_v = torch.cat([sr["mask"][:, v] for sr in recs_d], dim=0).bool()
            if m_d_v.sum() == 0:
                metrics[f"delta_{d:02d}_{var_name}_rmse_norm"] = float("nan")
                continue
            p_dv = torch.cat([sr["pred"][:, v]   for sr in recs_d], dim=0)[m_d_v]
            t_dv = torch.cat([sr["target"][:, v] for sr in recs_d], dim=0)[m_d_v]
            metrics[f"delta_{d:02d}_{var_name}_rmse_norm"] = float(
                (p_dv - t_dv).pow(2).mean().sqrt().item()
            )

    return metrics


# ---------------------------------------------------------------------------
# Per-station evaluation (for geographic map plots)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_per_station(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    spatial:   "torch.Tensor",
    obs_stats: "dict | None" = None,
) -> "list[dict]":
    """
    Compute per-station MAE and RMSE over all test windows.

    Unlike ``evaluate_full()`` which flattens B\xd7N stations together, this
    function tracks each of the N stations independently so you can later
    plot MAE/RMSE on a geographic map of Switzerland.

    Args:
        model:    StationMAE moved to ``device``, in eval mode.
        loader:   Test DataLoader (fixed-grid or single-delta).
        device:   torch.device.
        spatial:  Station static features, shape (N, 15).  Columns 0 and 1
                  are easting / northing in Swiss LV95 coordinates.
                  Pass ``test_ds.spatial`` directly.
        obs_stats: dict with ``"mean"`` and ``"std"`` tensors (shape (N, V)
                  for per-station normalisation).  Pass ``train_ds.obs_stats``
                  to get physical-unit metrics.  If None, normalised only.

    Returns:
        List of N dicts (one per station) with keys:
            station_idx                 -- 0-based index into the station list
            easting, northing           -- LV95 coordinates (metres)
            {var}_mae_norm              -- MAE in normalised space
            {var}_rmse_norm             -- RMSE in normalised space
            {var}_n_samples             -- number of valid (sample, delta) pairs
            {var}_mae, {var}_rmse       -- physical units (if obs_stats provided)
            overall_mae_norm            -- MAE across all variables (normalised)
            overall_rmse_norm           -- RMSE across all variables (normalised)

    Typical usage::

        rows = evaluate_per_station(model, test_loader, device,
                                    test_ds.spatial.cpu(), obs_stats=obs_stats)
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv("per_station_metrics.csv", index=False)
        # Plot: scatter(x=df.easting, y=df.northing, c=df.temperature_mae, ...)
    """
    model.eval()

    N = spatial.shape[0]

    # Vectorised accumulators -- no Python loop over stations or batch items.
    # We loop only over K lead-times (<=13 in the fixed-grid config).
    sum_abs_err = torch.zeros(N, NUM_TARGET_VARIABLES)   # (N, V_t) -- sum |err|
    sum_sq_err  = torch.zeros(N, NUM_TARGET_VARIABLES)   # (N, V_t) -- sum err**2
    n_valid     = torch.zeros(N, NUM_TARGET_VARIABLES, dtype=torch.long)

    for batch in loader:
        x           = batch["x"].to(device)
        x_mask      = batch["x_mask"].to(device)
        sp          = batch["spatial"].to(device)
        x_hours     = batch["x_hours"].to(device)
        y_raw       = batch["y"].to(device)
        y_mask_raw  = batch["y_mask"].to(device)
        y_hours     = batch["y_hours"].to(device)
        delta_steps = batch["delta_steps"].to(device)

        if sp.dim() == 3:
            sp = sp[0]

        if y_raw.dim() == 4:
            _, preds, _ = model.forward_multi_delta(
                x, x_mask, sp, x_hours,
                y_raw, y_mask_raw, y_hours, delta_steps,
            )
            # preds: (B, K, N, V_target)
            K = preds.shape[1]
        else:
            _, preds, _ = model(
                x, x_mask, sp, x_hours,
                y_raw, y_mask_raw, y_hours, delta_steps,
            )
            preds      = preds.unsqueeze(1)        # (B, 1, N, V_t)
            y_raw      = y_raw.unsqueeze(1)
            y_mask_raw = y_mask_raw.unsqueeze(1)
            K = 1

        for k in range(K):
            p_k = preds[:, k]                                    # (B, N, V_t)
            t_k = y_raw[:, k, :, :NUM_TARGET_VARIABLES]         # (B, N, V_t)
            m_k = y_mask_raw[:, k, :, :NUM_TARGET_VARIABLES]    # (B, N, V_t)

            err = (p_k - t_k).cpu()          # (B, N, V_t)
            m_k = m_k.cpu().bool()

            # Mask invalid sensor entries (set to 0 so they don't distort sums)
            err_m = err * m_k.float()

            # Sum over batch dimension B -> (N, V_t)
            sum_abs_err += err_m.abs().sum(dim=0)
            sum_sq_err  += err_m.pow(2).sum(dim=0)
            n_valid     += m_k.long().sum(dim=0)

    # Per-station standard deviations for physical-unit conversion
    has_phys = obs_stats is not None
    if has_phys:
        _std = obs_stats["std"].cpu()
        if _std.dim() == 2:
            std_ps = _std[:, :NUM_TARGET_VARIABLES]               # (N, V_t)
        else:
            std_ps = _std[:NUM_TARGET_VARIABLES].unsqueeze(0).expand(N, -1)

    rows: list[dict] = []
    for n in range(N):
        row: dict = {
            "station_idx": n,
            "easting":  float(spatial[n, 0].item()),
            "northing": float(spatial[n, 1].item()),
        }

        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            nv = int(n_valid[n, v].item())
            row[f"{var_name}_n_samples"] = nv
            if nv == 0:
                row[f"{var_name}_mae_norm"]  = float("nan")
                row[f"{var_name}_rmse_norm"] = float("nan")
                if has_phys:
                    row[f"{var_name}_mae"]   = float("nan")
                    row[f"{var_name}_rmse"]  = float("nan")
                continue

            mae_norm  = float(sum_abs_err[n, v].item() / nv)
            rmse_norm = float((sum_sq_err[n, v].item() / nv) ** 0.5)
            row[f"{var_name}_mae_norm"]  = mae_norm
            row[f"{var_name}_rmse_norm"] = rmse_norm

            if has_phys:
                std_v = float(std_ps[n, v].item())
                row[f"{var_name}_mae"]  = mae_norm  * std_v
                row[f"{var_name}_rmse"] = rmse_norm * std_v

        # Overall across all 5 variables for this station
        n_ov = int(n_valid[n].sum().item())
        if n_ov > 0:
            row["overall_mae_norm"]  = float(sum_abs_err[n].sum().item() / n_ov)
            row["overall_rmse_norm"] = float((sum_sq_err[n].sum().item() / n_ov) ** 0.5)
        else:
            row["overall_mae_norm"]  = float("nan")
            row["overall_rmse_norm"] = float("nan")

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Gap-filling evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_gap_filling(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    obs_stats: "dict | None" = None,
    n_repeats: int = 3,
) -> dict[str, float]:
    """
    Spatial gap-filling evaluation.

    For every batch window the *last* input timestep is used as the
    reconstruction target (delta = 0).  ``model.forward()`` is called
    ``n_repeats`` times per batch; each call applies a *different* random
    station mask (the encoder's ``_mask_stations()`` is unconditional, so
    masking varies even in ``eval`` mode).

    Metrics are computed **only on masked (hidden) stations** — the stations
    the model had to reconstruct without direct access to their values.  This
    isolates spatial interpolation skill from trivial copy-through of visible
    context.

    Args:
        model:      StationMAE (or compatible) already moved to ``device``.
        loader:     DataLoader from StationMAEDataset (any delta config;
                    delta information is overridden inside this function).
        device:     torch.device.
        obs_stats:  dict with ``"mean"`` and ``"std"`` tensors of shape
                    ``(NUM_VARIABLES,)``.  Pass ``train_ds.obs_stats`` for
                    physical-unit metrics; ``None`` returns normalised only.
        n_repeats:  Number of independent random masks per batch window.
                    Higher values give more stable estimates at the cost of
                    extra forward passes (default 3).

    Returns:
        metrics: dict[str, float] with keys:
            gap_n_masked_evals                     — total (station, repeat) pairs
            gap_{var}_norm_{rmse,mae,bias,r2}      — normalised space
            gap_overall_{rmse,mae,bias}_norm
            gap_{var}_{rmse,mae,bias,r2}           — physical units (if obs_stats)
            gap_overall_{rmse,mae,bias}            — physical (if obs_stats)
            gap_wind_speed_{rmse,mae,bias}         — m/s   (if obs_stats)
            gap_wind_dir_mae_deg                   — degrees (if obs_stats)
    """
    model.eval()

    all_preds:   list[torch.Tensor] = []   # (n_masked, 5) fragments
    all_targets: list[torch.Tensor] = []
    all_valid:   list[torch.Tensor] = []   # bool — sensor present at target step
    all_stds:    list[torch.Tensor] = []   # (n_masked, 5) per-station stds (or None)

    for batch in loader:
        x       = batch["x"].to(device)       # (B, W, N, V)
        x_mask  = batch["x_mask"].to(device)  # (B, W, N, V)
        spatial = batch["spatial"].to(device) # (N, F) or (B, N, F)
        x_hours = batch["x_hours"].to(device) # (B, W)

        if spatial.dim() == 3 and spatial.size(0) == x.size(0):
            spatial = spatial[0]              # drop batch dim if accidentally stacked

        B, W, N, V = x.shape

        # Reconstruction target: last observed input step (delta = 0)
        y_last  = x[:, -1, :, :]             # (B, N, V)
        ym_last = x_mask[:, -1, :, :]        # (B, N, V)
        yh_last = x_hours[:, -1]             # (B,)
        delta_z = torch.zeros(B, dtype=torch.long, device=device)

        # Pre-compute per-station stds if obs_stats is (N, V)
        _std_cpu = None
        if obs_stats is not None:
            _s = obs_stats["std"].cpu()
            if _s.dim() == 2:
                _std_cpu = _s[:, :NUM_TARGET_VARIABLES]   # (N, 5)

        for _ in range(n_repeats):
            # Each forward() call samples a fresh random station mask
            _, preds, masked_idx = model(
                x, x_mask, spatial, x_hours,
                y_last, ym_last, yh_last, delta_z,
            )
            # preds:      (B, N, 5)       — predictions for ALL stations
            # masked_idx: (B, N_masked)   — indices of the hidden stations

            for b in range(B):
                m_idx = masked_idx[b].cpu()                          # (N_masked,)
                all_preds.append(preds[b, m_idx, :].cpu())          # (N_masked, 5)
                all_targets.append(
                    y_last[b, m_idx, :NUM_TARGET_VARIABLES].cpu()   # (N_masked, 5)
                )
                all_valid.append(
                    ym_last[b, m_idx, :NUM_TARGET_VARIABLES].cpu()  # (N_masked, 5)
                )
                # Track exact per-station stds for physical unnormalization
                if _std_cpu is not None:
                    all_stds.append(_std_cpu[m_idx])                # (N_masked, 5)

    if not all_preds:
        return {"gap_n_masked_evals": 0.0}

    preds_cat   = torch.cat(all_preds,   dim=0)   # (total_masked, 5)
    targets_cat = torch.cat(all_targets, dim=0)
    valid_cat   = torch.cat(all_valid,   dim=0).bool()

    metrics: dict[str, float] = {
        "gap_n_masked_evals": float(preds_cat.shape[0]),
    }

    # ── Inner helper: write rmse/mae/bias/r2 under a given prefix ──────
    def _scalars(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor,
                 prefix: str, std_scale: float = 1.0) -> None:
        if m.sum() == 0:
            for k in ("rmse", "mae", "bias", "r2"):
                metrics[f"{prefix}_{k}"] = float("nan")
            return
        pv  = p[m] * std_scale
        tv  = t[m] * std_scale
        err = pv - tv
        metrics[f"{prefix}_rmse"] = float(err.pow(2).mean().sqrt().item())
        metrics[f"{prefix}_mae"]  = float(err.abs().mean().item())
        metrics[f"{prefix}_bias"] = float(err.mean().item())
        ss_res = err.pow(2).sum()
        ss_tot = (tv - tv.mean()).pow(2).sum()
        metrics[f"{prefix}_r2"]   = (
            float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        )

    # ── Normalised per-variable metrics ────────────────────────────────
    for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
        _scalars(
            preds_cat[:, v], targets_cat[:, v], valid_cat[:, v],
            prefix=f"gap_{var_name}_norm",
        )

    # ── Overall normalised ─────────────────────────────────────────────
    if valid_cat.sum() > 0:
        p_flat   = preds_cat[valid_cat]
        t_flat   = targets_cat[valid_cat]
        err_flat = p_flat - t_flat
        metrics["gap_overall_rmse_norm"] = float(err_flat.pow(2).mean().sqrt().item())
        metrics["gap_overall_mae_norm"]  = float(err_flat.abs().mean().item())
        metrics["gap_overall_bias_norm"] = float(err_flat.mean().item())
    else:
        metrics["gap_overall_rmse_norm"] = float("nan")
        metrics["gap_overall_mae_norm"]  = float("nan")
        metrics["gap_overall_bias_norm"] = float("nan")

    # ── Physical-unit metrics (requires obs_stats) ─────────────────────
    if obs_stats is not None:
        mean_t = obs_stats["mean"].cpu()
        std_t  = obs_stats["std"].cpu()

        # If per-station stds were tracked, use exact values for each masked station.
        # Otherwise fall back to station-mean std.
        if all_stds:
            stds_cat = torch.cat(all_stds, dim=0)   # (total_masked, 5) exact per-station
            use_per_station = True
        else:
            use_per_station = False
            if std_t.dim() == 2:
                std_per_var  = std_t.mean(dim=0)[:NUM_TARGET_VARIABLES]
                mean_per_var = mean_t.mean(dim=0)
            else:
                std_per_var  = std_t
                mean_per_var = mean_t

        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            if use_per_station:
                # Element-wise scaling: each station uses its own std
                scale = stds_cat[:, v]                      # (total_masked,)
                m = valid_cat[:, v]
                if m.sum() == 0:
                    for k in ("rmse", "mae", "bias", "r2"):
                        metrics[f"gap_{var_name}_{k}"] = float("nan")
                    continue
                err  = (preds_cat[m, v] - targets_cat[m, v]) * scale[m]
                metrics[f"gap_{var_name}_rmse"] = float(err.pow(2).mean().sqrt().item())
                metrics[f"gap_{var_name}_mae"]  = float(err.abs().mean().item())
                metrics[f"gap_{var_name}_bias"] = float(err.mean().item())
                tv = targets_cat[m, v] * scale[m]
                ss_res = err.pow(2).sum()
                ss_tot = (tv - tv.mean()).pow(2).sum()
                metrics[f"gap_{var_name}_r2"] = (
                    float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
                )
            else:
                std_v = float(std_per_var[v].item())
                _scalars(
                    preds_cat[:, v], targets_cat[:, v], valid_cat[:, v],
                    prefix=f"gap_{var_name}",
                    std_scale=std_v,
                )

        # Overall physical (pool rescaled errors across variables)
        if valid_cat.sum() > 0:
            errs_phys = []
            for v in range(NUM_TARGET_VARIABLES):
                m = valid_cat[:, v]
                if m.sum() > 0:
                    if use_per_station:
                        scale = stds_cat[m, v]
                        errs_phys.append((preds_cat[m, v] - targets_cat[m, v]) * scale)
                    else:
                        std_v = float(std_per_var[v].item())
                        errs_phys.append(
                            (preds_cat[m, v] - targets_cat[m, v]) * std_v
                        )
            if errs_phys:
                e = torch.cat(errs_phys)
                metrics["gap_overall_rmse"] = float(e.pow(2).mean().sqrt().item())
                metrics["gap_overall_mae"]  = float(e.abs().mean().item())
                metrics["gap_overall_bias"] = float(e.mean().item())

        # Wind speed / direction
        ui   = _IDX.get("wind_u")
        vi_i = _IDX.get("wind_v")
        if ui is not None and vi_i is not None:
            if use_per_station:
                mean_per_var = mean_t.mean(dim=0) if mean_t.dim() == 2 else mean_t
            mean_u  = float(mean_per_var[ui].item())
            mean_v  = float(mean_per_var[vi_i].item())
            m_uv = valid_cat[:, ui] & valid_cat[:, vi_i]
            if m_uv.sum() > 0:
                # Use element-wise per-station std when available so that stations
                # far from the mean std are denormalised correctly.
                if use_per_station:
                    std_u_vec   = stds_cat[m_uv, ui]
                    std_v_w_vec = stds_cat[m_uv, vi_i]
                else:
                    std_u_vec   = std_per_var[ui]
                    std_v_w_vec = std_per_var[vi_i]
                u_pred = preds_cat[m_uv, ui]   * std_u_vec   + mean_u
                u_true = targets_cat[m_uv, ui] * std_u_vec   + mean_u
                v_pred = preds_cat[m_uv, vi_i] * std_v_w_vec + mean_v
                v_true = targets_cat[m_uv, vi_i] * std_v_w_vec + mean_v

                ws_pred = (u_pred.pow(2) + v_pred.pow(2)).sqrt()
                ws_true = (u_true.pow(2) + v_true.pow(2)).sqrt()
                ws_err  = ws_pred - ws_true
                metrics["gap_wind_speed_rmse"] = float(ws_err.pow(2).mean().sqrt().item())
                metrics["gap_wind_speed_mae"]  = float(ws_err.abs().mean().item())
                metrics["gap_wind_speed_bias"] = float(ws_err.mean().item())

                wd_pred = _wind_dir_deg(u_pred, v_pred)
                wd_true = _wind_dir_deg(u_true, v_true)
                metrics["gap_wind_dir_mae_deg"] = _circular_mae_deg(wd_pred, wd_true)
            else:
                for k in ("gap_wind_speed_rmse", "gap_wind_speed_mae",
                          "gap_wind_speed_bias", "gap_wind_dir_mae_deg"):
                    metrics[k] = float("nan")

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


def print_gap_filling_metrics(metrics: dict[str, float], obs_stats: "dict | None" = None) -> None:
    """
    Pretty-print evaluate_gap_filling() results.

    Shows per-variable RMSE/MAE/Bias/R² for **masked stations only**,
    in both normalised and physical units (if obs_stats provided).
    """
    has_phys = obs_stats is not None
    unit_map  = _VAR_UNITS if has_phys else {v: "norm" for v in TARGET_VARIABLE_NAMES}
    n_evals   = int(metrics.get("gap_n_masked_evals", 0))

    print(f"\n── Gap-filling metrics (masked stations only · n_station_evals={n_evals:,}) " + "─" * 8)
    hdr = f"  {'Variable':<14}  {'RMSE':>8}  {'MAE':>8}  {'Bias':>8}  {'R²':>6}  Unit"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for var_name in TARGET_VARIABLE_NAMES:
        pfx  = (f"gap_{var_name}"
                if (has_phys and f"gap_{var_name}_rmse" in metrics)
                else f"gap_{var_name}_norm")
        rmse = metrics.get(f"{pfx}_rmse", float("nan"))
        mae  = metrics.get(f"{pfx}_mae",  float("nan"))
        bias = metrics.get(f"{pfx}_bias", float("nan"))
        r2   = metrics.get(f"{pfx}_r2",   float("nan"))
        unit = unit_map.get(var_name, "norm")
        print(f"  {var_name:<14}  {rmse:>8.4f}  {mae:>8.4f}  {bias:>+8.4f}  {r2:>6.3f}  {unit}")

    rmse_ov = metrics.get("gap_overall_rmse",
                          metrics.get("gap_overall_rmse_norm", float("nan")))
    mae_ov  = metrics.get("gap_overall_mae",
                          metrics.get("gap_overall_mae_norm",  float("nan")))
    bias_ov = metrics.get("gap_overall_bias",
                          metrics.get("gap_overall_bias_norm", float("nan")))
    print(f"  {'[overall]':<14}  {rmse_ov:>8.4f}  {mae_ov:>8.4f}  {bias_ov:>+8.4f}  {'':>6}")

    if has_phys and "gap_wind_speed_rmse" in metrics:
        print(f"\n── Wind speed & direction (gap-filling) " + "─" * 31)
        print(f"  {'wind_speed':<14}  RMSE={metrics['gap_wind_speed_rmse']:.4f} m/s  "
              f"MAE={metrics['gap_wind_speed_mae']:.4f} m/s  "
              f"Bias={metrics['gap_wind_speed_bias']:+.4f} m/s")
        print(f"  {'wind_dir':<14}  MAE={metrics.get('gap_wind_dir_mae_deg', float('nan')):.2f}°")
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
    print("\n── Per-variable metrics (all stations, present sensors) " + "─" * 15)
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


# ---------------------------------------------------------------------------
# Per-variable × per-delta RMSE matrix
# ---------------------------------------------------------------------------

def build_delta_variable_matrix(
    metrics: dict[str, float],
) -> "dict[str, dict[str, float]]":
    """
    Assemble a (variable → lead_time → rmse_norm) nested dict from the flat
    metrics dict returned by evaluate_full().

    Useful for building a heatmap in the notebook:
        rows = TARGET_VARIABLE_NAMES
        cols = lead-times in minutes (0, 30, 60, …, 360)
    """
    matrix: dict[str, dict[str, float]] = {v: {} for v in TARGET_VARIABLE_NAMES}

    delta_keys = sorted(
        k for k in metrics if k.startswith("delta_") and k.endswith("_overall_rmse_norm")
    )
    for dk in delta_keys:
        d    = int(dk.split("_")[1])
        mins = d * 10
        for var in TARGET_VARIABLE_NAMES:
            key = f"delta_{d:02d}_{var}_rmse_norm"
            matrix[var][f"{mins}min"] = metrics.get(key, float("nan"))

    return matrix


# ---------------------------------------------------------------------------
# Raw prediction collector (for notebook time-series plots)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    n_windows:  int = 100,
    save_path:  "str | None" = None,
) -> dict:
    """
    Run inference on up to ``n_windows`` test windows and return raw tensors
    needed for time-series and spatial plots in the notebook.

    Saves (and returns) a dict with:
        preds          (M, K, N, V_target)  — model predictions (normalised)
        targets        (M, K, N, V)         — ground truth (normalised)
        masks          (M, K, N, V)         — sensor availability
        delta_steps    (M, K)               — lead-time steps
        window_hours   (M,)                 — hours-since-epoch of window start
        target_hours   (M, K)               — hours-since-epoch of each target
        spatial        (N, 15)              — station static features

    Args:
        model:      StationMAE moved to ``device``, in eval mode.
        loader:     Test DataLoader (fixed-grid or single-delta).
        device:     torch.device.
        n_windows:  Max windows to collect (default 100, ~40 MB for d_model=1024).
        save_path:  If set, saves the dict as a .pt file at this path.
    """
    model.eval()

    all_preds   = []
    all_targets = []
    all_masks   = []
    all_deltas  = []
    all_w_hours = []
    all_t_hours = []
    spatial_saved = None
    collected = 0

    for batch in loader:
        if collected >= n_windows:
            break

        x           = batch["x"].to(device)
        x_mask      = batch["x_mask"].to(device)
        spatial     = batch["spatial"].to(device)
        x_hours     = batch["x_hours"].to(device)
        y_raw       = batch["y"].to(device)
        y_mask_raw  = batch["y_mask"].to(device)
        y_hours     = batch["y_hours"].to(device)
        delta_steps = batch["delta_steps"].to(device)

        if spatial.dim() == 3:
            spatial = spatial[0]
        if spatial_saved is None:
            spatial_saved = spatial.cpu()

        # Use multi-delta forward for efficiency
        if y_raw.dim() == 4:
            _, preds, _ = model.forward_multi_delta(
                x, x_mask, spatial, x_hours,
                y_raw, y_mask_raw, y_hours, delta_steps,
            )
            # preds: (B, K, N, V_target)
        else:
            _, preds, _ = model(x, x_mask, spatial, x_hours,
                                y_raw, y_mask_raw, y_hours, delta_steps)
            preds       = preds.unsqueeze(1)
            y_raw       = y_raw.unsqueeze(1)
            y_mask_raw  = y_mask_raw.unsqueeze(1)
            y_hours     = y_hours.unsqueeze(1)
            delta_steps = delta_steps.unsqueeze(1)

        B = preds.shape[0]
        take = min(B, n_windows - collected)

        all_preds.append(preds[:take].cpu())
        all_targets.append(y_raw[:take].cpu())
        all_masks.append(y_mask_raw[:take].cpu())
        all_deltas.append(delta_steps[:take].cpu())
        all_w_hours.append(x_hours[:take, 0].cpu())   # start of window
        all_t_hours.append(y_hours[:take].cpu())
        collected += take

    result = {
        "preds":        torch.cat(all_preds,   dim=0),   # (M, K, N, V_target)
        "targets":      torch.cat(all_targets, dim=0),   # (M, K, N, V)
        "masks":        torch.cat(all_masks,   dim=0),   # (M, K, N, V)
        "delta_steps":  torch.cat(all_deltas,  dim=0),   # (M, K)
        "window_hours": torch.cat(all_w_hours, dim=0),   # (M,)
        "target_hours": torch.cat(all_t_hours, dim=0),   # (M, K)
        "spatial":      spatial_saved,                    # (N, 15)
        "var_names":    TARGET_VARIABLE_NAMES,
        "n_windows":    collected,
    }

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(result, save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"  Saved {collected} windows → {save_path}  ({size_mb:.0f} MB)")

    return result


# ---------------------------------------------------------------------------
# Seasonal metrics breakdown
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_seasonal_metrics(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    obs_stats: "dict | None" = None,
) -> dict[str, dict[str, float]]:
    """
    Compute evaluate_full()-style metrics split by meteorological season.

    Seasons (NH convention):
        DJF — Dec, Jan, Feb  (winter)
        MAM — Mar, Apr, May  (spring)
        JJA — Jun, Jul, Aug  (summer)
        SON — Sep, Oct, Nov  (autumn)

    Returns a dict mapping season name → metrics dict (same keys as evaluate_full).
    """
    import datetime

    _EPOCH = datetime.datetime(1970, 1, 1)

    def _season(hours: float) -> str:
        dt    = _EPOCH + datetime.timedelta(hours=float(hours))
        month = dt.month
        if month in (12, 1, 2):  return "DJF"
        if month in (3,  4, 5):  return "MAM"
        if month in (6,  7, 8):  return "JJA"
        return "SON"

    # Collect per-season records: same structure as evaluate_full
    season_records: dict[str, list] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}

    model.eval()

    for batch in loader:
        x           = batch["x"].to(device)
        x_mask      = batch["x_mask"].to(device)
        spatial     = batch["spatial"].to(device)
        x_hours     = batch["x_hours"].to(device)
        y_raw       = batch["y"].to(device)
        y_mask_raw  = batch["y_mask"].to(device)
        y_hours     = batch["y_hours"].to(device)
        delta_steps = batch["delta_steps"].to(device)

        if spatial.dim() == 3:
            spatial = spatial[0]

        if y_raw.dim() == 4:
            _, preds, _ = model.forward_multi_delta(
                x, x_mask, spatial, x_hours,
                y_raw, y_mask_raw, y_hours, delta_steps,
            )
            K = preds.shape[1]
        else:
            _, preds, _ = model(x, x_mask, spatial, x_hours,
                                y_raw, y_mask_raw, y_hours, delta_steps)
            preds       = preds.unsqueeze(1)
            y_raw       = y_raw.unsqueeze(1)
            y_mask_raw  = y_mask_raw.unsqueeze(1)
            delta_steps = delta_steps.unsqueeze(1)
            K = 1

        B, _, N, _ = y_raw.shape

        # Classify each sample by the season of its window start
        for b in range(B):
            s = _season(x_hours[b, 0].item())
            for k in range(K):
                y_target      = y_raw[b, k, :, :NUM_TARGET_VARIABLES].cpu()
                y_mask_target = y_mask_raw[b, k, :, :NUM_TARGET_VARIABLES].cpu()
                pred_bk       = preds[b, k].cpu()
                ds_bk         = int(delta_steps[b, k].item()) if delta_steps.dim() == 2 \
                                 else int(delta_steps[b].item())
                season_records[s].append({
                    "pred":   pred_bk.reshape(N, NUM_TARGET_VARIABLES),
                    "target": y_target.reshape(N, NUM_TARGET_VARIABLES),
                    "mask":   y_mask_target.reshape(N, NUM_TARGET_VARIABLES),
                    "delta":  ds_bk,
                    "N":      N,
                })

    # Compute metrics per season
    seasonal_metrics: dict[str, dict[str, float]] = {}

    for season, recs in season_records.items():
        if not recs:
            seasonal_metrics[season] = {}
            continue

        preds_all   = torch.cat([r["pred"]   for r in recs], dim=0)
        targets_all = torch.cat([r["target"] for r in recs], dim=0)
        masks_all   = torch.cat([r["mask"]   for r in recs], dim=0).bool()

        m: dict[str, float] = {"n_samples": float(len(recs))}

        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            mv = masks_all[:, v]
            if mv.sum() == 0:
                m[f"{var_name}_rmse_norm"] = float("nan")
                m[f"{var_name}_mae_norm"]  = float("nan")
                continue
            err = preds_all[mv, v] - targets_all[mv, v]
            m[f"{var_name}_rmse_norm"] = float(err.pow(2).mean().sqrt().item())
            m[f"{var_name}_mae_norm"]  = float(err.abs().mean().item())

        if masks_all.sum() > 0:
            err_all = preds_all[masks_all] - targets_all[masks_all]
            m["overall_rmse_norm"] = float(err_all.pow(2).mean().sqrt().item())
            m["overall_mae_norm"]  = float(err_all.abs().mean().item())

        # Physical units
        if obs_stats is not None:
            _std = obs_stats["std"].cpu()
            std_per_var = _std.mean(dim=0) if _std.dim() == 2 else _std
            for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
                mv  = masks_all[:, v]
                sv  = float(std_per_var[v].item())
                if mv.sum() > 0:
                    err = (preds_all[mv, v] - targets_all[mv, v]) * sv
                    m[f"{var_name}_rmse"] = float(err.pow(2).mean().sqrt().item())
                    m[f"{var_name}_mae"]  = float(err.abs().mean().item())

        seasonal_metrics[season] = m

    return seasonal_metrics
