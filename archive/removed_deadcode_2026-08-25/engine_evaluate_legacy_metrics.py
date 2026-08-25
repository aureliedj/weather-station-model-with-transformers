"""
engine/evaluate.py — legacy in-script metric functions (ARCHIVED 2026-08-25)

Removed from src/engine/evaluate.py because nothing in the repository calls
them. Kept here verbatim for provenance: these are the functions that computed
metrics INSIDE the evaluation script, before test.py was changed to dump raw
tensors and let the notebooks own every metric.

    evaluate_full          full test-set evaluation in physical units:
                           RMSE / MAE / bias (MBE) / R^2, wind speed and
                           circular wind-direction error, per-lead breakdown.
    evaluate_gap_filling   the same, split by masked vs visible stations.
    _r2, _row_stat, _wind_dir_deg, _circular_mae_deg
                           private helpers used only by the two above.
    _VAR_UNITS, _IDX       module constants used only by the two above.

Results produced BEFORE the switch to notebook-owned metrics came from this
code. Results produced after come from notebooks/, which read predictions.pt.
The two are not guaranteed to agree — see EXPERIMENTS.md, "Open questions".

This file is NOT importable as-is: it was cut out of a module and still expects
that module's imports. It is a record, not a dependency. To run any of it,
paste the function back into src/engine/evaluate.py.

Original line numbers in the pre-removal file are given per block below.
"""

# The imports the removed code expected, reproduced for reference:
#     import torch
#     from model.embeddings import TARGET_VARIABLE_NAMES, NUM_TARGET_VARIABLES


# ==========================================================================
# original evaluate.py lines 62-73
# ==========================================================================

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

# ==========================================================================
# original evaluate.py lines 74-74
# ==========================================================================

_IDX = {v: i for i, v in enumerate(TARGET_VARIABLE_NAMES)}

# ==========================================================================
# original evaluate.py lines 77-83
# ==========================================================================

def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination R²."""
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum()
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

# ==========================================================================
# original evaluate.py lines 86-92
# ==========================================================================

def _wind_dir_deg(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Meteorological wind direction in [0, 360): direction FROM which wind blows."""
    # Meteorological convention: 0° = N (wind from north), 90° = E, etc.
    # dir_from = atan2(-u, -v) in radians, shifted to [0, 2π)
    rad = torch.atan2(-u, -v)
    deg = torch.rad2deg(rad)
    return deg % 360.0

# ==========================================================================
# original evaluate.py lines 95-135
# ==========================================================================

def _row_stat(table: torch.Tensor, v: int, station_idx: torch.Tensor) -> torch.Tensor:
    """
    Per-row normalisation statistic for variable ``v``.

    Physical-unit metrics must undo the SAME transform the data went through.
    Observations are z-scored per (station, variable), so the only correct
    inverse is that station's own mean/std::

        x_phys = x_norm * std[station, v] + mean[station, v]

    This module used to collapse ``std`` to its cross-station mean before
    converting, which silently mis-scales every station whose spread differs
    from the network average. The damage is worst for wind, where std spans
    0.51-7.19 m/s across the 155 stations (a 14x range) against only 1.4x for
    temperature: measured on v27, the averaged-std shortcut inflated
    wind-direction MAE from 15.8 deg to 20.3 deg for winds above 3 m/s.

    Args:
        table:       (N, V) per-station stats, or (V,) global stats.
        v:           variable column.
        station_idx: (R,) station index for each row of the flattened arrays.

    Returns:
        (R,) tensor aligned with the flattened rows.

    Falls back to the cross-station mean for any row whose station index is
    out of range — that should never happen, but a silent mis-index would
    corrupt every physical number, so it is guarded and reported rather than
    trusted.
    """
    R = station_idx.shape[0]
    if table.dim() == 1:                      # global stats — nothing to index
        return table[v].reshape(1).expand(R)

    n_stats = table.shape[0]
    col = table[:, v]
    valid = station_idx < n_stats
    out = col[station_idx.clamp(max=n_stats - 1)]
    if not bool(valid.all()):
        out = torch.where(valid, out, col.mean())
    return out

# ==========================================================================
# original evaluate.py lines 138-143
# ==========================================================================

def _circular_mae_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> float:
    """Mean absolute error on a circular quantity (degrees)."""
    diff = pred_deg - true_deg
    # Wrap to [-180, 180]
    diff = (diff + 180.0) % 360.0 - 180.0
    return float(diff.abs().mean().item())

# ==========================================================================
# original evaluate.py lines 146-417
# ==========================================================================

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
    # Station index for every flattened row. Each record was reshaped from
    # (B, N, V) in row-major order, so the station axis cycles fastest:
    # [n0 … n(N-1)] repeated B times. Without this the per-station
    # normalisation cannot be inverted correctly (see _row_stat).
    station_all = torch.cat(
        [torch.arange(r["N"]).repeat(r["pred"].shape[0] // r["N"])
         for r in records],
        dim=0,
    )

    metrics: dict[str, float] = {
        "avg_loss": total_loss / max(n_batches, 1),
    }

    # ── Helper: compute scalar metrics for one (pred, target, mask) slice ──
    def _scalars(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor,
                 prefix: str, std_scale=1.0) -> None:
        """Write rmse, mae, bias, r2 into metrics dict under `prefix`.

        ``std_scale`` may be a scalar (normalised metrics) or a per-row tensor
        aligned with ``p`` (physical metrics with per-station stats).
        """
        if m.sum() == 0:
            for k in ("rmse", "mae", "bias", "r2"):
                metrics[f"{prefix}_{k}"] = float("nan")
            return
        sc = std_scale[m] if torch.is_tensor(std_scale) else std_scale
        pv = p[m] * sc
        tv = t[m] * sc
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

        # EXACT per-station inverse transform. This used to average std over
        # stations first, which mis-scales every station whose spread differs
        # from the network mean — see _row_stat for the measured impact.
        # _row_stat falls back to the cross-station mean only for rows whose
        # station is unknown, and for (V,)-shaped global stats.
        _std_row  = {v: _row_stat(std_t,  v, station_all)
                     for v in range(NUM_TARGET_VARIABLES)}
        _mean_row = {v: _row_stat(mean_t, v, station_all)
                     for v in range(NUM_TARGET_VARIABLES)}

        # Per-variable in physical units
        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            _scalars(
                preds_all[:, v], targets_all[:, v], masks_all[:, v],
                prefix=var_name,
                std_scale=_std_row[v],
            )

        # Overall physical
        if masks_all.sum() > 0:
            errs_phys = []
            for v in range(NUM_TARGET_VARIABLES):
                m = masks_all[:, v]
                if m.sum() > 0:
                    errs_phys.append(
                        ((preds_all[m, v] - targets_all[m, v]) * _std_row[v][m])
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
            # Speed is a distance from the origin and direction is an angle, so
            # BOTH depend on getting each station's own mean and std right — an
            # averaged std shears the (u, v) plane and rotates the recovered
            # direction. This is the single worst-affected metric in the file.
            m_uv = masks_all[:, ui] & masks_all[:, vi]
            if m_uv.sum() > 0:
                su, mu = _std_row[ui][m_uv], _mean_row[ui][m_uv]
                sv, mv = _std_row[vi][m_uv], _mean_row[vi][m_uv]
                u_pred = preds_all[m_uv, ui]   * su + mu
                u_true = targets_all[m_uv, ui] * su + mu
                v_pred = preds_all[m_uv, vi]   * sv + mv
                v_true = targets_all[m_uv, vi] * sv + mv

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

# ==========================================================================
# original evaluate.py lines 578-851
# ==========================================================================

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
    all_means:   list[torch.Tensor] = []   # (n_masked, 5) per-station means (or None)
    # Means are tracked as well as stds because wind speed/direction need the
    # full affine inverse: speed is a distance from the ORIGIN, so a wrong
    # offset moves the calm point and rotates the recovered direction. Using a
    # per-station std with a global mean (the previous behaviour) is still wrong.

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

        # Pre-compute per-station stats if obs_stats is (N, V)
        _std_cpu = None
        _mean_cpu = None
        if obs_stats is not None:
            _s = obs_stats["std"].cpu()
            if _s.dim() == 2:
                _std_cpu = _s[:, :NUM_TARGET_VARIABLES]   # (N, 5)
            _m = obs_stats["mean"].cpu()
            if _m.dim() == 2:
                _mean_cpu = _m[:, :NUM_TARGET_VARIABLES]  # (N, 5)

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
                if _mean_cpu is not None:
                    all_means.append(_mean_cpu[m_idx])              # (N_masked, 5)

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
            means_cat = (torch.cat(all_means, dim=0) if all_means else None)
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
            if not use_per_station:
                mean_per_var = mean_t.mean(dim=0) if mean_t.dim() == 2 else mean_t
            m_uv = valid_cat[:, ui] & valid_cat[:, vi_i]
            if m_uv.sum() > 0:
                # Full per-station affine inverse for BOTH components. Using a
                # per-station std with a global mean still shifts the origin,
                # which biases speed and rotates direction.
                if use_per_station:
                    std_u_vec   = stds_cat[m_uv, ui]
                    std_v_w_vec = stds_cat[m_uv, vi_i]
                else:
                    std_u_vec   = std_per_var[ui]
                    std_v_w_vec = std_per_var[vi_i]
                if use_per_station and means_cat is not None:
                    mean_u = means_cat[m_uv, ui]
                    mean_v = means_cat[m_uv, vi_i]
                else:
                    _mpv = (mean_t.mean(dim=0) if mean_t.dim() == 2 else mean_t)
                    mean_u = float(_mpv[ui].item())
                    mean_v = float(_mpv[vi_i].item())
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
