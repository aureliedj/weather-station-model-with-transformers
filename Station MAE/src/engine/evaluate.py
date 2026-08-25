"""
engine/evaluate.py

Test-set inference helpers for Station-MAE.

Public API
----------
    collect_predictions(model, loader, device, ...)
        -> dict of raw tensors, written to predictions.pt by src/test.py.
        This is the live path. It computes NO metrics: predictions, targets,
        masks and time axes are dumped in normalised space and every metric is
        derived downstream in notebooks/, so the script and the analysis cannot
        disagree.

    evaluate_per_station(model, loader, device, obs_stats=None)
        -> list[dict], per-station MAE / RMSE over all test windows.
        Used only by notebooks/Station_MAE_Map.ipynb to draw the geographic
        error maps.

Historical note
---------------
This module used to compute metrics in-script (evaluate_full,
evaluate_gap_filling and their helpers). Those functions were unreachable once
test.py switched to dumping raw tensors, and were moved on 2026-08-25 to

    archive/removed_deadcode_2026-08-25/engine_evaluate_legacy_metrics.py

Results predating that switch were produced by the archived code. See
EXPERIMENTS.md, "Open questions".

Units per predicted variable
----------------------------
    temperature  -> degC        wind_u  -> m/s  (eastward)
    pressure     -> hPa         wind_v  -> m/s  (northward)
    humidity     -> %

Metrics are computed over all N stations wherever sensors are present
(y_mask == 1). See per-function docstrings for details.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.embeddings import TARGET_VARIABLE_NAMES, NUM_TARGET_VARIABLES


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

    Unlike a run-level metric, which flattens B\xd7N stations together, this
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
        masked_idx     (M, N_masked)        — encoder-hidden station indices per
                                              window (N_masked = mask_ratio × N;
                                              empty at mask_ratio = 0.0) — needed
                                              for gap-filling analysis downstream
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
    all_midx    = []
    all_log_var = []          # predicted log σ² (NLL models only; empty otherwise)
    all_deltas  = []
    all_w_hours = []
    all_t_hours = []
    spatial_saved = None
    collected = 0

    try:
        from tqdm import tqdm
        _bs = getattr(loader, "batch_size", None) or 1
        _total = min(len(loader), -(-int(n_windows) // _bs))
        _iter = tqdm(loader, total=_total, desc="predict", unit="batch")
    except ImportError:
        _iter = loader

    for batch in _iter:
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

        # bf16 autocast on CUDA: halves activation memory (the flat d1024
        # encoder OOMs in fp32 on small MIG slices) and matches the bf16 AMP
        # the model was trained with. No-op on CPU/MPS.
        import contextlib
        _amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else contextlib.nullcontext())

        # Use multi-delta forward for efficiency.
        # return_log_var=True also retrieves the predicted log σ² when the model
        # was built with use_nll_loss (v9-style NLL runs); it is None otherwise.
        log_var = None
        if y_raw.dim() == 4:
            with _amp:
                _, preds, midx, log_var = model.forward_multi_delta(
                    x, x_mask, spatial, x_hours,
                    y_raw, y_mask_raw, y_hours, delta_steps,
                    return_log_var=True,
                )
            preds = preds.float()
            if log_var is not None:
                log_var = log_var.float()
            # preds / log_var: (B, K, N, V_target)
        else:
            with _amp:
                _, preds, midx = model(x, x_mask, spatial, x_hours,
                                       y_raw, y_mask_raw, y_hours, delta_steps)
            preds       = preds.float()
            preds       = preds.unsqueeze(1)
            y_raw       = y_raw.unsqueeze(1)
            y_mask_raw  = y_mask_raw.unsqueeze(1)
            y_hours     = y_hours.unsqueeze(1)
            delta_steps = delta_steps.unsqueeze(1)

        B = preds.shape[0]
        take = min(B, n_windows - collected)

        # .clone(): on CPU devices .cpu() is a no-op returning a VIEW of the
        # worker's shared-memory tensor; keeping views across the whole test
        # set pins one mapped temp file per batch (file_system strategy) →
        # "Too many open files". clone() copies out and releases the file.
        # (On CUDA the .cpu() round-trip already copies; clone is a no-cost
        # safety net there.)
        all_preds.append(preds[:take].cpu().clone())
        all_targets.append(y_raw[:take].cpu().clone())
        all_masks.append(y_mask_raw[:take].cpu().clone())
        if log_var is not None:
            all_log_var.append(log_var[:take].cpu().clone())
        if midx is not None:
            all_midx.append(midx[:take].cpu().clone())
        all_deltas.append(delta_steps[:take].cpu().clone())
        all_w_hours.append(x_hours[:take, 0].cpu().clone())   # start of window
        all_t_hours.append(y_hours[:take].cpu().clone())
        collected += take

    result = {
        "preds":        torch.cat(all_preds,   dim=0),   # (M, K, N, V_target)
        "targets":      torch.cat(all_targets, dim=0),   # (M, K, N, V)
        "masks":        torch.cat(all_masks,   dim=0),   # (M, K, N, V)
        "masked_idx":   (torch.cat(all_midx, dim=0) if all_midx
                         else torch.empty(collected, 0, dtype=torch.long)),  # (M, N_masked)
        "delta_steps":  torch.cat(all_deltas,  dim=0),   # (M, K)
        "window_hours": torch.cat(all_w_hours, dim=0),   # (M,)
        "target_hours": torch.cat(all_t_hours, dim=0),   # (M, K)
        "spatial":      spatial_saved,                    # (N, 15)
        "var_names":    TARGET_VARIABLE_NAMES,
        "n_windows":    collected,
    }
    # Predicted uncertainty — only present for NLL models (v9-style).
    # log_var = log σ² in normalised space; σ = exp(0.5 · log_var).
    if all_log_var:
        result["log_var"] = torch.cat(all_log_var, dim=0)   # (M, K, N, V_target)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(result, save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"  Saved {collected} windows → {save_path}  ({size_mb:.0f} MB)")

    return result


