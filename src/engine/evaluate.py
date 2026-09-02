"""
engine/evaluate.py

Inference helper used by src/test.py: run a model over a test DataLoader and
collect the raw tensors that the analysis notebooks work from.
"""

import contextlib
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.embeddings import TARGET_VARIABLE_NAMES


@torch.no_grad()
def collect_predictions(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    n_windows: int = 100,
    save_path: "str | None" = None,
) -> dict:
    """
    Predict up to ``n_windows`` test windows and return (and optionally save)

        preds        (M, K, N, V_t)   normalised predictions
        targets      (M, K, N, V)     normalised targets
        masks        (M, K, N, V)     sensor availability
        masked_idx   (M, n_masked)    stations hidden from the encoder
        delta_steps  (M, K)           lead times in 10-min steps
        window_hours (M,)             hours since epoch of the window start
        target_hours (M, K)           hours since epoch of each target
        spatial      (N, 15)          static station features
        log_var      (M, K, N, V_t)   only for models with a log-variance head
        var_names, n_windows
    """
    model.eval()
    out = {k: [] for k in ("preds", "targets", "masks", "masked_idx", "log_var",
                           "delta_steps", "window_hours", "target_hours")}
    spatial_saved = None
    collected = 0

    try:
        from tqdm import tqdm
        bs = getattr(loader, "batch_size", None) or 1
        it = tqdm(loader, total=min(len(loader), -(-int(n_windows) // bs)),
                  desc="predict", unit="batch")
    except ImportError:
        it = loader

    # bf16 autocast on CUDA matches the mixed precision used in training.
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else contextlib.nullcontext())

    for batch in it:
        if collected >= n_windows:
            break
        x, x_mask   = batch["x"].to(device), batch["x_mask"].to(device)
        spatial     = batch["spatial"].to(device)
        x_hours     = batch["x_hours"].to(device)
        y, y_mask   = batch["y"].to(device), batch["y_mask"].to(device)
        y_hours     = batch["y_hours"].to(device)
        delta_steps = batch["delta_steps"].to(device)
        if spatial.dim() == 3:
            spatial = spatial[0]
        if spatial_saved is None:
            spatial_saved = spatial.cpu()

        with amp:
            _, preds, midx, log_var = model.forward_multi_delta(
                x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps,
                return_log_var=True)

        take = min(preds.shape[0], n_windows - collected)
        # .clone() detaches the slices from the workers' shared-memory files.
        out["preds"].append(preds[:take].float().cpu().clone())
        out["targets"].append(y[:take].cpu().clone())
        out["masks"].append(y_mask[:take].cpu().clone())
        out["masked_idx"].append(midx[:take].cpu().clone())
        if log_var is not None:
            out["log_var"].append(log_var[:take].float().cpu().clone())
        out["delta_steps"].append(delta_steps[:take].cpu().clone())
        out["window_hours"].append(x_hours[:take, 0].cpu().clone())
        out["target_hours"].append(y_hours[:take].cpu().clone())
        collected += take

    result = {k: torch.cat(v, dim=0) for k, v in out.items() if v}
    result["spatial"]   = spatial_saved
    result["var_names"] = list(TARGET_VARIABLE_NAMES)
    result["n_windows"] = collected

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(result, save_path)
        print(f"  Saved {collected:,} windows to {save_path} "
              f"({os.path.getsize(save_path) / 1e6:.0f} MB)")
    return result
