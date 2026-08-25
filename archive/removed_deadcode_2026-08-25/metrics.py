"""
model/metrics.py — the shared wandb metric contract.

Four models are trained in this project (v15/v17/v18 Station-MAE, the LSTM
baseline, simple-MAE, and the masked transformer) and their panels are only
comparable if they log the SAME metric under the SAME name, computed the SAME
way. They had drifted: the LSTM logged ``train/overall_rmse`` and nothing else
did; simple-MAE logged ``val/persist_skill`` where the others used
``sanity/persist_skill``; only two of the four logged
``val/horizon_sensitivity``. Each difference silently produces a panel with
one line on it.

This module owns the definitions so they cannot drift again.

The contract
------------
    train/loss                  per-step
    train/lr                    per-step
    train/overall_mae           epoch, +30 min lead
    train/overall_rmse          epoch, +30 min lead
    val/loss                    epoch
    val/overall_mae             epoch, +30 min lead      <- checkpoint monitor
    val/overall_rmse            epoch, +30 min lead
    val/{var}_mae               epoch, per target variable
    val/{var}_mae_phys          epoch, x obs_std -> physical units
    val/horizon_sensitivity     |pred(6 h) - pred(30 min)|; collapse detector
    sanity/dispersion_min       min over variables of std(pred)/std(target)
    sanity/persist_skill        1 - RMSE(model)/RMSE(persistence)
    sanity/ctx_ratio            error at visible / error at masked stations

Conventions that make the numbers comparable
--------------------------------------------
* The reported lead is +30 min (k=1 on the stride-3 delta grid) for every
  model. k=0 is excluded: at delta=0 a visible station can copy its own last
  observation, so it measures nothing.
* Validation runs with ALL stations visible (``--val_mask_ratio 0``), which is
  the task the LSTM solves, so the panels are like-for-like.
* Everything is in per-station-normalised units except ``*_mae_phys``.
* ``sanity/ctx_ratio`` needs both masked and visible stations and is therefore
  undefined for models that do not mask (the LSTM). It is omitted rather than
  logged as a placeholder — a fake value on a shared panel is worse than a
  missing line.
"""

from __future__ import annotations

import torch

__all__ = ["log_overall", "horizon_sensitivity", "dispersion_min",
           "persist_skill", "CONTRACT"]

CONTRACT = (
    "train/loss", "train/lr", "train/overall_mae", "train/overall_rmse",
    "val/loss", "val/overall_mae", "val/overall_rmse",
    "val/horizon_sensitivity",
    "sanity/dispersion_min", "sanity/persist_skill", "sanity/ctx_ratio",
)


def log_overall(log_fn, split: str, preds: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor, *, on_step: bool = False,
                on_epoch: bool = True, sync_dist: bool = True,
                prog_bar: bool = False) -> None:
    """
    Log ``{split}/overall_mae`` and ``{split}/overall_rmse`` over ``mask``.

    Both are aggregated over every (station, variable) element with a present
    sensor — NOT a mean of per-variable means — so a variable with sparse
    coverage does not get the same weight as a fully-covered one. All four
    trainers must call this rather than computing their own, which is how the
    LSTM ended up with RMSE where everything else had MAE.

    Args:
        log_fn:  the LightningModule's ``self.log``.
        split:   "train" or "val".
        preds:   (..., V) predictions, normalised units.
        targets: (..., V) targets, same shape.
        mask:    (..., V) bool or float; True/1.0 where the sensor is present.
    """
    m = mask.bool()
    if not m.any():
        return
    err = (preds - targets)[m]
    log_fn(f"{split}/overall_mae", err.abs().mean(),
           on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist,
           prog_bar=prog_bar)
    log_fn(f"{split}/overall_rmse", err.pow(2).mean().sqrt(),
           on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)


@torch.no_grad()
def horizon_sensitivity(preds: torch.Tensor, k_short: int = 1,
                        k_long: int = -1) -> torch.Tensor:
    """
    Mean |pred(long lead) - pred(short lead)|.

    A model that has collapsed to a constant emits the same value at every
    lead, so this falls toward zero. Its empirical floor in this project is
    ~0.05; a healthy v15-family run sits near 0.20.
    """
    return (preds[:, k_long] - preds[:, k_short]).abs().mean()


@torch.no_grad()
def dispersion_min(preds: torch.Tensor, targets: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Minimum over variables of std(pred)/std(target).

    Near zero means the model is emitting a near-constant for at least one
    variable — the failure that RMSE alone will not reveal, because predicting
    the conditional mean is a decent RMSE strategy.
    """
    m = mask.bool()
    out = []
    for v in range(preds.shape[-1]):
        mv = m[..., v]
        if mv.any():
            t = targets[..., v][mv].std()
            out.append(preds[..., v][mv].std() / t.clamp(min=1e-6))
    return torch.stack(out).min() if out else preds.new_zeros(())


@torch.no_grad()
def persist_skill(preds: torch.Tensor, targets: torch.Tensor,
                  mask: torch.Tensor, persistence: torch.Tensor) -> torch.Tensor:
    """
    1 - RMSE(model)/RMSE(persistence). Positive means better than persistence.

    ``persistence`` is the last observed value broadcast to every lead.
    """
    m = mask.bool()
    if not m.any():
        return preds.new_zeros(())
    rm = (preds - targets)[m].pow(2).mean().sqrt()
    rp = (persistence - targets)[m].pow(2).mean().sqrt()
    return 1.0 - rm / rp.clamp(min=1e-6)
