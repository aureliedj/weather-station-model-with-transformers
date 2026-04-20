"""
model/lightning_module.py

PyTorch Lightning module wrapping StationMAE.

Responsibilities
----------------
    training_step       — single-delta or multi-delta forward + loss
    validation_step     — single-delta forward, per-variable RMSE/MAE
    configure_optimizers — AdamW (decay/no-decay groups) + cosine-warmup scheduler

All metrics are emitted via self.log() and are automatically streamed to
whatever logger is attached to the Trainer (WandB, CSV, …).

WandB dashboard panels
-----------------------
    train/loss          — per-step and per-epoch training MSE
    train/lr            — learning rate (per step)
    val/loss            — epoch validation MSE  (used for ModelCheckpoint / EarlyStopping)
    val/overall_rmse    — epoch overall RMSE across all target variables
    val/{var}_rmse      — per-variable RMSE (temperature, pressure, …)
    val/{var}_mae       — per-variable MAE
"""

import math

import torch
import pytorch_lightning as pl


from .mae import StationMAE
from .embeddings import TARGET_VARIABLE_NAMES, NUM_TARGET_VARIABLES


class StationMAELightning(pl.LightningModule):
    """
    Thin Lightning wrapper around StationMAE.

    Args:
        model:  Fully constructed StationMAE instance (not yet moved to device —
                Lightning handles device placement).
        cfg:    Training hyper-parameters dict.  Expected keys:

                    lr              float  (default 1e-4)
                    weight_decay    float  (default 0.05)
                    epochs          int    (default 100)
                    warmup_epochs   int    (default 5)
    """

    def __init__(self, model: StationMAE, cfg: dict):
        super().__init__()
        self.model = model
        self.cfg   = cfg
        # Persists cfg into Lightning checkpoints (readable in test.py)
        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch: dict) -> tuple:
        """Move nothing (Lightning does device placement); just unpack keys."""
        x           = batch["x"]
        x_mask      = batch["x_mask"]
        spatial     = batch["spatial"]
        x_hours     = batch["x_hours"]
        y           = batch["y"]
        y_mask      = batch["y_mask"]
        y_hours     = batch["y_hours"]
        delta_steps = batch["delta_steps"]

        # spatial from the collate may arrive as (B, N, 15) — keep only (N, 15)
        if spatial.dim() == 3 and spatial.size(0) == x.size(0):
            spatial = spatial[0]

        return x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps = (
            self._unpack_batch(batch)
        )

        is_multi   = delta_steps.dim() == 2
        forward_fn = self.model.forward_multi_delta if is_multi else self.model.forward

        loss, _, _ = forward_fn(
            x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
        )

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr",   lr,   on_step=True, on_epoch=False, prog_bar=False)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps = (
            self._unpack_batch(batch)
        )

        # Validation always uses a single lead-time (first delta if multi)
        ds = delta_steps[:, 0] if delta_steps.dim() == 2 else delta_steps
        y_ = y[:, 0]      if y.dim()      == 4 else y
        ym = y_mask[:, 0] if y_mask.dim() == 4 else y_mask
        yh = y_hours[:, 0] if y_hours.dim() == 2 else y_hours

        loss, preds, masked_idx = self.model(
            x, x_mask, spatial, x_hours, y_, ym, yh, ds
        )

        # Restrict to masked stations + present sensors
        y_target      = y_[:, :, :NUM_TARGET_VARIABLES]
        y_mask_target = ym[:, :, :NUM_TARGET_VARIABLES]

        B = preds.size(0)
        preds_list, targets_list, masks_list = [], [], []
        for b in range(B):
            mi = masked_idx[b]
            preds_list.append(preds[b, mi])
            targets_list.append(y_target[b, mi])
            masks_list.append(y_mask_target[b, mi])

        preds_all   = torch.cat(preds_list,   dim=0)           # (M, 5)
        targets_all = torch.cat(targets_list, dim=0)
        masks_all   = torch.cat(masks_list,   dim=0).bool()

        # ── Log val loss (drives ModelCheckpoint + EarlyStopping) ──────
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # ── Per-variable RMSE / MAE ─────────────────────────────────────
        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            m = masks_all[:, v]
            if m.sum() > 0:
                p, t = preds_all[m, v], targets_all[m, v]
                self.log(f"val/{var_name}_rmse", (p - t).pow(2).mean().sqrt(),
                         on_epoch=True, sync_dist=True)
                self.log(f"val/{var_name}_mae",  (p - t).abs().mean(),
                         on_epoch=True, sync_dist=True)

        # ── Overall RMSE across all variables ──────────────────────────
        if masks_all.sum() > 0:
            pf = preds_all[masks_all]
            tf = targets_all[masks_all]
            self.log("val/overall_rmse", (pf - tf).pow(2).mean().sqrt(),
                     on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/overall_mae",  (pf - tf).abs().mean(),
                     on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer + LR scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        lr            = self.cfg.get("lr",            1e-4)
        weight_decay  = self.cfg.get("weight_decay",  0.05)
        warmup_epochs = self.cfg.get("warmup_epochs", 5)

        # Separate decay / no-decay parameter groups (standard transformer recipe)
        # Biases, LayerNorm, embeddings and mask tokens are excluded from weight decay.
        decay_params, nodecay_params = [], []
        no_decay_kw = ("bias", "norm", "embedding", "mask_token", "lambdas")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            bucket = nodecay_params if any(kw in name for kw in no_decay_kw) else decay_params
            bucket.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Lightning provides estimated_stepping_batches = epochs × steps_per_epoch
        # accounting for gradient accumulation, so the schedule is always correct.
        total_steps  = self.trainer.estimated_stepping_batches
        epochs       = self.cfg.get("epochs", 100)
        steps_per_ep = max(total_steps // epochs, 1)
        warmup_steps = warmup_epochs * steps_per_ep

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",   # call scheduler.step() after every optimiser step
                "frequency": 1,
            },
        }
