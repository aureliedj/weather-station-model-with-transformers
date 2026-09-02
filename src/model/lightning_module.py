"""
model/lightning_module.py

PyTorch Lightning wrapper around StationMAE.

Logged metrics
--------------
    train/loss            per-step training loss
    train/lr              learning rate per step
    train/overall_mae     epoch MAE at the +30 min lead (normalised units)
    val/loss              epoch validation loss at the +30 min lead
    val/overall_mae       epoch MAE at +30 min over all variables and stations
    val/{var}_mae         per-variable MAE at +30 min (normalised)
    val/{var}_mae_phys    the same in physical units (x mean training std)
    val/horizon_sensitivity  mean |pred(+6 h) - pred(+30 min)| on the first
                          validation batch (checks that the output depends on
                          the lead time)

Validation runs at ``cfg["val_mask_ratio"]`` (default 0: all stations
visible) with a fixed RNG stream, and restores the training mask ratio and
RNG state afterwards.
"""

import math

import torch
import pytorch_lightning as pl

from .mae import StationMAE
from .embeddings import TARGET_VARIABLE_NAMES, NUM_TARGET_VARIABLES


class StationMAELightning(pl.LightningModule):
    """
    Args:
        model: StationMAE instance.
        cfg:   dict with lr, min_lr, weight_decay, epochs, warmup_epochs,
               val_mask_ratio, obs_stats_std and the architecture settings
               (saved into the checkpoint as ``hyper_parameters["cfg"]``).
    """

    def __init__(self, model: StationMAE, cfg: dict):
        super().__init__()
        self.model = model
        self.cfg   = cfg
        self.save_hyperparameters(ignore=["model"])

    @staticmethod
    def _unpack_batch(batch: dict) -> tuple:
        spatial = batch["spatial"]
        if spatial.dim() == 3:
            spatial = spatial[0]                    # static features are shared
        return (batch["x"], batch["x_mask"], spatial, batch["x_hours"],
                batch["y"], batch["y_mask"], batch["y_hours"], batch["delta_steps"])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps = self._unpack_batch(batch)

        loss, preds, _ = self.model.forward_multi_delta(
            x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"],
                 on_step=True, on_epoch=False)

        # Epoch MAE at k=1 (+30 min), the same lead as val/overall_mae.
        with torch.no_grad():
            k  = min(1, preds.shape[1] - 1)
            Vt = NUM_TARGET_VARIABLES
            m  = y_mask[:, k, :, :Vt].bool()
            if m.any():
                err = (preds[:, k] - y[:, k, :, :Vt]).abs()
                self.log("train/overall_mae", err[m].mean(),
                         on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._train_mask_ratio = self.model.encoder.mask_ratio
        self.model.encoder.mask_ratio = float(self.cfg.get("val_mask_ratio", 0.0))
        self._saved_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self._saved_cuda_rng_state = torch.cuda.get_rng_state_all()
        torch.manual_seed(42)                       # reproducible validation masks

    def on_validation_epoch_end(self) -> None:
        self.model.encoder.mask_ratio = self._train_mask_ratio
        torch.set_rng_state(self._saved_rng_state)
        if torch.cuda.is_available() and hasattr(self, "_saved_cuda_rng_state"):
            torch.cuda.set_rng_state_all(self._saved_cuda_rng_state)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps = self._unpack_batch(batch)
        Vt = NUM_TARGET_VARIABLES

        # Validation is scored at one lead, k=1 (+30 min): k=0 is the
        # reconstruction of the last input step and is trivial for visible
        # stations. The forward pass therefore uses the single-delta path.
        k = min(1, delta_steps.shape[1] - 1)
        loss, preds, _ = self.model(
            x, x_mask, spatial, x_hours, y[:, k], y_mask[:, k], y_hours[:, k], delta_steps[:, k])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        p = preds.reshape(-1, Vt)
        t = y[:, k, :, :Vt].reshape(-1, Vt)
        m = y_mask[:, k, :, :Vt].reshape(-1, Vt).bool()

        obs_std = None
        std_raw = self.cfg.get("obs_stats_std")
        if std_raw is not None:
            std_t = torch.tensor(std_raw, dtype=p.dtype, device=p.device)
            obs_std = (std_t.mean(dim=0) if std_t.dim() == 2 else std_t)[:Vt]

        for v, name in enumerate(TARGET_VARIABLE_NAMES):
            if m[:, v].any():
                mae = (p[m[:, v], v] - t[m[:, v], v]).abs().mean()
                self.log(f"val/{name}_mae", mae, on_epoch=True, sync_dist=True)
                if obs_std is not None:
                    self.log(f"val/{name}_mae_phys", mae * obs_std[v],
                             on_epoch=True, sync_dist=True)
        if m.any():
            self.log("val/overall_mae", (p[m] - t[m]).abs().mean(),
                     on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and delta_steps.shape[1] > 1:
            with torch.no_grad():
                _, preds_far, _ = self.model(
                    x, x_mask, spatial, x_hours, y[:, -1], y_mask[:, -1],
                    y_hours[:, -1], delta_steps[:, -1])
            self.log("val/horizon_sensitivity", (preds_far - preds).abs().mean(),
                     on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimiser and schedule
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        lr            = self.cfg.get("lr", 1e-4)
        min_lr        = self.cfg.get("min_lr", 0.0)
        weight_decay  = self.cfg.get("weight_decay", 0.05)
        warmup_epochs = self.cfg.get("warmup_epochs", 5)

        # No weight decay on normalisation parameters, biases and learned
        # token vectors (classified by module type and shape, not by name).
        norm_ids = {
            id(p)
            for mod in self.model.modules()
            if isinstance(mod, (torch.nn.LayerNorm, torch.nn.GroupNorm,
                                torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
            for p in mod.parameters(recurse=False)
        }
        token_like = ("mask_token", "station_state", "var_absent_embedding", "mlp_b1", "mlp_b2")
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (id(param) in norm_ids or name.endswith(".bias")
                    or name.split(".")[-1] in token_like or param.ndim <= 1):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
        )

        # Linear warm-up, cosine decay to a floor of min_lr.
        total_steps  = self.trainer.estimated_stepping_batches
        epochs       = self.cfg.get("epochs", 100)
        steps_per_ep = max(total_steps // epochs, 1)
        warmup_steps = warmup_epochs * steps_per_ep
        min_ratio    = min_lr / max(lr, 1e-12)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.5 * (1.0 + math.cos(math.pi * progress)), min_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
