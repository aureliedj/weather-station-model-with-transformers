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

WandB metric names
------------------
Lightning adds _step / _epoch suffixes automatically when BOTH on_step=True
and on_epoch=True are set on the same metric — so "train/loss" would appear
as "train/loss_step" and "train/loss_epoch" in WandB.  To keep names clean,
each metric is logged with exactly one mode:

    train/loss          — per-step training MSE     (on_step=True, on_epoch=False)
    train/overall_rmse  — epoch-level train RMSE    (on_step=False, on_epoch=True)
    train/lr            — learning rate per step    (on_step=True, on_epoch=False)
    val/loss              — epoch validation MSE      (drives ModelCheckpoint / EarlyStopping)
    val/overall_rmse      — epoch overall RMSE across all target variables
    val/{var}_rmse        — per-variable RMSE (normalized, dimensionless)
    val/{var}_mae         — per-variable MAE  (normalized, dimensionless)
    val/{var}_rmse_phys   — per-variable RMSE in physical units (RMSE_norm × obs_std[v])
    val/{var}_mae_phys    — per-variable MAE  in physical units (MAE_norm  × obs_std[v])
                            Physical metrics require cfg["obs_stats_std"] set by main.py.
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

        loss, preds, _ = forward_fn(
            x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
        )

        lr = self.optimizers().param_groups[0]["lr"]

        # on_step=True, on_epoch=False → metric appears as "train/loss" in WandB
        # (no _step suffix — Lightning only adds suffixes when both modes are True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/lr",   lr,   on_step=True, on_epoch=False, prog_bar=False)

        # ── train/overall_rmse — epoch-level, mirrors val/overall_rmse ────────
        # Use first delta for consistent comparison with val (which uses K=1).
        with torch.no_grad():
            if is_multi:
                p_flat = preds[:, 0].reshape(-1, NUM_TARGET_VARIABLES)
                t_flat = y[:, 0, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES)
                m_flat = y_mask[:, 0, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES).bool()
            else:
                p_flat = preds.reshape(-1, NUM_TARGET_VARIABLES)
                t_flat = y[:, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES)
                m_flat = y_mask[:, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES).bool()
            if m_flat.sum() > 0:
                mask_any = m_flat.any(dim=-1)
                self.log(
                    "train/overall_rmse",
                    (p_flat[mask_any] - t_flat[mask_any]).pow(2).mean().sqrt(),
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
                )

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        """
        Fix the global RNG state before every validation epoch.

        The encoder's _mask_stations() uses torch.rand() to sample which
        stations to mask.  Without a fixed seed this differs every epoch,
        making validation metrics subtly noisy (some epochs happen to mask
        easier/harder stations).  Fixing the seed guarantees identical station
        masks across epochs so val/loss changes reflect model improvement, not
        sampling noise.

        Strategy: save the current training RNG state, then set a fixed seed.
        on_validation_epoch_end restores the original state so the training
        random stream is completely unaffected.
        """
        self._saved_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self._saved_cuda_rng_state = torch.cuda.get_rng_state_all()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def on_validation_epoch_end(self) -> None:
        """Restore the training RNG state saved in on_validation_epoch_start."""
        if hasattr(self, "_saved_rng_state"):
            torch.set_rng_state(self._saved_rng_state)
        if torch.cuda.is_available() and hasattr(self, "_saved_cuda_rng_state"):
            torch.cuda.set_rng_state_all(self._saved_cuda_rng_state)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps = (
            self._unpack_batch(batch)
        )

        # Validation always uses a single lead-time (first delta if multi)
        ds = delta_steps[:, 0] if delta_steps.dim() == 2 else delta_steps
        y_ = y[:, 0]      if y.dim()      == 4 else y
        ym = y_mask[:, 0] if y_mask.dim() == 4 else y_mask
        yh = y_hours[:, 0] if y_hours.dim() == 2 else y_hours

        loss, preds, _ = self.model(
            x, x_mask, spatial, x_hours, y_, ym, yh, ds
        )

        # Metrics over ALL N stations wherever sensors are present —
        # consistent with the training objective which also covers all stations.
        B, N = preds.shape[:2]
        y_target      = y_[:, :, :NUM_TARGET_VARIABLES]    # (B, N, 5)
        y_mask_target = ym[:, :, :NUM_TARGET_VARIABLES]    # (B, N, 5)

        # Flatten (B, N, V) → (B*N, V) for metric computation
        preds_all   = preds.reshape(B * N, NUM_TARGET_VARIABLES)
        targets_all = y_target.reshape(B * N, NUM_TARGET_VARIABLES)
        masks_all   = y_mask_target.reshape(B * N, NUM_TARGET_VARIABLES).bool()

        # ── Log val loss (drives ModelCheckpoint + EarlyStopping) ──────
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # ── Per-variable std for unnormalization ────────────────────────
        # obs_stats_std is stored in cfg by main.py.
        # Old format: list[float] of length V  → (V,) tensor
        # New format: (N, V) nested list       → mean across stations for monitoring
        # If absent (old checkpoints), fall back to 1.0 (metrics stay normalised).
        _obs_std_raw = self.cfg.get("obs_stats_std", None)
        if _obs_std_raw is not None:
            _std_t = torch.tensor(_obs_std_raw, dtype=preds_all.dtype,
                                  device=preds_all.device)
            if _std_t.dim() == 2:
                # (N, V) per-station stats — average over stations for a single
                # representative std per variable (used only for monitoring)
                obs_std = _std_t.mean(dim=0)[:NUM_TARGET_VARIABLES]   # (V,)
            else:
                # Legacy (V,) global stats — use directly
                obs_std = _std_t[:NUM_TARGET_VARIABLES]                # (V,)
        else:
            obs_std = None

        # ── Per-variable RMSE / MAE (all stations, present sensors only) ─
        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            m = masks_all[:, v]
            if m.sum() > 0:
                p, t = preds_all[m, v], targets_all[m, v]
                rmse_norm = (p - t).pow(2).mean().sqrt()
                mae_norm  = (p - t).abs().mean()
                self.log(f"val/{var_name}_rmse", rmse_norm, on_epoch=True, sync_dist=True)
                self.log(f"val/{var_name}_mae",  mae_norm,  on_epoch=True, sync_dist=True)
                # Physical units: multiply normalized error by the training-split std
                if obs_std is not None:
                    std_v = obs_std[v]
                    self.log(f"val/{var_name}_rmse_phys", rmse_norm * std_v,
                             on_epoch=True, sync_dist=True)
                    self.log(f"val/{var_name}_mae_phys",  mae_norm  * std_v,
                             on_epoch=True, sync_dist=True)

        # ── Overall RMSE across all variables and stations ─────────────
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
        min_lr        = self.cfg.get("min_lr",        0.0)
        weight_decay  = self.cfg.get("weight_decay",  0.05)
        warmup_epochs = self.cfg.get("warmup_epochs", 5)

        # ── Parameter groups: standard transformer recipe ──────────────────
        # Biases, LayerNorm params, embeddings, and mask_token are excluded
        # from weight decay.  Decaying embeddings or norms can harm training.
        decay_params, nodecay_params = [], []
        no_decay_kw = ("bias", "norm", "embedding", "mask_token", "lambdas")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            bucket = nodecay_params if any(kw in name for kw in no_decay_kw) else decay_params
            bucket.append(param)

        # ── Optimizer: AdamW with MAE-tuned betas ─────────────────────────
        # betas=(0.9, 0.95): higher β₂ dampens the squared-gradient running
        # mean less aggressively than the default 0.999.  This is the recipe
        # from the original MAE paper (He et al. 2022) and works well for
        # transformer models trained with cosine-decay schedules.
        # weight_decay=0.05: applied only to weight matrices (not biases /
        # norms / embeddings).  Provides L2 regularisation on model weights.
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # ── LR schedule: linear warmup → cosine decay → floor at min_lr ───
        # Lightning provides estimated_stepping_batches = epochs × steps_per_epoch
        # accounting for gradient accumulation, so the schedule is always correct.
        #
        # min_lr floor: prevents the LR from decaying all the way to zero near
        # the end of training.  Setting min_lr = lr × 0.01 (e.g. 1e-6 for
        # lr=1e-4) keeps fine-tuning capacity in the final epochs and avoids
        # the model stalling when the LR is near numerical precision.
        total_steps   = self.trainer.estimated_stepping_batches
        epochs        = self.cfg.get("epochs", 100)
        steps_per_ep  = max(total_steps // epochs, 1)
        warmup_steps  = warmup_epochs * steps_per_ep
        min_lr_ratio  = min_lr / max(lr, 1e-12)   # LR multiplier floor

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warm-up from 0 → peak LR
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Floor at min_lr so the schedule never decays below it
            return max(cosine, min_lr_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",   # call scheduler.step() after every optimiser step
                "frequency": 1,
            },
        }
