"""
model/lstm_baseline.py

Per-station LSTM baseline.

The LSTM sees one station's input window (W steps x 6 variables, optionally
concatenated with the sensor-availability mask) and predicts that station's
5 target variables at all K lead times from its final hidden state with one
linear head. It uses no station coordinates, no topography and no other
station, so the difference to the Transformer measures the value of
cross-station information. The station axis is folded into the batch in
``StationLSTMLightning``.
"""

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES, TARGET_VARIABLE_NAMES


class StationLSTM(nn.Module):
    """
    Args:
        num_vars:         input variables per step (6).
        num_target_vars:  predicted variables (5).
        hidden:           LSTM hidden size.
        num_layers:       stacked LSTM layers.
        dropout:          dropout between LSTM layers.
        horizon_steps:    lead time (10-min steps) of each output slot, e.g.
                          [0, 3, ..., 36]; stored as a buffer in the checkpoint.
        use_mask_feature: concatenate the sensor mask to the input (6 -> 12 features).
    """

    def __init__(
        self,
        num_vars:         int = NUM_VARIABLES,
        num_target_vars:  int = NUM_TARGET_VARIABLES,
        hidden:           int = 256,
        num_layers:       int = 3,
        dropout:          float = 0.1,
        num_horizons:     int = 13,
        horizon_steps:    "list[int] | None" = None,
        use_mask_feature: bool = False,
    ):
        super().__init__()
        if horizon_steps is None:
            horizon_steps = list(range(num_horizons))
        horizon_steps = [int(s) for s in horizon_steps]
        assert len(horizon_steps) == len(set(horizon_steps)), "horizon_steps must be unique"

        self.num_target_vars  = num_target_vars
        self.num_horizons     = len(horizon_steps)
        self.use_mask_feature = use_mask_feature
        self.register_buffer("horizon_steps", torch.tensor(horizon_steps, dtype=torch.long))

        in_size = num_vars * (2 if use_mask_feature else 1)
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, self.num_horizons * num_target_vars)

    def forward(self, x: torch.Tensor, x_mask: "torch.Tensor | None" = None) -> torch.Tensor:
        """x, x_mask: (M, W, V) station sequences -> (M, K, V_t)."""
        if self.use_mask_feature and x_mask is not None:
            x = torch.cat([x, x_mask], dim=-1)
        out, _ = self.lstm(x)
        p = self.head(out[:, -1, :])
        return p.view(x.size(0), self.num_horizons, self.num_target_vars)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StationLSTMLightning(pl.LightningModule):
    """
    Folds the station axis into the batch, applies the same masked Huber loss
    as the Transformer and logs the same validation metric names.
    """

    def __init__(self, model: StationLSTM, cfg: dict):
        super().__init__()
        self.model = model
        self.cfg   = cfg
        self.save_hyperparameters(ignore=["model"])
        w = cfg.get("var_weights", [1.0] * model.num_target_vars)
        self.register_buffer("var_weights", torch.tensor(w, dtype=torch.float32))
        self.huber_delta     = float(cfg.get("huber_delta", 1.0))
        self.num_target_vars = model.num_target_vars

    @staticmethod
    def _fold(t: torch.Tensor) -> torch.Tensor:
        """(B, T, N, V) -> (B*N, T, V)."""
        B, T, N, V = t.shape
        return t.permute(0, 2, 1, 3).reshape(B * N, T, V)

    def _unpack(self, batch: dict):
        return (self._fold(batch["x"]), self._fold(batch["x_mask"]),
                self._fold(batch["y"]), self._fold(batch["y_mask"]))

    def _loss(self, preds, y, ymask) -> torch.Tensor:
        Vt  = self.num_target_vars
        err = preds - y[..., :Vt]
        m   = ymask[..., :Vt].to(preds.dtype)
        ad  = err.abs()
        huber = torch.where(ad <= self.huber_delta, 0.5 * err.pow(2),
                            self.huber_delta * (ad - 0.5 * self.huber_delta))
        w = self.var_weights[:Vt].view(1, 1, Vt)
        return (huber * m * w).sum() / (m * w).sum().clamp(min=1.0)

    def training_step(self, batch, batch_idx):
        x, xm, y, ym = self._unpack(batch)
        preds = self.model(x, xm)
        loss  = self._loss(preds, y, ym)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, xm, y, ym = self._unpack(batch)
        preds = self.model(x, xm)
        self.log("val/loss", self._loss(preds, y, ym), on_epoch=True, prog_bar=True, sync_dist=True)

        k  = min(1, preds.shape[1] - 1)             # +30 min, as for the Transformer
        Vt = self.num_target_vars
        p, t, m = preds[:, k], y[:, k, :Vt], ym[:, k, :Vt].bool()

        obs_std = None
        std_raw = self.cfg.get("obs_stats_std")
        if std_raw is not None:
            s = torch.tensor(std_raw, dtype=p.dtype, device=p.device)
            obs_std = (s.mean(dim=0) if s.dim() == 2 else s)[:Vt]

        for v, name in enumerate(TARGET_VARIABLE_NAMES):
            if m[:, v].any():
                mae = (p[m[:, v], v] - t[m[:, v], v]).abs().mean()
                self.log(f"val/{name}_mae", mae, on_epoch=True, sync_dist=True)
                if obs_std is not None:
                    self.log(f"val/{name}_mae_phys", mae * obs_std[v], on_epoch=True, sync_dist=True)
        if m.any():
            self.log("val/overall_mae", (p[m] - t[m]).abs().mean(),
                     on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        lr            = self.cfg.get("lr", 1e-3)
        min_lr        = self.cfg.get("min_lr", 0.0)
        weight_decay  = self.cfg.get("weight_decay", 0.0)
        warmup_epochs = self.cfg.get("warmup_epochs", 3)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=weight_decay)

        total_steps  = self.trainer.estimated_stepping_batches
        epochs       = self.cfg.get("epochs", 50)
        steps_per_ep = max(total_steps // max(epochs, 1), 1)
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
