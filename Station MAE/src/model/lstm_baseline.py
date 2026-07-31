"""
model/lstm_baseline.py

Vanilla per-station LSTM baseline for the Station-MAE project.

Purpose
-------
A deliberately simple, *spatially-blind* recurrent baseline to answer the
question "is the transformer actually good, or would a plain per-station
temporal model do just as well?".  It processes ONE station's 12 h history
(W=72 steps × 6 variables) and predicts that same station's target variables
at every forecast horizon in a single shot.

Deliberate design choices (see meeting notes)
--------------------------------------------
  * No spatial / topographic features — the LSTM sees only the variable history
    of a single station, so `transformer − LSTM` isolates the value of the
    transformer's spatial / joint modelling.
  * No station masking — the baseline is a pure forecaster; compare it against
    the transformer's mr0.00 (all-stations-visible) numbers.
  * No positional / step-index embedding — the LSTM encodes order through its
    recurrence.
  * No delta-time embedding — a multi-horizon output head predicts all K
    lead-times at once (no autoregressive rollout, no error accumulation).

Output-grid options (see StationLSTM.forward)
--------------------------------------------
The head is a flat `Linear(hidden → K·V_target)`: slot k is an independent set
of weights, so **every slot must be supervised or it learns nothing**.  Two
configurations are therefore supported, both of which supervise all K slots:

  A. "grid"  — K = 13, Δ = 0, 3, …, 36 steps (30-min spacing, 0–6 h).
               Head width == supervised horizons == evaluated horizons.
               Standard direct-multi-step (DLinear / PatchTST / Weather-5K).
               Exactly matches the transformer's output grid → paired comparison.

  C. "dense" — K = 37, Δ = 0, 1, …, 36 steps (native 10-min resolution).
               All 37 slots supervised; the 13 that coincide with the
               transformer grid are selected at EVALUATION time via
               `select_horizons()`.  Gives a full-resolution error curve, but
               trains on 2.8× denser supervision than the transformer, so the
               comparison is no longer strictly like-for-like.

The rejected middle ground — emit 37, supervise 13 — would leave 24 slots with
zero gradient; their weights never move off initialisation.  Not supported.

The station dimension N is folded into the batch (see StationLSTMLightning), so
one DataLoader batch of B windows becomes B×N independent station sequences —
the whole thing runs on cuDNN and is far cheaper than the transformer.
"""

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .embeddings import (
    NUM_VARIABLES,
    NUM_TARGET_VARIABLES,
    TARGET_VARIABLE_NAMES,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StationLSTM(nn.Module):
    """
    Per-station multi-horizon LSTM.

    Args:
        num_vars:        Input variables per timestep (6, incl. precipitation).
        num_target_vars: Predicted variables (5, excludes precipitation).
        hidden:          LSTM hidden size (embedding space — set ≥ transformer d_model).
        num_layers:      Stacked LSTM layers (default 3).
        dropout:         Dropout between LSTM layers (ignored if num_layers==1).
        num_horizons:    K forecast horizons predicted in one shot (e.g. 13).
                         Ignored when `horizon_steps` is given.
        horizon_steps:   Lead-time (in 10-min steps) that each output slot
                         predicts, e.g. [0, 3, …, 36] (mode A, K=13) or
                         [0, 1, …, 36] (mode C, K=37).  Pass the dataset's
                         `delta_grid` so the model records its own output
                         semantics — this is what makes `select_horizons()`
                         safe.  If None, falls back to [0 … num_horizons-1].
        use_mask_feature: If True, concatenate the sensor-availability mask to the
                          input so the model can distinguish a true 0 (station mean
                          in normalised space) from a missing sensor. Default False
                          keeps the baseline vanilla.
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
        # Non-learnable record of what each output slot means (survives ckpt save/load)
        self.register_buffer("horizon_steps",
                             torch.tensor(horizon_steps, dtype=torch.long),
                             persistent=True)

        in_size = num_vars * (2 if use_mask_feature else 1)
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Multi-horizon head: last hidden state → K × num_target_vars
        self.head = nn.Linear(hidden, self.num_horizons * num_target_vars)

    def forward(self, x: torch.Tensor, x_mask: "torch.Tensor | None" = None) -> torch.Tensor:
        """
        Args:
            x:      (M, W, V)  station sequences (M = batch × stations).
            x_mask: (M, W, V)  sensor availability, only used if use_mask_feature.

        Returns:
            (M, K, num_target_vars) — slot k is the forecast at lead-time
            `self.horizon_steps[k]` (in 10-min steps).

        Shapes, for the two supported configurations
        (W=72, V=6, V_target=5, hidden=1024, M = B × 155 stations):

            mode A — "grid"   horizon_steps = [0, 3, 6, …, 36]   K = 13
              x            (M, 72, 12)      6 vars + 6 mask flags
              lstm out     (M, 72, 1024)    3 stacked layers
              h            (M, 1024)        final step only
              head         Linear(1024 → 65)          65 = 13 × 5
              return       (M, 13, 5)       Δ = 0, 30, 60, …, 360 min

            mode C — "dense"  horizon_steps = [0, 1, 2, …, 36]   K = 37
              x            (M, 72, 12)
              lstm out     (M, 72, 1024)
              h            (M, 1024)
              head         Linear(1024 → 185)         185 = 37 × 5
              return       (M, 37, 5)       Δ = 0, 10, 20, …, 360 min
              → then `select_horizons(preds, [0,3,…,36])` → (M, 13, 5) at eval

        The head is the ONLY difference between the modes: 65 vs 185 outputs
        (+123 k params). The recurrence is identical.
        """
        if self.use_mask_feature and x_mask is not None:
            x = torch.cat([x, x_mask], dim=-1)
        out, _ = self.lstm(x)                 # (M, W, hidden)
        h = out[:, -1, :]                     # (M, hidden) — final-step summary
        p = self.head(h)                      # (M, K·Vt)
        return p.view(x.size(0), self.num_horizons, self.num_target_vars)

    # ── mode C helper: project a dense output onto a coarser evaluation grid ──
    def select_horizons(self, preds: torch.Tensor, wanted_steps) -> torch.Tensor:
        """
        Subselect output slots onto `wanted_steps` (lead-times in 10-min steps).

        Used in mode C to align the 37-horizon output with the transformer's
        13-horizon grid at EVALUATION time. Every requested step must exist in
        `self.horizon_steps` — no interpolation, no silent fallback.

        Args:
            preds:        (..., K, V_target) as returned by forward().
            wanted_steps: iterable of lead-times, e.g. [0, 3, …, 36].

        Returns:
            (..., len(wanted_steps), V_target)
        """
        have = self.horizon_steps.tolist()
        pos  = {s: i for i, s in enumerate(have)}
        missing = [int(s) for s in wanted_steps if int(s) not in pos]
        if missing:
            raise ValueError(
                f"horizons {missing} are not predicted by this model "
                f"(it emits Δ={have}). Retrain with a grid that contains them."
            )
        idx = torch.tensor([pos[int(s)] for s in wanted_steps],
                           dtype=torch.long, device=preds.device)
        return preds.index_select(-2, idx)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------------

class StationLSTMLightning(pl.LightningModule):
    """
    Lightning wrapper for StationLSTM.

    Folds the station dimension into the batch, applies a variable-weighted Huber
    loss over present sensors, and logs the SAME val metric names as the
    transformer (val/loss, val/overall_rmse, val/{var}_rmse, …) so both models
    slot into the same monitor / comparison.
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

    # ── batch reshaping: (B, W, N, V) → (B·N, W, V) ──────────────────────
    @staticmethod
    def _fold(t: torch.Tensor) -> torch.Tensor:
        """(B, T, N, V) → (B·N, T, V) — station dim folded into batch."""
        B, T, N, V = t.shape
        return t.permute(0, 2, 1, 3).reshape(B * N, T, V)

    def _unpack(self, batch: dict):
        x  = self._fold(batch["x"])            # (M, W, V)
        xm = self._fold(batch["x_mask"])       # (M, W, V)
        y  = self._fold(batch["y"])            # (M, K, V)
        ym = self._fold(batch["y_mask"])       # (M, K, V)
        return x, xm, y, ym

    # ── variable-weighted, masked Huber loss over all horizons ───────────
    def _loss(self, preds: torch.Tensor, y: torch.Tensor, ymask: torch.Tensor) -> torch.Tensor:
        Vt   = self.num_target_vars
        yt   = y[..., :Vt]
        m    = ymask[..., :Vt].to(preds.dtype)
        err  = preds - yt
        ad   = err.abs()
        huber = torch.where(ad <= self.huber_delta,
                            0.5 * err.pow(2),
                            self.huber_delta * (ad - 0.5 * self.huber_delta))
        w   = self.var_weights[:Vt].view(1, 1, Vt)
        num = (huber * m * w).sum()
        den = (m * w).sum().clamp(min=1.0)
        return num / den

    def training_step(self, batch, batch_idx):
        x, xm, y, ym = self._unpack(batch)
        preds = self.model(x, xm)
        loss  = self._loss(preds, y, ym)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/lr",   lr,   on_step=True, on_epoch=False)
        with torch.no_grad():
            k = min(1, preds.shape[1] - 1)
            p = preds[:, k]; t = y[:, k, :self.num_target_vars]
            mm = ym[:, k, :self.num_target_vars].bool()
            if mm.any():
                self.log("train/overall_rmse",
                         (p[mm] - t[mm]).pow(2).mean().sqrt(),
                         on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, xm, y, ym = self._unpack(batch)
        preds = self.model(x, xm)
        self.log("val/loss", self._loss(preds, y, ym),
                 on_epoch=True, prog_bar=True, sync_dist=True)

        # Primary monitoring horizon: k=1 (30 min) — matches the transformer.
        k = min(1, preds.shape[1] - 1)
        p = preds[:, k]                                   # (M, Vt)
        t = y[:, k, :self.num_target_vars]
        m = ym[:, k, :self.num_target_vars].bool()

        # per-variable std for physical metrics (mean over stations if (N,V))
        _std_raw = self.cfg.get("obs_stats_std", None)
        obs_std = None
        if _std_raw is not None:
            _s = torch.tensor(_std_raw, dtype=p.dtype, device=p.device)
            obs_std = (_s.mean(dim=0) if _s.dim() == 2 else _s)[:self.num_target_vars]

        for v, name in enumerate(TARGET_VARIABLE_NAMES):
            mm = m[:, v]
            if mm.any():
                pp, tt = p[mm, v], t[mm, v]
                rmse = (pp - tt).pow(2).mean().sqrt()
                mae  = (pp - tt).abs().mean()
                self.log(f"val/{name}_rmse", rmse, on_epoch=True, sync_dist=True)
                self.log(f"val/{name}_mae",  mae,  on_epoch=True, sync_dist=True)
                if obs_std is not None:
                    self.log(f"val/{name}_rmse_phys", rmse * obs_std[v], on_epoch=True, sync_dist=True)
                    self.log(f"val/{name}_mae_phys",  mae  * obs_std[v], on_epoch=True, sync_dist=True)

        if m.any():
            self.log("val/overall_rmse", (p[m] - t[m]).pow(2).mean().sqrt(),
                     on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/overall_mae", (p[m] - t[m]).abs().mean(),
                     on_epoch=True, sync_dist=True)

    # ── optimizer + warmup-cosine schedule (AdamW or RMSprop) ────────────
    def configure_optimizers(self):
        lr            = self.cfg.get("lr", 1e-3)
        min_lr        = self.cfg.get("min_lr", 0.0)
        weight_decay  = self.cfg.get("weight_decay", 0.0)
        warmup_epochs = self.cfg.get("warmup_epochs", 3)
        opt_name      = self.cfg.get("optimizer", "adamw").lower()

        params = [pp for pp in self.model.parameters() if pp.requires_grad]
        if opt_name == "rmsprop":
            # RMSprop — classic choice for RNNs (see torch.optim.RMSprop docs).
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-8,
                                            weight_decay=weight_decay, momentum=0.0)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=weight_decay)

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
