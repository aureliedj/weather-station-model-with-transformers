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
    train/overall_mae   — epoch-level train MAE     (on_step=False, on_epoch=True)
    train/lr            — learning rate per step    (on_step=True, on_epoch=False)
    val/loss              — epoch validation MSE      (drives ModelCheckpoint / EarlyStopping)
    val/overall_mae       — epoch overall MAE across all target variables
                            (measured at cfg["val_mask_ratio"], default 0.0 =
                             ALL stations visible, matching the LSTM and
                             simple-MAE panels; training masking is unchanged)
    val/{var}_mae         — per-variable MAE (normalized, dimensionless)
    val/{var}_mae_phys    — per-variable MAE in physical units (MAE_norm × obs_std[v])
    (RMSE is deliberately NOT logged during validation — see validation_step.)
                            Physical metrics require cfg["obs_stats_std"] set by main.py.
    val/horizon_sensitivity — mean |pred(delta=36) − pred(delta=3)| on the first val batch.
                            ≈ 0 → decoder output is flat across lead times (DeltaTimeEmbedding not used);
                            > 0.05 → decoder output varies meaningfully with lead time.
"""

import contextlib
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

        # ── train/overall_mae — epoch-level, mirrors val/overall_mae ─────────
        # Use k=1 (delta=3 = 30 min, all stations) to mirror val/overall_mae.
        # k=0 is delta=0 with masked-only loss — not representative of
        # forecasting skill and would make train MAE incomparable to val.
        with torch.no_grad():
            if is_multi:
                _k_train = min(1, preds.shape[1] - 1)   # k=1 → delta=3 (30 min)
                p_flat = preds[:, _k_train].reshape(-1, NUM_TARGET_VARIABLES)
                t_flat = y[:, _k_train, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES)
                m_flat = y_mask[:, _k_train, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES).bool()
            else:
                p_flat = preds.reshape(-1, NUM_TARGET_VARIABLES)
                t_flat = y[:, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES)
                m_flat = y_mask[:, :, :NUM_TARGET_VARIABLES].reshape(-1, NUM_TARGET_VARIABLES).bool()
            if m_flat.sum() > 0:
                mask_any = m_flat.any(dim=-1)
                self.log(
                    "train/overall_mae",
                    (p_flat[mask_any] - t_flat[mask_any]).abs().mean(),
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
                )

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        """
        Initialise two independent RNG streams for dual-seed validation.

        Each validation batch is evaluated twice — once with seed 42 (stream A)
        and once with seed 123 (stream B) — producing two different station-mask
        assignments.  Predictions and metrics are averaged over both passes.

        This halves the variance of val metrics compared to a single random mask,
        making epoch-to-epoch comparisons more reliable without doubling compute
        (each val batch is only ~23 batches total with block-mode val).

        Both streams are advanced consistently across batches so batch N of stream
        A is always reproducible from one training run to the next.

        The training RNG state is saved here and restored in on_validation_epoch_end
        so the training random stream is completely unaffected.
        """
        # ── Validation mask ratio ────────────────────────────────────────
        # Default 0.0: validate with ALL 155 stations visible, so val/* is the
        # same task the LSTM and simple-MAE runs report (a 30-min forecast
        # from the full network). Training keeps its own mask_ratio; this is
        # restored in on_validation_epoch_end.
        self._val_mask_ratio  = float(self.cfg.get("val_mask_ratio", 0.0))
        self._train_mask_ratio = self.model.encoder.mask_ratio
        self.model.encoder.mask_ratio = self._val_mask_ratio

        self._saved_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self._saved_cuda_rng_state = torch.cuda.get_rng_state_all()

        # Initialise stream A (seed 42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self._val_rng_a      = torch.get_rng_state()
        self._val_cuda_rng_a = (torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available() else None)

        # Initialise stream B (seed 123)
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)
        self._val_rng_b      = torch.get_rng_state()
        self._val_cuda_rng_b = (torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available() else None)

        # Activate stream A for the first batch
        torch.set_rng_state(self._val_rng_a)
        if torch.cuda.is_available() and self._val_cuda_rng_a is not None:
            torch.cuda.set_rng_state_all(self._val_cuda_rng_a)

    @torch.no_grad()
    def _sanity_check(self) -> None:
        """
        Per-epoch architecture sanity check on one cached validation batch.

        Catches at TRAINING time the failure classes found in the v10–v13
        post-mortems, instead of days later in the notebook:
          • finiteness      — NaN/Inf predictions
          • dispersion      — pred std / target std per variable
                              (≪ 1 = conditional-mean collapse, v10/v13 symptom)
          • persistence     — Δ>0 RMSE vs last-observation baseline
          • context path    — at Δ=0, VISIBLE stations must beat MASKED ones.
                              visible ≈ masked means the model is not using
                              the observations it can see (the v13 signature).
                              v15: the old input_context pathway is gone, so
                              this now checks the telescopic tokenization +
                              residual head are doing that job instead.
        Logged to wandb as sanity/* and printed as one line. Runs one extra
        multi-delta forward per epoch — negligible cost.
        """
        batch = getattr(self, "_sanity_batch", None)
        if batch is None:
            return
        self._sanity_batch = None
        # Diagnostics need masked stations (ctx_ratio compares visible vs
        # masked), so this runs at the TRAINED mask ratio even though the
        # logged val/* metrics above use val_mask_ratio (0 = all visible).
        _mr_metrics = self.model.encoder.mask_ratio
        self.model.encoder.mask_ratio = getattr(
            self, "_train_mask_ratio", _mr_metrics)
        x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps = (
            self._unpack_batch(batch)
        )
        if delta_steps.dim() != 2 or delta_steps.shape[1] < 2 or y.dim() != 4:
            return  # sanity check needs a multi-delta batch

        _amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                if x.is_cuda else contextlib.nullcontext())
        with _amp:
            _, preds, midx = self.model.forward_multi_delta(
                x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
            )
        preds = preds.float()
        Vt = preds.shape[-1]
        t  = y[..., :Vt]
        m  = y_mask[..., :Vt] > 0.5

        # 1. finiteness
        finite = bool(torch.isfinite(preds).all())

        # 2. dispersion (mean-collapse detector)
        disp = []
        for v in range(Vt):
            mv = m[..., v]
            if mv.any():
                disp.append((preds[..., v][mv].std()
                             / t[..., v][mv].std().clamp(min=1e-6)).item())
        disp_min = min(disp) if disp else float("nan")

        # 3. persistence skill on Δ>0 (norm space; last obs = Δ=0 target)
        K       = preds.shape[1]
        persist = t[:, :1].expand_as(t)
        both    = m & m[:, :1]
        fc      = slice(1, K)
        e, ep   = (preds[:, fc] - t[:, fc])[both[:, fc]], \
                  (persist[:, fc] - t[:, fc])[both[:, fc]]
        skill = float("nan")
        if e.numel() > 0:
            skill = (1.0 - e.pow(2).mean().sqrt()
                     / ep.pow(2).mean().sqrt().clamp(min=1e-6)).item()

        # 4. context path: visible must beat masked at Δ=0
        ctx_ratio = float("nan")
        if (midx is not None and midx.numel() > 0
                and int(delta_steps[0, 0]) == 0):
            B, _, N, _ = preds.shape
            sel = torch.zeros(B, N, dtype=torch.bool, device=preds.device)
            sel.scatter_(1, midx, True)
            e0, m0 = (preds[:, 0] - t[:, 0]).abs(), m[:, 0]
            mm, vv = m0 & sel.unsqueeze(-1), m0 & ~sel.unsqueeze(-1)
            if mm.any() and vv.any():
                ctx_ratio = (e0[vv].mean()
                             / e0[mm].mean().clamp(min=1e-6)).item()

        parts = [
            ("✓" if finite else "⚠ NaN/Inf in preds!"),
            f"disp_min {disp_min:.2f}"
            + ("" if not disp_min == disp_min or disp_min > 0.5 else " ⚠ mean-collapse"),
            f"persist_skill {skill:+.3f}"
            + ("" if not skill == skill or skill > -0.5 else " ⚠ far below persistence"),
            f"ctx_ratio {ctx_ratio:.2f}"
            + ("" if not ctx_ratio == ctx_ratio or ctx_ratio < 0.9
               else " ⚠ visible≈masked — context path?"),
        ]
        self.model.encoder.mask_ratio = _mr_metrics   # restore
        print(f"  [sanity e{self.current_epoch}] " + " | ".join(parts))
        for k_, v_ in [("sanity/dispersion_min", disp_min),
                       ("sanity/persist_skill", skill),
                       ("sanity/ctx_ratio",     ctx_ratio)]:
            if v_ == v_:   # skip NaN
                self.log(k_, v_, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Run the sanity check, then restore the training RNG state saved in
        on_validation_epoch_start. Order matters: the sanity forward consumes
        validation-stream RNG (mask sampling), never the training stream."""
        self._sanity_check()
        # Restore the TRAINING mask ratio (validation ran at val_mask_ratio).
        if hasattr(self, "_train_mask_ratio"):
            self.model.encoder.mask_ratio = self._train_mask_ratio
        if hasattr(self, "_saved_rng_state"):
            torch.set_rng_state(self._saved_rng_state)
        if torch.cuda.is_available() and hasattr(self, "_saved_cuda_rng_state"):
            torch.cuda.set_rng_state_all(self._saved_cuda_rng_state)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        # Cache the first val batch for the per-epoch sanity check
        # (consumed and released in on_validation_epoch_end).
        if batch_idx == 0:
            self._sanity_batch = batch

        x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps = (
            self._unpack_batch(batch)
        )

        # ── Choose validation lead-time ──────────────────────────────────
        # delta_grid = [0, 3, 6, …, 36]  (K=13):
        #   k=0 → delta=0  (reconstruction, masked stations only — not useful as
        #                   a monitoring metric since it covers only ~50% of stations)
        #   k=1 → delta=3  (30 min ahead, all stations — the primary forecast metric)
        # We monitor at k=1 so val/overall_mae reflects genuine forecasting skill.
        _is_multi = (delta_steps.dim() == 2 and delta_steps.shape[1] > 1)
        _k = 1 if _is_multi else 0   # k=1 → delta=3 (30 min); k=0 fallback for single-delta
        ds = delta_steps[:, _k] if _is_multi else delta_steps
        y_ = y[:, _k]       if y.dim()      == 4 else y
        ym = y_mask[:, _k]  if y_mask.dim() == 4 else y_mask
        yh = y_hours[:, _k] if y_hours.dim() == 2 else y_hours

        # ── Forward pass(es) at the VALIDATION mask ratio ─────────────────
        # The encoder mask ratio was switched to cfg["val_mask_ratio"]
        # (default 0.0 = ALL 155 stations visible) in
        # on_validation_epoch_start, so val/* measures the same task as the
        # LSTM and simple-MAE panels: a 30-min forecast with the full network
        # as input. It is restored to the trained ratio afterwards, and the
        # per-epoch sanity check deliberately runs at the TRAINED ratio so
        # ctx_ratio / masked-station diagnostics keep their meaning.
        #
        # Dual-seed averaging only makes sense when masking is stochastic:
        # at ratio 0 there is nothing to sample, both passes would be
        # identical, so a single pass is used (and validation is 2x faster).
        if self._val_mask_ratio > 0:
            # Pass A uses stream A RNG (already active on entry to this step).
            loss_a, preds_a, _ = self.model(
                x, x_mask, spatial, x_hours, y_, ym, yh, ds
            )
            # Checkpoint stream A state, switch to stream B
            self._val_rng_a = torch.get_rng_state()
            if torch.cuda.is_available():
                self._val_cuda_rng_a = torch.cuda.get_rng_state_all()

            torch.set_rng_state(self._val_rng_b)
            if torch.cuda.is_available() and self._val_cuda_rng_b is not None:
                torch.cuda.set_rng_state_all(self._val_cuda_rng_b)

            loss_b, preds_b, _ = self.model(
                x, x_mask, spatial, x_hours, y_, ym, yh, ds
            )
            # Checkpoint stream B, restore stream A for the next batch
            self._val_rng_b = torch.get_rng_state()
            if torch.cuda.is_available():
                self._val_cuda_rng_b = torch.cuda.get_rng_state_all()

            torch.set_rng_state(self._val_rng_a)
            if torch.cuda.is_available() and self._val_cuda_rng_a is not None:
                torch.cuda.set_rng_state_all(self._val_cuda_rng_a)

            # Averaged loss and predictions
            loss  = (loss_a  + loss_b)  * 0.5
            preds = (preds_a + preds_b) * 0.5
        else:
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

        # ── Per-variable MAE (all stations, present sensors only) ────────
        # MAE only: RMSE was dropped from validation logging (project
        # decision) — it duplicates MAE's ranking on these runs while being
        # dominated by a handful of extreme events, and the reported metric
        # is MAE anyway. RMSE is still available downstream from the test
        # dumps (Test_Results_Exploration.ipynb computes MAE, MSE and RMSE
        # from predictions.pt).
        for v, var_name in enumerate(TARGET_VARIABLE_NAMES):
            m = masks_all[:, v]
            if m.sum() > 0:
                p, t = preds_all[m, v], targets_all[m, v]
                mae_norm = (p - t).abs().mean()
                self.log(f"val/{var_name}_mae", mae_norm, on_epoch=True, sync_dist=True)
                # Physical units: multiply normalized error by the training-split std
                if obs_std is not None:
                    self.log(f"val/{var_name}_mae_phys", mae_norm * obs_std[v],
                             on_epoch=True, sync_dist=True)

        # ── Overall MAE across all variables and stations ──────────────
        if masks_all.sum() > 0:
            pf = preds_all[masks_all]
            tf = targets_all[masks_all]
            self.log("val/overall_mae", (pf - tf).abs().mean(),
                     on_epoch=True, prog_bar=True, sync_dist=True)

        # ── Horizon-sensitivity probe (first batch of each val epoch only) ─
        # Measures mean |pred(delta=6h) − pred(delta=30min)| to verify the
        # decoder modulates its output with lead time.
        #   ≈ 0.0  → decoder ignores DeltaTimeEmbedding (constant output)
        #   > 0.05 → decoder varies meaningfully across horizons
        # Uses stream A's current RNG state (advances independently of A/B above).
        if batch_idx == 0 and _is_multi and delta_steps.shape[1] > 1:
            _k_far = delta_steps.shape[1] - 1   # last K index (delta=36 = 6h)
            ds_far = delta_steps[:, _k_far]
            y_far  = y[:, _k_far]      if y.dim()      == 4 else y
            ym_far = y_mask[:, _k_far] if y_mask.dim() == 4 else y_mask
            yh_far = y_hours[:, _k_far] if y_hours.dim() == 2 else y_hours
            with torch.no_grad():
                _, preds_far, _ = self.model(
                    x, x_mask, spatial, x_hours, y_far, ym_far, yh_far, ds_far
                )
            self.log("val/horizon_sensitivity", (preds_far - preds).abs().mean(),
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
        # Biases, LayerNorm params, and token-like learned vectors are excluded
        # from weight decay. Decaying norms or token embeddings harms training.
        #
        # This used to match on NAME SUBSTRINGS — ("bias", "norm", "embedding",
        # "mask_token", "lambdas") — which silently missed three classes,
        # because a parameter's qualified name does not always contain the word
        # describing what it is:
        #
        #   *_emb.proj.2.{weight,bias}  the LayerNorm INSIDE every embedding
        #                               MLP (PositionalEmbedding, StationEmbedding,
        #                               TemporalEmbedding, DeltaTimeEmbedding,
        #                               StepIndexEmbedding). It sits at index 2 of
        #                               an nn.Sequential, so its name is
        #                               "...proj.2.weight" — no "norm" anywhere.
        #                               Ten LayerNorm gains were being decayed.
        #   decoder.station_state       learned 2-vector "visible/masked" code,
        #                               exactly the same class of object as
        #                               mask_token, but without the substring.
        #   var_proj.mlp_b1 / mlp_b2    biases of the batched per-variable MLP,
        #                               named "b1"/"b2" rather than "bias".
        #
        # Classify by MODULE TYPE and role instead, which cannot drift as
        # parameters are renamed (the recipe used by timm / nanoGPT / the MAE
        # reference implementation).
        _norm_param_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm,
                              torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
            for p in m.parameters(recurse=False)
        }
        # Learned token / code vectors: no decay, same rationale as mask_token.
        _token_like = ("mask_token", "station_state", "var_absent_embedding",
                       "slot_emb", "var_biases", "mlp_b1", "mlp_b2")

        decay_params, nodecay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = (
                id(param) in _norm_param_ids            # any normalisation gain/bias
                or name.endswith(".bias")               # every nn.Linear / MHA bias
                or name.split(".")[-1] in _token_like   # learned token/code vectors
                or param.ndim <= 1                      # catch-all for 1-D params
            )
            (nodecay_params if no_decay else decay_params).append(param)

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
