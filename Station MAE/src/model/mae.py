"""
model/mae.py

Full Station-MAE model — ties the encoder and decoder together.

Data flow
---------

   Input window (B, W, N, V)
        │
        │  +  x_mask  +  spatial  +  x_hours
        ▼
   StationMAEEncoder
        │
        │  encoded_vis  (B, W·N_vis, d_model)
        │  masked_indices  (B, N_masked)
        │  visible_indices (B, N_vis)
        ▼
   StationMAEDecoder
        │  + spatial  +  y_hours  +  delta_steps
        ▼
   preds  (B, N, num_target_vars)
        │
        ▼
   _supervised_loss  →  loss   (scalar, MSE on all N stations × present sensors)


Training objective
------------------
We mask a fraction of stations in the encoder (default 50 %).  The decoder
must predict all variables for EVERY station at the target time (t + Δt),
but the gradient signal flows only through the masked stations — visible
stations are used as context, not as supervision targets.

Multi-delta training
--------------------
``forward_multi_delta`` accepts K lead-times per sample (y/y_mask of shape
(B, K, N, V)).  The encoder runs once; the decoder also runs once, processing
all K lead-times in a single pass via N×K query tokens.  This amortises both
the O(L²) encoder attention and the decoder attention across all horizons,
improving gradient diversity while minimising per-step compute.

Inference
---------
Call ``model.predict(…)`` for a no-grad forward pass that returns
(B, N, num_target_vars) predictions without computing the loss.
"""

import torch
import torch.nn as nn

from .encoder import StationMAEEncoder
from .decoder import StationMAEDecoder
from .embeddings import (
    NUM_VARIABLES,
    NUM_TARGET_VARIABLES,
    TEMPORAL_FOURIER_DIM,
    StepIndexEmbedding,
)


class StationMAE(nn.Module):
    """
    Station-MAE: Masked AutoEncoder for weather station data.

    Combines an asymmetric encoder–decoder pair following He et al. (2022)
    adapted for irregular station grids with topographic context and
    Aurora-inspired multi-scale temporal encodings.

    Key design choices
    ------------------
    * Station-level masking  — an entire station's window is hidden from the
      encoder, simulating gap-filling under missing-station scenarios.
    * Three embeddings per token  — variable (what), spatial (where),
      temporal Aurora Fourier (when).
    * Decoder query tokens use a fourth embedding: delta-time (lead).
    * Asymmetric depth  — deep encoder (default 4 layers), shallow decoder
      (default 2 layers) following the MAE recipe.

    Args:
        d_model:          Shared model dimension for encoder and decoder.
        enc_heads:        Encoder attention heads.
        enc_layers:       Encoder transformer depth.
        dec_heads:        Decoder attention heads.
        dec_layers:       Decoder transformer depth (lighter than encoder).
        mlp_ratio:        FFN hidden-dim = d_model × mlp_ratio.
        dropout:          Dropout applied in attention and FFN.
        mask_ratio:       Fraction of stations masked per sample (0–1).
        num_vars:         Input variables per station (6).
        num_target_vars:  Predicted variables per station (5, excludes precip).
        fourier_dim:      Fourier feature dimension for temporal embedding.
        use_checkpoint:          Enable gradient checkpointing on every transformer block.
                                 Recomputes block activations during backprop instead of
                                 storing them — cuts activation VRAM by ~66% at ~33% extra
                                 compute.  Recommended for enc_layers ≥ 6 on ≤ 30 GB GPUs.
        factorised_encoder:      Use FactorisedTransformerBlock (axial attention over the
                                 W×N grid) instead of flat self-attention over W·N tokens.
                                 ~100× cheaper at W=288, N=100.
        cross_attention_decoder: Use CrossAttentionBlock (query tokens cross-attend to
                                 encoder context) instead of concatenated self-attention.
                                 Decoder sequence drops from W·N_vis+N·K to just N·K.
        drop_path_rate:          Maximum stochastic-depth drop probability (linearly
                                 scheduled from 0 at layer 0 to drop_path_rate at the
                                 deepest layer, in both encoder and decoder).
                                 Recommended range: 0.05–0.20.  Default 0.0 = disabled.
        masked_only_loss:        If True, the training loss is computed only over
                                 encoder-masked stations.  Visible stations are used
                                 as context only and receive no gradient.
                                 Appropriate for inpainting (max_delta=0) where visible
                                 stations have a shortcut via their own input window.
        joint_encoder:           If True, use JointSpatioTemporalBlock in the encoder —
                                 full attention over all W×N tokens simultaneously, with
                                 temporal RoPE on Q and K.  Flash Attention keeps VRAM
                                 linear in sequence length.  Takes precedence over
                                 factorised_encoder when both are set.
                                 Recommended with use_checkpoint=True on ≤ 24 GB GPUs.
    """

    def __init__(
        self,
        d_model:          int   = 128,
        enc_heads:        int   = 4,
        enc_layers:       int   = 4,
        dec_heads:        int   = 4,
        dec_layers:       int   = 2,
        mlp_ratio:        float = 4.0,
        dropout:          float = 0.1,
        mask_ratio:       float = 0.5,
        num_vars:         int   = NUM_VARIABLES,
        num_target_vars:  int   = NUM_TARGET_VARIABLES,
        fourier_dim:      int   = TEMPORAL_FOURIER_DIM,
        use_checkpoint:          bool  = False,
        factorised_encoder:      bool  = False,
        encoder_spatial_attn:    bool  = True,
        temporal_window:         int   = 0,
        temporal_patch:          int   = 1,
        cross_attention_decoder: bool  = False,
        drop_path_rate:          float = 0.0,
        masked_only_loss:         bool  = False,
        joint_encoder:            bool  = False,
        input_context_cross_attn: bool  = False,
        delta0_weight:            float = 1.0,
        use_nll_loss:             bool  = False,
        use_persist_norm:         bool  = False,
        window_size:              int   = 72,
    ):
        """
        delta0_weight controls the relative loss weight of the k=0 horizon
        (delta=0, reconstruction of the current timestep) versus all forecast
        horizons (k≥1).

        Justification: at delta=0, visible stations can almost directly copy
        their own last input token — their contribution to the loss is near zero
        and provides almost no gradient signal.  The k=0 target is mainly useful
        for masked stations (gap-filling), but its presence at equal weight with
        the 12 forecast horizons dilutes the forecasting gradient by 1/13.

        delta0_weight=1.0  : uniform (current default, all K horizons equal)
        delta0_weight=0.1  : down-weight k=0 to 1/10 of a forecast horizon
        delta0_weight=0.0  : exclude k=0 entirely; model is a pure forecaster

        use_nll_loss controls whether the per-variable loss is MSE/Huber or the
        heteroscedastic Gaussian NLL (equivalent to CRPS for Gaussian predictions):

            NLL = 0.5 × (err² / σ² + log σ²)
                = 0.5 × (err² × exp(−log_var) + log_var)

        where log_var = log σ² is predicted by a second linear head in the decoder
        (initialised to 0 so σ² = 1 at training start).  The model learns to widen
        its predictive uncertainty for hard samples (large residuals, long horizons,
        spatially isolated masked stations).

        use_nll_loss=False : default — Huber(δ=1.0) for ALL variables with per-variable weights
        use_nll_loss=True  : Gaussian NLL for all variables; enables calibrated uncertainty

        References: Gneiting & Raftery (2007), Andrychowicz et al. (2023, MetNet-3),
                    Bodnar et al. (2024, Aurora §3)

        use_persist_norm divides each variable's per-sensor loss by that variable's
        persistence MSE — the expected squared error of the naive "repeat last
        observed value" baseline, estimated from the validation set before training.

        This converts the loss from raw MSE scale to a skill-score scale:
            L_v_normalised = L_v / MSE_persist_v
        where L_v_normalised = 1.0 means the model matches persistence exactly,
        and L_v_normalised < 1.0 means the model beats it.

        The key effect is automatic gradient rebalancing: variables that are
        spatially easy to predict (pressure, small MSE_persist) are up-weighted
        relative to difficult variables (wind, large MSE_persist), so that all
        variables contribute comparable gradient magnitudes regardless of their
        inherent predictability.

        Call set_persist_mse() with a (num_target_vars,) tensor before training.
        The buffer is saved in checkpoints and restored on resume.

        use_persist_norm=False : default — raw MSE scale (backward compat)
        use_persist_norm=True  : skill-score scale, auto-balances gradient contributions

        References: Rasp & Lerch (2018), Demaeyer et al. (2023, EUPPBench)
        """
        super().__init__()

        self.mask_ratio        = mask_ratio
        self.num_vars          = num_vars
        self.delta0_weight     = float(delta0_weight)
        self.num_target_vars   = num_target_vars
        self.masked_only_loss  = masked_only_loss
        self.use_nll_loss      = use_nll_loss
        self.use_persist_norm  = use_persist_norm

        # Persistence MSE normaliser — one scalar per target variable.
        # Initialised to 1.0 so the loss is unchanged until set_persist_mse()
        # is called.  Saved/restored automatically via Lightning checkpoints.
        self.register_buffer(
            "persist_mse",
            torch.ones(num_target_vars, dtype=torch.float32),
        )

        # ── Variable-weighted Huber loss ──────────────────────────────────
        # Weights: [temperature, pressure, humidity, wind_u, wind_v]
        #
        # DOWN-weighted noisy variables (v12): only the weight vector changes vs
        # v11's uniform [1,1,1,1,1] — Huber δ=1.0 for all variables is unchanged,
        # so this stays a clean A/B on the weights alone.
        #
        # Diagnosis from v11's wandb curves: temperature and pressure keep
        # improving and do NOT overfit, while humidity and wind_u/wind_v bottom
        # out ~epoch 30-40 and then degrade (val RMSE rises).  v10 made this worse
        # by UP-weighting wind (1.5x); the correct direction is the opposite —
        # DOWN-weight the noisy, heavy-tailed channels so the shared encoder
        # spends less capacity fitting unpredictable gust / saturation noise.
        #
        # Weights [temperature, pressure, humidity, wind_u, wind_v]:
        #   temperature 1.0  — reference, key variable, well-behaved
        #   pressure    1.0  — trivial/easy (tiny residuals; weight ~irrelevant)
        #   humidity    0.7  — overfits, bounded/skewed → moderate down-weight
        #   wind_u/v    0.5  — heaviest overfit, noise-limited → strongest down-weight
        _w = torch.tensor([1.0, 1.0, 0.7, 0.5, 0.5], dtype=torch.float32)
        self.register_buffer("var_weights", _w)
        self.huber_delta = 1.0   # Huber δ in normalised space (= 1 std-dev)
        # Huber is applied to ALL variables (not just wind) — see _supervised_loss.

        # Shared step-index embedding — one instance, passed to both encoder and
        # decoder, so "encoder step k" and "decoder step k" are provably the
        # same vector rather than two independently-trained approximations of
        # the same idea. See StepIndexEmbedding usage notes in encoder.py /
        # decoder.py for why this needs to be a shared instance, not just a
        # matching Fourier-basis formula.
        shared_step_emb = StepIndexEmbedding(d_model=d_model, dropout=dropout)

        self.encoder = StationMAEEncoder(
            d_model=d_model,
            num_heads=enc_heads,
            num_layers=enc_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mask_ratio=mask_ratio,
            num_vars=num_vars,
            fourier_dim=fourier_dim,
            use_checkpoint=use_checkpoint,
            factorised=factorised_encoder,
            spatial_attn=encoder_spatial_attn,
            temporal_window=temporal_window,
            temporal_patch=temporal_patch,
            drop_path_rate=drop_path_rate,
            joint=joint_encoder,
            step_emb=shared_step_emb,
        )

        self.decoder = StationMAEDecoder(
            d_model=d_model,
            num_heads=dec_heads,
            num_layers=dec_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_vars=num_vars,
            num_target_vars=num_target_vars,
            fourier_dim=fourier_dim,
            use_checkpoint=use_checkpoint,
            cross_attention=cross_attention_decoder,
            drop_path_rate=drop_path_rate,
            input_context_cross_attn=input_context_cross_attn,
            predict_uncertainty=use_nll_loss,
            window_size=window_size,
            step_emb=shared_step_emb,
        )

    # ------------------------------------------------------------------
    # Single-delta training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:           torch.Tensor,   # (B, W, N, V)
        x_mask:      torch.Tensor,   # (B, W, N, V)
        spatial:     torch.Tensor,   # (N, 15) or (B, N, 15)
        x_hours:     torch.Tensor,   # (B, W)
        y:           torch.Tensor,   # (B, N, V)
        y_mask:      torch.Tensor,   # (B, N, V)
        y_hours:     torch.Tensor,   # (B,)
        delta_steps: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full training forward pass for a single lead-time per sample.

        Returns:
            loss:           scalar — MSE on all N stations (present sensors only)
            preds:          (B, N, num_target_vars)
            masked_indices: (B, N_masked)   — encoder mask, used by caller for analysis
        """
        encoded, masked_idx, _, input_ctx = self.encoder(x, x_mask, spatial, x_hours)
        decoder_out = self.decoder(encoded, spatial, y_hours, delta_steps, input_context=input_ctx)
        if self.use_nll_loss:
            preds, log_var = decoder_out
        else:
            preds, log_var = decoder_out, None

        y_target      = y[:, :, :self.num_target_vars]
        y_mask_target = y_mask[:, :, :self.num_target_vars]
        # Optionally restrict loss to masked stations (inpainting regime)
        _midx = masked_idx if self.masked_only_loss else None
        loss  = self._supervised_loss(preds, y_target, y_mask_target, _midx, log_var=log_var)

        return loss, preds, masked_idx

    # ------------------------------------------------------------------
    # Multi-delta training forward
    # ------------------------------------------------------------------

    def forward_multi_delta(
        self,
        x:           torch.Tensor,   # (B, W, N, V)
        x_mask:      torch.Tensor,   # (B, W, N, V)
        spatial:     torch.Tensor,   # (N, 15) or (B, N, 15)
        x_hours:     torch.Tensor,   # (B, W)
        y:           torch.Tensor,   # (B, K, N, V)
        y_mask:      torch.Tensor,   # (B, K, N, V)
        y_hours:     torch.Tensor,   # (B, K)
        delta_steps: torch.Tensor,   # (B, K)  long tensor
        return_log_var: bool = False,
    ) -> tuple:
        """
        Multi-delta training forward: encoder once, decoder once for all K.

        Args:
            return_log_var: if True, also return the predicted log σ² tensor
                (B, K, N, num_target_vars) — or None when the model was not
                built with use_nll_loss.  Used by the evaluation pipeline to
                persist calibrated uncertainty alongside the point predictions.
                Default False keeps the original 3-tuple for existing callers.

        The encoder processes the input window once (the expensive O(L²) step).
        The decoder also runs once — it builds N×K query tokens (one per
        station × lead-time pair) and processes them all in a single transformer
        pass, returning (B, K, N, num_target_vars) in one shot.

        This amortises both encoder and decoder attention across all K horizons
        and exposes the model to a diverse range of lead-times within every
        gradient step.

        Args:
            x, x_mask, spatial, x_hours : as in forward()
            y:           (B, K, N, V)    — K target snapshots per sample
            y_mask:      (B, K, N, V)    — sensor availability per target
            y_hours:     (B, K)          — hours-since-epoch for each target
            delta_steps: (B, K)          — lead-time (10-min steps) per target

        Returns:
            loss:           scalar — horizon-weighted MSE over K lead-times
            preds:          (B, K, N, num_target_vars)
            masked_indices: (B, N_masked)
        """
        K = delta_steps.shape[1]

        # ── Encoder: runs once ──────────────────────────────────────────
        encoded, masked_idx, _, input_ctx = self.encoder(x, x_mask, spatial, x_hours)
        # encoded: (B, W*N_vis, d_model)

        # ── Decoder: runs once for all K lead-times ──────────────────────
        decoder_out = self.decoder(encoded, spatial, y_hours, delta_steps, input_context=input_ctx)
        if self.use_nll_loss:
            preds_all, log_var_all = decoder_out
        else:
            preds_all, log_var_all = decoder_out, None
        # preds_all: (B, K, N, num_target_vars)
        # log_var_all: (B, K, N, num_target_vars)  or  None

        # ── Horizon-weighted loss ─────────────────────────────────────────
        # Uniform weights across all K horizons (normalised to sum=1).
        # delta0_weight is applied to k=0 when delta=0 is the first grid entry,
        # though with masked-only supervision at k=0 the gradient is already
        # naturally smaller (~50% of stations), so delta0_weight=1.0 is preferred.
        h_weights = encoded.new_ones(K)
        if delta_steps[0, 0].item() == 0:
            h_weights[0] = self.delta0_weight
        h_weights = h_weights / h_weights.sum()   # normalise → sums to 1

        # ── Per-horizon masking strategy ──────────────────────────────────
        # delta=0 (inpainting / reconstruction):
        #   Supervision on MASKED stations only.  Visible stations have a direct
        #   shortcut through input_context — their predictions are trivially close
        #   to their own input values, contributing near-zero loss and gradient.
        #   Limiting to masked stations gives a pure gap-filling signal.
        #
        # delta>0 (forecasting):
        #   Supervision on ALL stations unless masked_only_loss is set globally.
        #   Both visible and masked stations provide a genuine forecast gradient
        #   — no shortcut exists at any horizon past the last input step.
        loss_acc = encoded.new_zeros(())
        for k in range(K):
            y_target_k      = y[:, k, :, :self.num_target_vars]       # (B, N, num_target_vars)
            y_mask_target_k = y_mask[:, k, :, :self.num_target_vars]  # (B, N, num_target_vars)
            lv_k = log_var_all[:, k] if log_var_all is not None else None

            # k=0 and delta=0: always restrict to masked stations
            if k == 0 and delta_steps[0, 0].item() == 0:
                _midx_k = masked_idx
            else:
                _midx_k = masked_idx if self.masked_only_loss else None

            loss_acc = loss_acc + h_weights[k] * self._supervised_loss(
                preds_all[:, k], y_target_k, y_mask_target_k, _midx_k, log_var=lv_k
            )

        if return_log_var:
            return loss_acc, preds_all, masked_idx, log_var_all
        return loss_acc, preds_all, masked_idx

    # ------------------------------------------------------------------
    # Inference (no grad, no loss)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x:           torch.Tensor,
        x_mask:      torch.Tensor,
        spatial:     torch.Tensor,
        x_hours:     torch.Tensor,
        y_hours:     torch.Tensor,
        delta_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-only forward pass — no masking randomness (eval mode).

        Args:
            (same shapes as forward, but y / y_mask not needed)
            delta_steps: (B,) for single horizon, or call once per horizon.

        Returns:
            preds: (B, N, num_target_vars) predictions for all stations.
        """
        self.eval()
        encoded, _, _, input_ctx = self.encoder(x, x_mask, spatial, x_hours)
        decoder_out = self.decoder(encoded, spatial, y_hours, delta_steps, input_context=input_ctx)
        # When NLL mode is active, decoder returns (mean, log_var) — return mean only.
        if self.use_nll_loss:
            return decoder_out[0]
        return decoder_out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _supervised_loss(
        self,
        preds:      torch.Tensor,                    # (B, N, V)
        y:          torch.Tensor,                    # (B, N, V)
        y_mask:     torch.Tensor,                    # (B, N, V)
        masked_idx: "torch.Tensor | None" = None,
        log_var:    "torch.Tensor | None" = None,    # (B, N, V) — log σ² per var; None → MSE/Huber
    ) -> torch.Tensor:
        """
        Variable-weighted loss: Gaussian NLL when log_var is provided, otherwise
        Huber(δ=1.0) for ALL variables.

        Two modes, selected by whether log_var is passed:

        Huber mode (log_var=None, default):
          Weights (temperature=1.0, pressure=0.5, humidity=0.8, wind_u=1.5, wind_v=1.5):
            • Huber(δ=1.0) applied to every variable in per-station-normalised space.
              Below δ=1σ the loss is L2 (standard MSE gradient); above δ it is L1
              (capped gradient), preventing extreme meteorological events from
              dominating parameter updates.
            • Weights correct for predictability imbalance: pressure is easy to
              predict (0.5×), wind is hard (1.5×), others at intermediate levels.
            • Per-station normalisation already handles raw scale (std≈1), so these
              weights target the gradient imbalance from residual difficulty alone.

        NLL mode (log_var provided, --nll_loss):
          Heteroscedastic Gaussian NLL for all variables:
            NLL_v = 0.5 × (err² × exp(−log_var_v) + log_var_v)
          log_var is clamped to [−10, 10] to prevent σ² from collapsing or exploding.
          Variable weights still apply; Huber is replaced by the Gaussian likelihood.
          This is equivalent to CRPS for Gaussian predictive distributions
          (Gneiting & Raftery, 2007), rewarding calibrated uncertainty alongside accuracy.

        Each variable's loss is normalised by its own present-sensor count so that
        variables with fewer valid sensors are not artificially up-weighted.

        Returns a scalar: weighted sum of per-variable losses / sum of weights.
        """
        if masked_idx is not None:
            B, N_m = masked_idx.shape
            V   = y.shape[-1]
            idx = masked_idx.unsqueeze(-1).expand(B, N_m, V)
            preds  = preds.gather(1, idx)
            y      = y.gather(1, idx)
            y_mask = y_mask.gather(1, idx)
            if log_var is not None:
                log_var = log_var.gather(1, idx)

        sensor_ok = y_mask.bool()                                   # (B, N, V)
        V = preds.shape[-1]
        # NLL mode is self-weighting via σ² — uniform weights avoid double-counting.
        # MSE/Huber mode uses the hand-tuned variable weights.
        if log_var is not None:
            weights = preds.new_ones(V)
        else:
            weights = self.var_weights[:V]                          # (V,)

        loss_per_var = []
        for v in range(V):
            ok  = sensor_ok[..., v]                                 # (B, N)
            if ok.sum() == 0:
                loss_per_var.append(preds.new_zeros(()))
                continue
            err = preds[..., v] - y[..., v]                        # (B, N)

            if log_var is not None:
                # ── NLL mode: heteroscedastic Gaussian (CRPS-equivalent) ──
                # Clamp to [-10, 10]: σ² = e^{-10} ≈ 4.5e-5 (min), e^{10} ≈ 22026 (max)
                lv   = log_var[..., v].clamp(-10.0, 10.0)          # (B, N)
                elem = 0.5 * (err.pow(2) * (-lv).exp() + lv)
            else:
                # ── Huber(δ=1.0): applied to ALL variables ────────────────
                # L2 for |err| ≤ δ, L1 for |err| > δ.
                # In per-station-normalised space δ=1.0 means errors within
                # one standard deviation are penalised quadratically;
                # outliers (extreme events) contribute a capped linear gradient.
                abs_err = err.abs()
                elem = torch.where(
                    abs_err <= self.huber_delta,
                    0.5 * err.pow(2),
                    self.huber_delta * (abs_err - 0.5 * self.huber_delta),
                )

            # Normalise by this variable's own sensor count, not the global total
            var_loss = (elem * ok.float()).sum() / ok.float().sum().clamp(min=1.0)

            # ── Persistence-normalised loss (Fix 2) ───────────────────────
            # Divide by the persistence MSE for this variable so that the loss
            # is expressed as a skill score (1.0 = persistence level).
            # This equalises gradient contributions across variables regardless
            # of how spatially predictable each one is — wind (large persist MSE)
            # is down-weighted, pressure (small persist MSE) is up-weighted.
            # persist_mse is clamped to ≥ 1e-6 to guard against near-zero values.
            if self.use_persist_norm:
                var_loss = var_loss / self.persist_mse[v].clamp(min=1e-6)

            loss_per_var.append(var_loss)

        weighted = (weights * torch.stack(loss_per_var)).sum()
        return weighted / weights.sum()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def set_persist_mse(self, persist_mse: "torch.Tensor") -> None:
        """Set per-variable persistence MSE for loss normalisation (Fix 2).

        Must be called before training when ``use_persist_norm=True``.
        The values are estimated from the validation set by the
        ``_estimate_persist_mse`` helper in main.py.

        Args:
            persist_mse: (num_target_vars,) float tensor — persistence MSE
                         per variable in normalised (z-score) space.
                         Typically in the range [0.05, 0.80] depending on
                         how spatially predictable each variable is.

        Example::

            persist_mse = _estimate_persist_mse(val_loader, num_target_vars=5)
            model.set_persist_mse(persist_mse)
        """
        self.persist_mse.copy_(persist_mse.to(self.persist_mse.device))

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
