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
        cross_attention_decoder: bool  = False,
        drop_path_rate:          float = 0.0,
        masked_only_loss:         bool  = False,
        joint_encoder:            bool  = False,
        input_context_cross_attn: bool  = False,
        delta0_weight:            float = 1.0,
        use_nll_loss:             bool  = False,
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

        use_nll_loss=False : default — MSE for non-wind, Huber for wind (backward compat)
        use_nll_loss=True  : Gaussian NLL for all variables; enables calibrated uncertainty

        References: Gneiting & Raftery (2007), Andrychowicz et al. (2023, MetNet-3),
                    Bodnar et al. (2024, Aurora §3)
        """
        super().__init__()

        self.mask_ratio       = mask_ratio
        self.num_vars         = num_vars
        self.delta0_weight    = float(delta0_weight)
        self.num_target_vars  = num_target_vars
        self.masked_only_loss = masked_only_loss
        self.use_nll_loss     = use_nll_loss

        # ── Variable-weighted Huber loss ──────────────────────────────────
        # Weights: [temperature, pressure, humidity, wind_u, wind_v]
        # Temperature gets 2.0× — primary variable of interest; stronger gradient
        # signal pushes the model to sharpen its temperature predictions.
        # Pressure and humidity stay at 1.0 (physically coupled — down-weighting
        # pressure cascades to degraded humidity learning).
        # Wind gets 1.5× weight + Huber loss to reduce mean-regression bias.
        _w = torch.tensor([2.0, 1.0, 1.0, 1.5, 1.5], dtype=torch.float32)
        self.register_buffer("var_weights", _w)
        self.huber_delta   = 1.0          # Huber transition point in normalised space
        self._wind_indices = {3, 4}       # wind_u=3, wind_v=4 in TARGET_VARIABLE_NAMES

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
            drop_path_rate=drop_path_rate,
            joint=joint_encoder,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-delta training forward: encoder once, decoder once for all K.

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
        # k=0 (delta=0) gets weight self.delta0_weight; all forecast horizons
        # (k≥1) get weight 1.0.  Weights are normalised to sum=1 so the loss
        # magnitude stays comparable regardless of the weight choice.
        #
        # Motivation (Challu et al. 2023; v5-small training dynamics):
        #   At delta=0 visible stations can nearly copy their own last input,
        #   contributing near-zero loss that dilutes the forecasting gradient.
        #   Down-weighting this horizon shifts gradient mass toward the K-1
        #   forecast targets without removing the gap-filling signal entirely.
        h_weights = encoded.new_ones(K)
        h_weights[0] = self.delta0_weight
        h_weights = h_weights / h_weights.sum()   # normalise → sums to 1

        _midx    = masked_idx if self.masked_only_loss else None
        loss_acc = encoded.new_zeros(())
        for k in range(K):
            y_target_k      = y[:, k, :, :self.num_target_vars]       # (B, N, num_target_vars)
            y_mask_target_k = y_mask[:, k, :, :self.num_target_vars]  # (B, N, num_target_vars)
            lv_k = log_var_all[:, k] if log_var_all is not None else None  # (B, N, num_target_vars) or None
            loss_acc = loss_acc + h_weights[k] * self._supervised_loss(
                preds_all[:, k], y_target_k, y_mask_target_k, _midx, log_var=lv_k
            )

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
        Huber for wind components and half-MSE for all others.

        Two modes, selected by whether log_var is passed:

        MSE/Huber mode (log_var=None, default):
          Weights (temperature=2.0, pressure=1.0, humidity=1.0, wind_u=1.5, wind_v=1.5):
            • Temperature 2.0× — primary variable; stronger gradient signal.
            • Pressure and humidity equal weight — physically correlated.
            • Wind 1.5× + Huber — reduces systematic underprediction bias.

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
        weights = self.var_weights[:V]                              # (V,)

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
            elif v in self._wind_indices:
                # ── Huber loss: L2 below delta, L1 above ─────────────────
                abs_err = err.abs()
                elem = torch.where(
                    abs_err <= self.huber_delta,
                    0.5 * err.pow(2),
                    self.huber_delta * (abs_err - 0.5 * self.huber_delta)
                )
            else:
                elem = 0.5 * err.pow(2)                            # half-MSE matches Huber scale

            # Normalise by this variable's own sensor count, not the global total
            loss_per_var.append((elem * ok.float()).sum() / ok.float().sum().clamp(min=1.0))

        weighted = (weights * torch.stack(loss_per_var)).sum()
        return weighted / weights.sum()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
