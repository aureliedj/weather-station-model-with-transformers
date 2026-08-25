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
    STATION_CHAR_DIM,
    POSITION_FOURIER_DIM,
    StepIndexEmbedding,
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
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
    """

    # Every key main.py records in hyper_parameters["cfg"] that changes the
    # ARCHITECTURE, mapped to its constructor argument. Anything not listed here
    # is a data/optimiser setting and does not affect how the model is built.
    #
    # This table exists because main.py and test.py used to build the model from
    # two independently maintained kwargs lists, and they drifted: test.py never
    # read value_embedding, wind_encoder, static_in_token, direct_head or
    # readout, so any v18+ checkpoint silently rebuilt itself with the pre-v18
    # defaults. The structural guard caught it, but only after a four-minute
    # dataset build. One table, two callers, no drift.
    _CFG_TO_ARG = {
        "d_model":            "d_model",
        "enc_heads":          "enc_heads",
        "enc_layers":         "enc_layers",
        "dec_heads":          "dec_heads",
        "dec_layers":         "dec_layers",
        "mlp_ratio":          "mlp_ratio",
        "mask_ratio":         "mask_ratio",
        "window":             "window_size",
        "factorised_encoder": "factorised_encoder",
        "encoder_spatial_attn": "encoder_spatial_attn",
        "temporal_window":    "temporal_window",
        "temporal_patch":     "temporal_patch",
        "value_embedding":    "value_embedding",
        "static_in_token":    "static_in_token",
        "readout":            "readout",
        "residual_head":      "residual_head",
        "cross_attn_decoder": "cross_attention_decoder",
        "station_local_decoder": "station_local_decoder",
        "drop_path_rate":     "drop_path_rate",
        "var_weights":        "var_weights",
    }

    @classmethod
    def from_cfg(cls, cfg: dict, **overrides) -> "StationMAE":
        """
        Build a model from a saved ``hyper_parameters["cfg"]`` dict.

        Defaults are the constructor's own, so a cfg from an older run rebuilds
        the behaviour that run had. ``overrides`` win over cfg — test.py uses
        that for ``dropout=0.0`` and for CLI flags that override the checkpoint.

        Two keys are derived rather than copied:
          * ``num_horizons`` from ``max_delta // delta_grid_stride + 1``, the
            same arithmetic main.py uses; it must match the trained
            ``direct_proj`` width or the head reshapes into the wrong horizons.
        """
        kw = {arg: cfg[key] for key, arg in cls._CFG_TO_ARG.items() if key in cfg}

        if "max_delta" in cfg:
            stride = int(cfg.get("delta_grid_stride", 3) or 3)
            kw["num_horizons"] = int(cfg["max_delta"]) // stride + 1

        kw.update(overrides)
        return cls(**kw)

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
        value_embedding:         str   = "linear",
        static_in_token:         bool  = False,
        readout:                 str   = "last",         # "last" | "mean"
        num_horizons:            int   = 13,
        cross_attention_decoder: bool  = False,
        station_local_decoder:   bool  = False,   # decoder attends within one station only
        drop_path_rate:          float = 0.0,
        residual_head:            bool  = False,        # ŷ = y(t0) + f(·)
        var_weights:              "list | None" = None,
        use_nll_loss:             bool  = False,
        window_size:              int   = 72,
    ):
        """

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

        The buffer is saved in checkpoints and restored on resume.

        use_persist_norm=False : default — raw MSE scale (backward compat)
        use_persist_norm=True  : skill-score scale, auto-balances gradient contributions

        References: Rasp & Lerch (2018), Demaeyer et al. (2023, EUPPBench)
        """
        super().__init__()

        self.mask_ratio        = mask_ratio
        self.num_vars          = num_vars
        self.num_target_vars   = num_target_vars
        self.use_nll_loss      = use_nll_loss

        # Persistence MSE normaliser — one scalar per target variable.
        # Retained at 1.0: present in every saved state_dict, so keeping the
        # buffer means existing checkpoints load without unexpected keys.
        # is called.  Saved/restored automatically via Lightning checkpoints.
        self.register_buffer(
            "persist_mse",
            torch.ones(num_target_vars, dtype=torch.float32),
        )

        # ── Variable-weighted Huber loss ──────────────────────────────────
        # Weights: [temperature, pressure, humidity, wind_u, wind_v]
        #
        # Per-variable loss weights — UNIFORM by default, matching the LSTM
        # baseline (run_lstm_cloud.sh: VAR_WEIGHTS="1.0 1.0 1.0 1.0 1.0").
        #
        # This used to default to [1.0, 1.0, 0.7, 0.5, 0.5], down-weighting
        # humidity and wind as "noisy". That made the per-variable comparison
        # against the LSTM invalid: the transformer got 0.7x and 0.5x the
        # gradient on exactly the two variables it was losing on. Runs trained
        # before 2026-08-02 used the old weights; none of them survive in
        # checkpoints/ (see EXPERIMENTS.md).
        #
        # Weights affect the LOSS only, never the predictions, so evaluating an
        # old checkpoint under the new default changes nothing that is dumped.
        _w = (torch.tensor(var_weights, dtype=torch.float32) if var_weights
              else torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32))
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

        # Shared position / topography / absolute-time embeddings — same
        # reasoning as shared_step_emb, extended to the rest of the "WHERE"
        # and "WHEN" signals a query and its matching key both carry. Without
        # this, the encoder's pos_emb(station n) and the decoder's
        # pos_emb(station n) are two separately-initialised MLPs over the
        # same input; cross-attention then has to LEARN an approximate
        # alignment between them rather than starting from an exact one.
        # Deliberately NOT extended to var_proj — the encoder embeds an
        # observed VALUE that has no decoder-side counterpart to share.
        shared_pos_emb      = PositionalEmbedding(d_model=d_model, fourier_dim=POSITION_FOURIER_DIM,
                                                   dropout=dropout)
        shared_station_emb  = StationEmbedding(d_model=d_model, input_dim=STATION_CHAR_DIM,
                                                dropout=dropout)
        shared_temporal_emb = TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim,
                                                 dropout=dropout)

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
            value_embedding=value_embedding,
            static_in_token=static_in_token,
            drop_path_rate=drop_path_rate,
            step_emb=shared_step_emb,
            pos_emb=shared_pos_emb,
            station_emb=shared_station_emb,
            temporal_emb=shared_temporal_emb,
        )
        self.residual_head    = bool(residual_head)

        # `readout` selected how a direct (decoder-free) head pooled the
        # encoder output — "last" temporal slot or "mean" over slots. That head
        # was removed (it was disabled in every run); the argument is validated
        # and stored but no longer selects anything. See archive/README.md.
        assert readout in ("last", "mean"), readout
        self.readout      = readout
        self.num_horizons = int(num_horizons)
        self.station_local_decoder = bool(station_local_decoder)
        if self.station_local_decoder:
            # Same requirement as direct_head, same reason: the station-local
            # decoder attends to each station's OWN encoder tokens, so a station
            # dropped by masking would have nothing to attend to.
            assert mask_ratio == 0.0, (
                "station_local_decoder folds the station axis into the batch, so "
                "every station must contribute encoder tokens; got "
                f"mask_ratio={mask_ratio}. Use --mask_ratio 0, or drop "
                "--station_local_decoder.")
        self.num_target_vars_ = num_target_vars

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
            station_local=station_local_decoder,
            drop_path_rate=drop_path_rate,
            predict_uncertainty=use_nll_loss,
            window_size=window_size,
            step_emb=shared_step_emb,
            pos_emb=shared_pos_emb,
            station_emb=shared_station_emb,
            temporal_emb=shared_temporal_emb,
        )

    # ------------------------------------------------------------------
    # Persistence-residual base
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------



    def _station_masked(self, masked_idx, B, N, device):
        """(B, N) bool — True where the encoder could not see the station."""
        flag = torch.zeros(B, N, dtype=torch.bool, device=device)
        if masked_idx is not None and masked_idx.numel() > 0:
            flag.scatter_(1, masked_idx, True)
        return flag

    def _persistence_base(
        self,
        x:          torch.Tensor,            # (B, W, N, V)  normalised obs
        x_mask:     torch.Tensor,            # (B, W, N, V)  sensor availability
        masked_idx: "torch.Tensor | None",   # (B, N_masked) long
    ) -> "torch.Tensor | None":
        """
        Last-observation base for the residual head:  ŷ = base + f(·).

        base[b, n, v] = x[b, W-1, n, v]  if station n is VISIBLE and sensor
                        present at the last step; 0 (the per-station mean in
                        normalised space) otherwise.

        Masked stations are zeroed EXPLICITLY: their last observation is
        hidden information, and feeding it through the base would bypass the
        encoder mask — the exact leak class the old input_context had. With a
        zero base, the decoder learns the full value for hidden stations (the
        gap-filling regime) and only the deviation-from-persistence for
        visible ones (the forecast regime).
        Returns None when the residual head is disabled.
        """
        if not self.residual_head:
            return None
        Vt   = self.num_target_vars_
        base = x[:, -1, :, :Vt] * x_mask[:, -1, :, :Vt]     # (B, N, Vt)
        if masked_idx is not None and masked_idx.numel() > 0:
            B, N = base.shape[:2]
            vis = torch.ones(B, N, 1, dtype=base.dtype, device=base.device)
            vis.scatter_(1, masked_idx.unsqueeze(-1), 0.0)
            base = base * vis
        return base

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

        Loss semantics: δ=0 is scored on the MASKED
        stations only (inpainting) and δ>0 on all stations. The residual head
        only re-parameterises WHAT the decoder outputs (deviation from the
        last observation for visible stations), never WHERE the loss applies.

        Returns:
            loss:           scalar — MSE on all N stations (present sensors only)
            preds:          (B, N, num_target_vars)
            masked_indices: (B, N_masked)   — encoder mask, used by caller for analysis
        """
        encoded, masked_idx, visible_idx = self.encoder(x, x_mask, spatial, x_hours)
        # station_local decoder: every station must still be present. A
        # divisibility check inside the decoder is NOT sufficient — with
        # N_vis = N/2 the token count can still divide by N and the reshape
        # then silently uses the wrong temporal length (caught by
        # tests/test_station_local_decoder.py). masked_idx is the direct signal.
        if self.station_local_decoder and masked_idx is not None \
                and masked_idx.numel() > 0:
            raise RuntimeError(
                f"station_local_decoder needs every station present, but "
                f"{masked_idx.shape[-1]} of {x.shape[2]} stations were masked "
                f"(encoder.mask_ratio={self.encoder.mask_ratio}). Set "
                f"mask_ratio=0 for this model.")

        decoder_out = self.decoder(
            encoded, spatial, y_hours, delta_steps,
            station_masked=self._station_masked(
                masked_idx, x.shape[0], x.shape[2], x.device),
            )
        if self.use_nll_loss:
            preds, log_var = decoder_out
        else:
            preds, log_var = decoder_out, None
        base = self._persistence_base(x, x_mask, masked_idx)
        if base is not None:
            preds = preds + base                       # (B, N, Vt)

        y_target      = y[:, :, :self.num_target_vars]
        y_mask_target = y_mask[:, :, :self.num_target_vars]
        # Optionally restrict loss to masked stations (inpainting regime)
        _midx = None          # masked_only_loss was never enabled
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
        encoded, masked_idx, visible_idx = self.encoder(x, x_mask, spatial, x_hours)
        # encoded: (B, T*N_vis, d_model)   (T = patched sequence length)

        # ── Predictions: Delta-query decoder ─────────────────────────────
        # (the direct-head branch was removed: direct_head=False in every run)
        # station_local decoder: every station must still be present. A
        # divisibility check inside the decoder is NOT sufficient — with
        # N_vis = N/2 the token count can still divide by N and the reshape
        # then silently uses the wrong temporal length (caught by
        # tests/test_station_local_decoder.py). masked_idx is the direct signal.
        if self.station_local_decoder and masked_idx is not None \
                and masked_idx.numel() > 0:
            raise RuntimeError(
                f"station_local_decoder needs every station present, but "
                f"{masked_idx.shape[-1]} of {x.shape[2]} stations were masked "
                f"(encoder.mask_ratio={self.encoder.mask_ratio}). Set "
                f"mask_ratio=0 for this model.")

        decoder_out = self.decoder(
            encoded, spatial, y_hours, delta_steps,
            station_masked=self._station_masked(
                masked_idx, x.shape[0], x.shape[2], x.device),
            )
        if self.use_nll_loss:
            preds_all, log_var_all = decoder_out
        else:
            preds_all, log_var_all = decoder_out, None
        # preds_all: (B, K, N, num_target_vars)
        # log_var_all: (B, K, N, num_target_vars)  or  None

        # Residual head: every horizon predicts a deviation from the last
        # observation (zero base for masked stations — no leakage).
        base = self._persistence_base(x, x_mask, masked_idx)
        if base is not None:
            preds_all = preds_all + base.unsqueeze(1)   # (B, K, N, Vt)

        # ── Horizon-weighted loss ─────────────────────────────────────────
        # Uniform weights across all K horizons (normalised to sum=1).
        # delta0_weight is applied to k=0 when delta=0 is the first grid entry,
        # though with masked-only supervision at k=0 the gradient is already
        # naturally smaller (~50% of stations), so delta0_weight=1.0 is preferred.
        # Built WITHOUT a host sync. `delta_steps[0, 0].item()` forced a
        # GPU->CPU transfer on every training step and, under --compile, a
        # graph break at the same point every iteration. torch.where keeps the
        # decision on-device; the numerics are identical.
        # Uniform across horizons: delta0_weight was 1.0 in every run, so the
        # torch.where collapsed to new_ones(K) / K.
        h_weights = encoded.new_ones(K) / K

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

            # k=0 and delta=0: restrict to masked stations — but ONLY if any
            # station is actually masked.
            #
            # At mask_ratio=0 masked_idx has shape (B, 0): not None, so the
            # gather below yields empty tensors, sensor_ok.sum() == 0, and the
            # delta=0 term contributes EXACTLY ZERO gradient. The model would
            # then never be trained to emit delta=0 at all, while the LSTM
            # baseline is supervised there for every station — which silently
            # invalidates any no-masking comparison.
            #
            # With nothing masked there is no shortcut to protect against
            # (the "trivial copy" concern only applies when some stations are
            # visible and others hidden), so supervise all stations, exactly as
            # the LSTM does.
            # EVERY horizon is supervised on EVERY station with a present
            # sensor, delta=0 included.
            #
            # v15-v20 restricted delta=0 to masked stations, on the reasoning
            # that a visible station could trivially copy its own last input.
            # The effect was that a visible station's delta=0 output received
            # ZERO gradient and was an untrained free parameter. On v20's test
            # dump those rows scored 0.389 against 0.263 for the masked ones,
            # which read as a residual-head defect; it was an unsupervised
            # output behaving like one, and it also made sanity/ctx_ratio
            # meaningless (an unsupervised quantity divided by a supervised one).
            #
            # Copying IS the correct answer at delta=0 for a station the encoder
            # can see, and supervising it is free — the target is one of the
            # inputs. Uniform supervision is also what the LSTM baseline gets,
            # so the comparison is like-for-like.
            _has_masked = masked_idx is not None and masked_idx.shape[1] > 0
            _midx_k = None    # masked_only_loss was never enabled

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
        encoded, masked_idx, visible_idx = self.encoder(x, x_mask, spatial, x_hours)
        # station_local decoder: every station must still be present. A
        # divisibility check inside the decoder is NOT sufficient — with
        # N_vis = N/2 the token count can still divide by N and the reshape
        # then silently uses the wrong temporal length (caught by
        # tests/test_station_local_decoder.py). masked_idx is the direct signal.
        if self.station_local_decoder and masked_idx is not None \
                and masked_idx.numel() > 0:
            raise RuntimeError(
                f"station_local_decoder needs every station present, but "
                f"{masked_idx.shape[-1]} of {x.shape[2]} stations were masked "
                f"(encoder.mask_ratio={self.encoder.mask_ratio}). Set "
                f"mask_ratio=0 for this model.")

        decoder_out = self.decoder(
            encoded, spatial, y_hours, delta_steps,
            station_masked=self._station_masked(
                masked_idx, x.shape[0], x.shape[2], x.device),
            )
        # When NLL mode is active, decoder returns (mean, log_var) — return mean only.
        preds = decoder_out[0] if self.use_nll_loss else decoder_out
        base  = self._persistence_base(x, x_mask, masked_idx)
        if base is not None:
            preds = preds + (base.unsqueeze(1) if preds.dim() == 4 else base)
        return preds

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
          Weights (default [1.0, 1.0, 0.7, 0.5, 0.5] = temp, pressure,
          humidity, wind_u, wind_v; override with --var_weights):
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
            # NOTE: there used to be an `if ok.sum() == 0: continue` guard here.
            # It forced a GPU->CPU sync per variable per horizon — K*V = 65 per
            # training step — and broke the compiled graph at the same point
            # every iteration. It was also redundant: the denominator below is
            # already clamped to >= 1, so an all-absent variable yields
            # 0 / 1 = 0, exactly what the guard returned. Safe because absent
            # sensors carry 0.0 (dataset.py nan_to_num), never NaN, so the
            # masked-out elements cannot poison the sum.
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

            loss_per_var.append(var_loss)

        weighted = (weights * torch.stack(loss_per_var)).sum()
        return weighted / weights.sum()


    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
