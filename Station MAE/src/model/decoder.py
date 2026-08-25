"""
model/decoder.py

MAE decoder for Station-MAE.

Design
------
The decoder answers a single question per forward pass:
    "Given what the encoder saw in the input window, what are all N stations
     doing at the TARGET time t + Δt?"

It does so in four steps:

  1. Station queries  —  For each of the N stations build a target-time query
     vector by summing five positional signals:

         query[n,k] = mask_token                           (learnable, shared)
                    + spatial_emb(spatial[n])              — WHERE  (static topo/location)
                    + temporal_emb(y_hours[k])             — WHEN   (Aurora Fourier absolute time)
                    + delta_emb(delta_steps[k])            — LEAD   (forecast horizon in hours)
                    + step_emb(W-1 + delta_steps[k])       — LEAD   (position in unified step-count
                                                                      timeline shared with encoder)

     The step_emb places each decoder query in the SAME integer coordinate system
     as the encoder: encoder input tokens use indices 0..W-1, decoder query tokens
     use indices W-1+Δ (e.g. 74, 77, …, 107 for Δ=3,6,…,36 and W=72).
     This is semantically clean — no aliasing between encoder and decoder indices —
     and lets the model learn "I am 3 steps after the last encoder step" rather
     than just "my lead time is 0.5 h".

  2. Full sequence  —  Concatenate the W·N_vis encoder tokens (context) with
     the N station queries (targets):

         full_seq = [encoded_vis ‖ station_queries]     shape (B, W·N_vis + N, d_model)

  3. Self-attention  —  A lightweight stack of Transformer blocks attends over
     the full sequence so every station query can see all encoder context.

  4. Prediction head  —  The last N tokens are projected to (N, V) via a
     linear layer.  Loss is computed externally (in StationMAE) on masked
     stations only.

Why concatenate rather than cross-attend?
    Simple self-attention over the full sequence lets encoder tokens also
    attend to each other in the decoder, which helps when reconstructing
    correlated variables.  For longer windows the cross-attention variant
    saves memory; that extension is straightforward.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _cp_checkpoint

from .embeddings import (
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
    DeltaTimeEmbedding,
    StepIndexEmbedding,
    POSITION_FOURIER_DIM,
    STATION_CHAR_DIM,
    TEMPORAL_FOURIER_DIM,
    DELTA_FOURIER_DIM,
    NUM_VARIABLES,
    NUM_TARGET_VARIABLES,
)
from .encoder import TransformerBlock, DropPath   # reuse Pre-LN block and DropPath


# ---------------------------------------------------------------------------
# Cross-attention decoder block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Decoder block where query tokens attend to encoder context via cross-attention.

    Structure (Pre-LN throughout):
      1. Self-attention    — queries attend to each other (captures inter-station
                             structure within the query set).
      2. Cross-attention   — queries (Q) attend to encoder output (K, V); lets
                             every query token directly read from the full encoder
                             context without concatenating the two sequences.
      3. FFN

    Why cross-attention instead of concatenated self-attention?
        The original decoder concatenates [encoder_tokens ‖ query_tokens] and runs
        joint self-attention.  This forces every encoder token to also attend to
        every query token — wasting capacity and adding N (or N·K) extra tokens to
        an already large encoder sequence.  Cross-attention is cleaner:

          Old (self-attn):  sequence length = W·N_vis + N·K   (~29,400 at W=288)
          New (cross-attn): query length    = N·K             (~600)
                            context length  = W·N_vis          (~28,800, read-only)

        Memory cost of the cross-attention matrix:
            O(N·K × W·N_vis) with standard attention
            O(N·K)           with Flash Attention (need_weights=False)

    Args:
        d_model:   Model dimension — must match encoder d_model.
        num_heads: Attention heads (shared across self and cross sub-layers).
        mlp_ratio: FFN hidden-dim ratio (default 4.0).
        dropout:   Dropout rate (default 0.1).
        drop_path: Stochastic depth drop probability (default 0.0 = disabled).
                   Applied to the residuals of self-attn, cross-attn, and FFN.
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # ── Self-attention on queries ─────────────────────────────────────
        self.norm_sa    = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # ── Cross-attention: Q from queries, K/V from encoder context ─────
        self.norm_q     = nn.LayerNorm(d_model)   # normalise query side
        self.norm_kv    = nn.LayerNorm(d_model)   # normalise encoder context
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # ── FFN ───────────────────────────────────────────────────────────
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

        # ── Stochastic depth ──────────────────────────────────────────────
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        q:  torch.Tensor,   # (B, L_q,  d_model) — decoder query tokens
        kv: torch.Tensor,   # (B, L_kv, d_model) — encoder context (key/value)
        kv_padding_mask: "torch.Tensor | None" = None,  # (B, L_kv) bool; True = ignore key
    ) -> torch.Tensor:
        """
        Args:
            kv_padding_mask: optional (B, L_kv) bool mask, True where the K/V
                token must NOT be attended (softmax logit = −inf).
                v15: unused by this project's decoder (the input_context
                pathway that needed it was removed); kept because
                CrossAttentionBlock is generic and a zero/degenerate key would
                otherwise still receive softmax mass — softmax has no native
                notion of an "empty" key (standard padding-mask practice,
                cf. Transformer/BERT).

        Returns:
            (B, L_q, d_model)
        """
        # 1. Self-attention among queries
        q_n  = self.norm_sa(q)
        sa, _ = self.self_attn(q_n, q_n, q_n, need_weights=False)
        q    = q + self.drop_path(sa)

        # 2. Cross-attention: queries read from encoder context
        kv_n  = self.norm_kv(kv)          # normalise once, reuse for K and V
        ca, _ = self.cross_attn(
            self.norm_q(q),               # Q: normalised queries
            kv_n,                         # K: normalised encoder context
            kv_n,                         # V: same
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        q = q + self.drop_path(ca)

        # 3. FFN
        q = q + self.drop_path(self.ffn(self.norm_ff(q)))
        return q


class StationMAEDecoder(nn.Module):
    """
    MAE decoder for weather station reconstruction and forecasting.

    Args:
        d_model:              Model dimension — must match encoder d_model.
        num_heads:            Attention heads (default 4).
        num_layers:           Transformer blocks (default 2; lighter than encoder).
        mlp_ratio:            FFN hidden-dim ratio (default 4.0).
        dropout:              Dropout rate (default 0.1).
        num_vars:             Input variables per station (default 6).
        num_target_vars:      Variables to predict per station (default 5, excludes
                              precipitation which is used as input only).
        station_char_dim:     Dimension of station characteristic features p2 (default 13).
        fourier_dim:          Fourier dimension for TemporalEmbedding (default 32).
        delta_fourier_dim:    Fourier dimension for DeltaTimeEmbedding (default 16).
        position_fourier_dim: Fourier features per coordinate for PositionalEmbedding (default 16).
        use_checkpoint:       If True, use gradient checkpointing on each decoder block.
        cross_attention:      If True, use CrossAttentionBlock (query tokens cross-attend
                              to encoder context) instead of concatenated self-attention.
                              Reduces decoder sequence length from W·N_vis+N·K to just N·K,
                              which is much cheaper when the encoder context is long.
    """

    def __init__(
        self,
        d_model:              int   = 128,
        num_heads:            int   = 4,
        num_layers:           int   = 2,
        mlp_ratio:            float = 4.0,
        dropout:              float = 0.1,
        num_vars:             int   = NUM_VARIABLES,
        num_target_vars:      int   = NUM_TARGET_VARIABLES,
        station_char_dim:     int   = STATION_CHAR_DIM,
        fourier_dim:          int   = TEMPORAL_FOURIER_DIM,
        delta_fourier_dim:    int   = DELTA_FOURIER_DIM,
        position_fourier_dim: int   = POSITION_FOURIER_DIM,
        use_checkpoint:          bool  = False,
        cross_attention:         bool  = False,
        drop_path_rate:          float = 0.0,
        predict_uncertainty:      bool = False,
        window_size:              int  = 72,
        step_emb:                 "nn.Module | None" = None,
        pos_emb:                  "nn.Module | None" = None,
        station_emb:              "nn.Module | None" = None,
        temporal_emb:             "nn.Module | None" = None,
        station_local:            bool = False,
    ):
        super().__init__()

        self.d_model             = d_model
        self.num_target_vars     = num_target_vars
        self.use_checkpoint      = use_checkpoint
        self.use_cross_attention = cross_attention
        self.predict_uncertainty = predict_uncertainty
        # station_local=True folds the station axis into the batch dimension, so
        # a station's queries attend only to one another and cross-attend only
        # to THAT station's encoder tokens. Combined with an encoder built with
        # spatial_attn=False this yields a genuinely station-independent model,
        # while keeping the Delta-query decoder intact. Requires mask_ratio 0
        # (a masked station contributes no encoder tokens to attend to).
        self.station_local       = bool(station_local)
        # W: input window length.  Decoder step indices = W-1 + delta_steps,
        # continuing the encoder's 0..W-1 timeline into future positions.
        self.window_size         = window_size

        # ------------------------------------------------------------------
        # Learnable mask token — shared starting point for all station queries.
        # Position is injected via the four embeddings below.
        # ------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ------------------------------------------------------------------
        # STATION-STATE embedding — "was this station visible to the encoder?"
        #
        # Two learned vectors: index 0 = visible, index 1 = masked. Added to the
        # query for station n according to whether n was hidden in THIS forward
        # pass.
        #
        # Why this is needed. With the residual head the decoder predicts
        #     y_hat = base + f(.)
        # where base = the station's last observation if VISIBLE, and 0 if
        # MASKED (a masked station's last value is hidden information and must
        # not leak). So f has to emit a small deviation-from-persistence in one
        # regime and a full absolute value in the other — and the query
        # previously carried no signal distinguishing them:
        #     mask_token + position + topography + time + delta + step
        # is identical for a visible and a hidden station. One head, bimodal
        # target, unobservable condition. That is why --residual_head degraded
        # the v15 sanity run and was switched off.
        #
        # This is NOT leakage. Which stations are offline is known at
        # deployment time; only their VALUES must stay hidden. The embedding
        # carries one bit per station — availability — not the observation.
        self.station_state = nn.Parameter(torch.zeros(2, d_model))
        nn.init.trunc_normal_(self.station_state, std=0.02)

        # ------------------------------------------------------------------
        # Positional / contextual embeddings for query token construction
        # Token = mask_token + p1 + p2 + t + delta
        # ------------------------------------------------------------------
        # p1/p2/t — shared with the encoder (see mae.py) rather than built as
        # separate copies. A station's position, topography and a given
        # absolute time are the same fact whether the encoder is embedding a
        # key for that station or the decoder is embedding a query for it —
        # two independently-constructed modules would only share their fixed
        # Fourier frequencies, not their trained MLP weights, so the query's
        # positional fingerprint and the matching key's would drift apart
        # during training instead of being provably identical. Same rationale
        # as step_emb below. Falls back to its own instance when used
        # standalone (e.g. tests, notebooks). NOT applied to the observation
        # value embedding — the decoder has no observed value to share.
        #
        # p1 — WHERE (position): Fourier encoding of easting/northing
        self.pos_emb      = pos_emb if pos_emb is not None else \
            PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim,
                                dropout=dropout)

        # p2 — WHERE (characteristics): MLP over topographic features
        self.station_emb  = station_emb if station_emb is not None else \
            StationEmbedding(d_model=d_model, input_dim=station_char_dim,
                             dropout=dropout)

        # t  — WHEN: Aurora Fourier temporal encoding for the TARGET timestep
        self.temporal_emb = temporal_emb if temporal_emb is not None else \
            TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim,
                                              dropout=dropout)

        # Δt — LEAD: Fourier encoding over continuous lead-time hours
        self.delta_emb    = DeltaTimeEmbedding(d_model=d_model, fourier_dim=delta_fourier_dim,
                                               dropout=dropout)

        # s  — STEP: integer position in the unified encoder+decoder timeline.
        #
        # The encoder assigns step indices 0..W-1 to its input tokens.
        # The decoder's query tokens at lead Δ are placed at index W-1+Δ,
        # continuing the same integer coordinate system past the window edge:
        #
        #   Encoder  :  0 ──── 1 ──── … ──── W-1          (input steps)
        #   Decoder  :                              W-1+3  W-1+6  …  W-1+36
        #              (e.g. W=72 → encoder 0..71, decoder 74, 77, …, 107)
        #
        # This guarantees decoder indices are strictly outside the encoder range
        # (no aliasing: "step 6 in the encoder" ≠ "step 6 in the decoder"),
        # and gives the model a common positional reference that complements
        # DeltaTimeEmbedding's continuous-hours signal with a discrete step count
        # calibrated to the window scale.
        #
        # This module instance is shared with the encoder (passed in from
        # StationMAE, see mae.py) rather than built as a separate copy. The
        # decoder's fixed_grid horizons only touch a sparse subset of indices
        # past W-1 (e.g. stride-3 → 74, 77, ..., 107 for W=72), overlapping the
        # encoder's dense 0..W-1 range at exactly one point (W-1, delta=0).
        # Two independently-constructed StepIndexEmbedding instances would only
        # share their fixed Fourier frequencies, not their trained MLP weights,
        # so "encoder step k" and "decoder step k" would silently diverge during
        # training. Sharing the instance makes them provably identical instead.
        # Falls back to its own instance when used standalone.
        self.step_emb = step_emb if step_emb is not None else \
            StepIndexEmbedding(d_model=d_model, dropout=dropout)

        # --- Post-assembly normalisation for station query tokens ---
        # Applied after summing mask_token + spatial + temporal + delta,
        # matching the encoder's token_norm for consistent scale at decoder input.
        self.query_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # Decoder attention stack
        # cross_attention=True  → CrossAttentionBlock  (queries cross-attend encoder)
        # cross_attention=False → TransformerBlock     (concatenated self-attention)
        #
        # Stochastic depth rates increase linearly from 0 → drop_path_rate
        # across decoder layers, matching the encoder convention.
        # ------------------------------------------------------------------
        dp_rates = [
            drop_path_rate * i / max(num_layers - 1, 1)
            for i in range(num_layers)
        ]
        if cross_attention:
            self.blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, num_heads, mlp_ratio, dropout,
                                    drop_path=dp_rates[i])
                for i in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, mlp_ratio, dropout,
                                 drop_path=dp_rates[i])
                for i in range(num_layers)
            ])
        self.norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # v15: the input-context cross-attention block has been REMOVED.
        # Its job — anchoring short leads to the raw last observation — is
        # now done structurally: the telescopic patch schedule keeps the last
        # hour at native resolution in the encoder sequence, and the
        # persistence-residual head (mae.py) provides the "copy + small
        # correction" prior directly. Checkpoints containing
        # decoder.input_cross_attn.* weights predate v15 and must be
        # evaluated with the code that trained them.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Prediction head: d_model → num_target_vars
        # Precipitation is excluded from the output — it is used as input
        # context only (zero-inflated distribution makes MSE a poor fit).
        # ------------------------------------------------------------------
        self.head = nn.Linear(d_model, num_target_vars)

        # ------------------------------------------------------------------
        # Optional log-variance head (heteroscedastic NLL / CRPS mode).
        #
        # When predict_uncertainty=True, a second linear layer produces
        # log σ²_v per variable, enabling the Gaussian heteroscedastic NLL:
        #
        #     NLL = 0.5 × (err² / σ² + log σ²)
        #         = 0.5 × (err² × exp(−log_var) + log_var)
        #
        # Initialised with zero weights and bias so that at training start
        # σ² = exp(log_var) = 1, making the NLL identical in scale to MSE.
        # The model then learns to widen uncertainty for difficult samples
        # (large residuals, long horizons, masked stations far from neighbours).
        #
        # Only instantiated when predict_uncertainty=True — no overhead when
        # using the default MSE/Huber loss.
        # ------------------------------------------------------------------
        if predict_uncertainty:
            self.log_var_head = nn.Linear(d_model, num_target_vars)
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.zeros_(self.log_var_head.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoded_vis:   torch.Tensor,           # (B, W*N_vis, d_model) — encoder output
        spatial:       torch.Tensor,           # (N, 15) or (B, N, 15)
        y_hours:       torch.Tensor,           # (B,) or (B, K)
        delta_steps:   torch.Tensor,           # (B,) or (B, K)
        station_masked: "torch.Tensor | None" = None,   # (B, N) bool: hidden from the encoder
    ) -> torch.Tensor:
        """
        Predict variables for all N stations at one or K target times.

        Supports two calling conventions depending on the shape of ``y_hours``
        and ``delta_steps``:

        Single-delta  (K=1):
            y_hours:     (B,)
            delta_steps: (B,)
            Returns:     (B, N, num_target_vars)

        Multi-delta  (K>1):
            y_hours:     (B, K)
            delta_steps: (B, K)
            Returns:     (B, K, N, num_target_vars)

            All K lead-times are processed in a **single** transformer pass.
            N×K query tokens are built — one per (station, lead-time) pair —
            and concatenated with the encoder context before self-attention.
            This avoids K separate decoder forward passes while giving every
            lead-time query its own distinct temporal and delta embedding.

        Args:
            encoded_vis:  (B, W*N_vis, d_model)  encoder output (visible tokens)
            spatial:      (N, 15) or (B, N, 15)  static features for ALL N stations;
                          columns 0:2 → easting/northing (PositionalEmbedding p1)
                          columns 2:  → topographic characteristics (StationEmbedding p2)
            y_hours:      (B,) or (B, K)          target time(s) as hours since epoch
            delta_steps:  (B,) or (B, K)          forecast horizon(s) in 10-min steps

        Returns:
            preds: (B, N, num_target_vars)        — single-delta
                or (B, K, N, num_target_vars)     — multi-delta
        """
        B = encoded_vis.size(0)

        # --- Resolve spatial shape → (B, N, 15) ---
        if spatial.dim() == 2:
            N         = spatial.size(0)
            spatial_b = spatial.unsqueeze(0).expand(B, -1, -1)   # (B, N, 15)
        else:
            N         = spatial.size(1)
            spatial_b = spatial                                   # (B, N, 15)

        # Detect single- vs multi-delta from y_hours dimensionality
        is_multi = (y_hours.dim() == 2)   # True → (B, K);  False → (B,)
        K        = y_hours.shape[1] if is_multi else 1

        # ------------------------------------------------------------------
        # 1. Spatial query base (shared across all K lead-times)
        #    spatial_q[b,n] = mask_token
        #                   + p1: pos_emb(spatial[n, :2])     — WHERE (position)
        #                   + p2: station_emb(spatial[n, 2:]) — WHERE (topo)
        # ------------------------------------------------------------------
        # Every query starts from the shared mask_token. (The v23 "anchor"
        # variant, which started VISIBLE queries from their own encoder token,
        # was never enabled in any run and has been removed; station_state
        # below still carries the visible/masked distinction.)
        spatial_q = self.mask_token.expand(B, N, -1).contiguous()   # (B, N, d_model)
        spatial_q = spatial_q + self.pos_emb(spatial_b[..., :2])   # (B, N, d_model)
        spatial_q = spatial_q + self.station_emb(spatial_b[..., 2:])  # (B, N, d_model)

        # Station state: index 1 where the encoder could not see this station.
        # Defaults to "all visible" when the caller passes nothing, which is
        # what evaluation at mask_ratio 0 wants.
        if station_masked is None:
            spatial_q = spatial_q + self.station_state[0]
        else:
            spatial_q = spatial_q + self.station_state[station_masked.long()]

        if not is_multi:
            # ── Single-delta path ─────────────────────────────────────────
            temp_emb = self.temporal_emb(y_hours)                   # (B, d_model)
            delt_emb = self.delta_emb(delta_steps)                  # (B, d_model)
            # Step index in the unified encoder+decoder timeline: W-1 + Δ
            # e.g. delta_steps=3 → step_idx=74 for W=72
            step_idx = (self.window_size - 1) + delta_steps         # (B,) long
            step_emb = self.step_emb(step_idx)                      # (B, d_model)

            queries = spatial_q \
                    + temp_emb.unsqueeze(1) \
                    + delt_emb.unsqueeze(1) \
                    + step_emb.unsqueeze(1)                         # (B, N, d_model)
            queries = self.query_norm(queries)                      # (B, N, d_model)

            if self.use_cross_attention and self.station_local:
                # Same fold as the multi-delta path, with K = 1: each station
                # becomes its own attention problem. Without this branch the
                # single-delta path would still mix stations while the
                # multi-delta path did not.
                # NOTE: divisibility alone is NOT a sufficient check. With
                # N_vis = N/2 the token count T*N_vis can still divide by N,
                # and the reshape then silently uses a wrong temporal length
                # (T/2) while mixing stations. StationMAE therefore rejects any
                # masked station BEFORE calling the decoder, using masked_idx.
                # This check remains only as a last-resort shape guard for
                # callers that use the decoder directly.
                _WN = encoded_vis.shape[1]
                if _WN % N != 0:
                    raise RuntimeError(
                        f"station_local decoder needs every station present: "
                        f"encoder returned {_WN} tokens which is not divisible "
                        f"by N={N}. Train and evaluate with --mask_ratio 0.")
                _W = _WN // N
                kv_local = (encoded_vis
                            .view(B, _W, N, self.d_model)
                            .permute(0, 2, 1, 3)
                            .reshape(B * N, _W, self.d_model))
                h = queries.reshape(B * N, 1, self.d_model)
                for block in self.blocks:
                    if self.use_checkpoint and torch.is_grad_enabled():
                        h = _cp_checkpoint(block, h, kv_local, use_reentrant=False)
                    else:
                        h = block(h, kv_local)
                h = h.reshape(B, N, self.d_model)

            elif self.use_cross_attention:
                h = queries
                for block in self.blocks:
                    if self.use_checkpoint and torch.is_grad_enabled():
                        h = _cp_checkpoint(block, h, encoded_vis, use_reentrant=False)
                    else:
                        h = block(h, encoded_vis)
            else:
                full_seq = torch.cat([encoded_vis, queries], dim=1)
                h = full_seq
                for block in self.blocks:
                    if self.use_checkpoint and torch.is_grad_enabled():
                        h = _cp_checkpoint(block, h, use_reentrant=False)
                    else:
                        h = block(h)
                h = h[:, -N:, :]                                    # (B, N, d_model)

            h = self.norm(h)
            mean = self.head(h)                                     # (B, N, num_target_vars)
            if self.predict_uncertainty:
                return mean, self.log_var_head(h)                   # (mean, log_var)
            return mean

        else:
            # ── Multi-delta path ──────────────────────────────────────────
            temp_emb = self.temporal_emb(y_hours)                   # (B, K, d_model)
            delt_emb = self.delta_emb(delta_steps)                  # (B, K, d_model)
            # Step index in the unified encoder+decoder timeline: W-1 + Δ
            # delta_steps: (B, K) long  →  step_idx: (B, K)
            step_idx = (self.window_size - 1) + delta_steps         # (B, K) long
            step_emb = self.step_emb(step_idx)                      # (B, K, d_model)

            spatial_exp = spatial_q.unsqueeze(2).expand(B, N, K, -1)   # (B, N, K, d_model)
            temp_exp    = temp_emb.unsqueeze(1).expand(B, N, K, -1)    # (B, N, K, d_model)
            delt_exp    = delt_emb.unsqueeze(1).expand(B, N, K, -1)    # (B, N, K, d_model)
            step_exp    = step_emb.unsqueeze(1).expand(B, N, K, -1)    # (B, N, K, d_model)

            queries_NK  = spatial_exp + temp_exp + delt_exp + step_exp  # (B, N, K, d_model)
            queries_NK  = self.query_norm(queries_NK)
            queries_seq = queries_NK.reshape(B, N * K, self.d_model)   # (B, N*K, d_model)

            if self.use_cross_attention and self.station_local:
                # ── Station-independent decoding ────────────────────────────
                # Fold the station axis into the batch so each station is a
                # separate attention problem: its K queries attend only to one
                # another, and cross-attend only to its OWN W encoder tokens.
                #
                # Layouts (verified against encoder.py and the head below):
                #   encoded_vis : (B, W*N, d)  W-major, station fastest
                #   queries_NK  : (B, N, K, d) station-major, lead fastest
                #
                # Requires every station present: a masked station has no
                # encoder tokens, so there would be nothing to attend to.
                # NOTE: divisibility alone is NOT a sufficient check. With
                # N_vis = N/2 the token count T*N_vis can still divide by N,
                # and the reshape then silently uses a wrong temporal length
                # (T/2) while mixing stations. StationMAE therefore rejects any
                # masked station BEFORE calling the decoder, using masked_idx.
                # This check remains only as a last-resort shape guard for
                # callers that use the decoder directly.
                _WN = encoded_vis.shape[1]
                if _WN % N != 0:
                    raise RuntimeError(
                        f"station_local decoder needs every station present: "
                        f"encoder returned {_WN} tokens which is not divisible "
                        f"by N={N}. Train and evaluate with --mask_ratio 0.")
                _W = _WN // N
                kv_local = (encoded_vis
                            .view(B, _W, N, self.d_model)   # (B, W, N, d)
                            .permute(0, 2, 1, 3)            # (B, N, W, d)
                            .reshape(B * N, _W, self.d_model))
                h = queries_NK.reshape(B * N, K, self.d_model)
                for block in self.blocks:
                    if self.use_checkpoint and torch.is_grad_enabled():
                        h = _cp_checkpoint(block, h, kv_local, use_reentrant=False)
                    else:
                        h = block(h, kv_local)
                # Back to (B, N*K, d) — station-major, lead fastest, exactly the
                # layout the head and its reshape below already expect.
                h = h.reshape(B, N * K, self.d_model)

            elif self.use_cross_attention:
                h = queries_seq
                for block in self.blocks:
                    if self.use_checkpoint and torch.is_grad_enabled():
                        h = _cp_checkpoint(block, h, encoded_vis, use_reentrant=False)
                    else:
                        h = block(h, encoded_vis)
            else:
                full_seq = torch.cat([encoded_vis, queries_seq], dim=1)
                h = full_seq
                for block in self.blocks:
                    if self.use_checkpoint and torch.is_grad_enabled():
                        h = _cp_checkpoint(block, h, use_reentrant=False)
                    else:
                        h = block(h)
                h = h[:, -N * K:, :]                               # (B, N*K, d_model)

            h = self.norm(h)                                       # (B, N*K, d_model)

            mean = self.head(h)                                    # (B, N*K, num_target_vars)
            mean = mean.reshape(B, N, K, self.num_target_vars)
            mean = mean.permute(0, 2, 1, 3).contiguous()          # (B, K, N, num_target_vars)
            if self.predict_uncertainty:
                log_var = self.log_var_head(h)                     # (B, N*K, num_target_vars)
                log_var = log_var.reshape(B, N, K, self.num_target_vars)
                log_var = log_var.permute(0, 2, 1, 3).contiguous()  # (B, K, N, num_target_vars)
                return mean, log_var
            return mean
