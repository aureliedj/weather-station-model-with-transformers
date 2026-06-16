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
     vector by summing three positional signals:

         query[n] = mask_token                      (learnable, shared)
                  + spatial_emb(spatial[n])         — WHERE  (static topo/location)
                  + temporal_emb(y_hours)            — WHEN   (Aurora Fourier time)
                  + delta_emb(delta_steps)           — LEAD   (forecast horizon)

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
    ) -> torch.Tensor:
        """
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
        input_context_cross_attn: bool = False,
    ):
        super().__init__()

        self.d_model           = d_model
        self.num_vars          = num_vars
        self.num_target_vars   = num_target_vars
        self.use_checkpoint    = use_checkpoint
        self.use_cross_attention = cross_attention

        # ------------------------------------------------------------------
        # Learnable mask token — shared starting point for all station queries.
        # Position is injected via the four embeddings below.
        # ------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ------------------------------------------------------------------
        # Positional / contextual embeddings for query token construction
        # Token = mask_token + p1 + p2 + t + delta
        # ------------------------------------------------------------------
        # p1 — WHERE (position): Fourier encoding of easting/northing
        self.pos_emb      = PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim,
                                                dropout=dropout)

        # p2 — WHERE (characteristics): MLP over topographic features
        self.station_emb  = StationEmbedding(d_model=d_model, input_dim=station_char_dim,
                                             dropout=dropout)

        # t  — WHEN: Aurora Fourier temporal encoding for the TARGET timestep
        self.temporal_emb = TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim,
                                              dropout=dropout)

        # Δt — LEAD: Fourier encoding over continuous lead-time hours
        self.delta_emb    = DeltaTimeEmbedding(d_model=d_model, fourier_dim=delta_fourier_dim,
                                               dropout=dropout)

        # s  — STEP: Integer step-index embedding (same Fourier basis as encoder).
        # The decoder receives delta_steps as integer 10-min step counts
        # (e.g. 0, 3, 6, 9, … for 30-min spacing).  This embedding maps those
        # step counts to d_model using the same log-spaced sinusoidal basis as
        # the encoder's StepIndexEmbedding, so the model can directly align
        # "input token at step 3" with "decoder query at delta_step 3".
        # Complements delta_emb (which encodes continuous hours) with an
        # explicit discrete-step signal consistent with the encoder context.
        self.step_emb = StepIndexEmbedding(d_model=d_model, dropout=dropout)

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
        # Optional: final cross-attention to last-timestep input tokens.
        #
        # After the main decoder blocks have integrated encoder context, this
        # extra CrossAttentionBlock lets query tokens directly read the most
        # recent (t=W-1) observation of ALL N stations — including the masked
        # ones whose last known value is still visible here.
        #
        # This anchors short-horizon predictions to the last observed state,
        # which is exactly what persistence uses.  The model can now learn
        # "copy + small correction" rather than having to reconstruct the
        # recent state from the compressed encoder output alone.
        # ------------------------------------------------------------------
        self.use_input_context = input_context_cross_attn
        if input_context_cross_attn:
            self.input_cross_attn = CrossAttentionBlock(
                d_model, num_heads, mlp_ratio, dropout, drop_path=0.0
            )

        # ------------------------------------------------------------------
        # Prediction head: d_model → num_target_vars
        # Precipitation is excluded from the output — it is used as input
        # context only (zero-inflated distribution makes MSE a poor fit).
        # ------------------------------------------------------------------
        self.head = nn.Linear(d_model, num_target_vars)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoded_vis:   torch.Tensor,           # (B, W*N_vis, d_model) — encoder output
        spatial:       torch.Tensor,           # (N, 15) or (B, N, 15)
        y_hours:       torch.Tensor,           # (B,) or (B, K)
        delta_steps:   torch.Tensor,           # (B,) or (B, K)
        input_context: "torch.Tensor | None" = None,  # (B, N, d_model) last-timestep tokens
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
        spatial_q = self.mask_token.expand(B, N, -1).contiguous()  # (B, N, d_model)
        spatial_q = spatial_q + self.pos_emb(spatial_b[..., :2])   # (B, N, d_model)
        spatial_q = spatial_q + self.station_emb(spatial_b[..., 2:])  # (B, N, d_model)

        if not is_multi:
            # ── Single-delta path ─────────────────────────────────────────
            temp_emb = self.temporal_emb(y_hours)                   # (B, d_model)
            delt_emb = self.delta_emb(delta_steps)                  # (B, d_model)
            step_emb = self.step_emb(delta_steps)                   # (B, d_model)

            queries = spatial_q \
                    + temp_emb.unsqueeze(1) \
                    + delt_emb.unsqueeze(1) \
                    + step_emb.unsqueeze(1)                         # (B, N, d_model)
            queries = self.query_norm(queries)                      # (B, N, d_model)

            if self.use_cross_attention:
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

            # ── Optional: cross-attend to last-timestep input tokens ────
            if self.use_input_context and input_context is not None:
                h = self.input_cross_attn(h, input_context)         # (B, N, d_model)

            h = self.norm(h)
            return self.head(h)                                     # (B, N, num_target_vars)

        else:
            # ── Multi-delta path ──────────────────────────────────────────
            temp_emb = self.temporal_emb(y_hours)                   # (B, K, d_model)
            delt_emb = self.delta_emb(delta_steps)                  # (B, K, d_model)
            step_emb = self.step_emb(delta_steps)                   # (B, K, d_model)

            spatial_exp = spatial_q.unsqueeze(2).expand(B, N, K, -1)   # (B, N, K, d_model)
            temp_exp    = temp_emb.unsqueeze(1).expand(B, N, K, -1)    # (B, N, K, d_model)
            delt_exp    = delt_emb.unsqueeze(1).expand(B, N, K, -1)    # (B, N, K, d_model)
            step_exp    = step_emb.unsqueeze(1).expand(B, N, K, -1)    # (B, N, K, d_model)

            queries_NK  = spatial_exp + temp_exp + delt_exp + step_exp  # (B, N, K, d_model)
            queries_NK  = self.query_norm(queries_NK)
            queries_seq = queries_NK.reshape(B, N * K, self.d_model)   # (B, N*K, d_model)

            if self.use_cross_attention:
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

            # ── Optional: cross-attend to last-timestep input tokens ────
            # For multi-delta, expand input_context to match N*K queries.
            if self.use_input_context and input_context is not None:
                ctx_exp = input_context.unsqueeze(2).expand(B, N, K, -1) \
                                       .reshape(B, N * K, self.d_model)   # (B, N*K, D)
                h = self.input_cross_attn(h, ctx_exp)

            h = self.norm(h)                                        # (B, N*K, d_model)

            preds = self.head(h)                                    # (B, N*K, num_target_vars)
            preds = preds.reshape(B, N, K, self.num_target_vars)
            preds = preds.permute(0, 2, 1, 3).contiguous()         # (B, K, N, num_target_vars)
            return preds
