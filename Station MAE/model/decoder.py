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

from .embeddings import (
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
    DeltaTimeEmbedding,
    POSITION_FOURIER_DIM,
    STATION_CHAR_DIM,
    TEMPORAL_FOURIER_DIM,
    DELTA_FOURIER_DIM,
    NUM_VARIABLES,
    NUM_TARGET_VARIABLES,
)
from .encoder import TransformerBlock   # reuse the same Pre-LN block


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
    ):
        super().__init__()

        self.d_model         = d_model
        self.num_vars        = num_vars
        self.num_target_vars = num_target_vars

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
        self.pos_emb      = PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim)

        # p2 — WHERE (characteristics): MLP over topographic features
        self.station_emb  = StationEmbedding(d_model=d_model, input_dim=station_char_dim)

        # t  — WHEN: Aurora Fourier temporal encoding for the TARGET timestep
        self.temporal_emb = TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim)

        # Δt — LEAD: Fourier encoding over continuous lead-time hours
        self.delta_emb    = DeltaTimeEmbedding(d_model=d_model, fourier_dim=delta_fourier_dim)

        # --- Post-assembly normalisation for station query tokens ---
        # Applied after summing mask_token + spatial + temporal + delta,
        # matching the encoder's token_norm for consistent scale at decoder input.
        self.query_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # Lightweight self-attention stack (2 layers by default)
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

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
        encoded_vis: torch.Tensor,    # (B, W*N_vis, d_model)  — encoder output
        spatial:     torch.Tensor,    # (N, 15) or (B, N, 15)  — [:2]=pos, [2:]=char
        y_hours:     torch.Tensor,    # (B,)   hours since epoch for target step
        delta_steps: torch.Tensor,    # (B,)   integer forecast lead-time
    ) -> torch.Tensor:
        """
        Predict variables for all N stations at the target time.

        Args:
            encoded_vis:  (B, W*N_vis, d_model)  encoder output (visible tokens)
            spatial:      (N, 15) or (B, N, 15)  static features for ALL N stations;
                          columns 0:2 → easting/northing (PositionalEmbedding p1)
                          columns 2:  → topographic characteristics (StationEmbedding p2)
            y_hours:      (B,)                   target time as hours since epoch
            delta_steps:  (B,)                   forecast horizon in 10-min steps
                                                  (0 = pure reconstruction)

        Returns:
            preds: (B, N, num_target_vars)  raw predictions for every station.
                   num_target_vars = 5 (temperature, pressure, humidity, wind_u, wind_v).
                   Loss is computed externally on masked stations only.
        """
        B = encoded_vis.size(0)

        # --- Resolve spatial shape → (B, N, 15) ---
        if spatial.dim() == 2:
            N         = spatial.size(0)
            spatial_b = spatial.unsqueeze(0).expand(B, -1, -1)   # (B, N, 15)
        else:
            N         = spatial.size(1)
            spatial_b = spatial                                   # (B, N, 15)

        # ------------------------------------------------------------------
        # 1. Build station query tokens for the target timestep
        #    query[b,n] = mask_token
        #               + p1: pos_emb(spatial[n, :2])       — WHERE (position)
        #               + p2: station_emb(spatial[n, 2:])   — WHERE (characteristics)
        #               + t:  temporal_emb(y_hours[b])       — WHEN
        #               + Δt: delta_emb(delta_steps[b])      — LEAD
        # ------------------------------------------------------------------

        # Start with shared learnable mask token: (B, N, d_model)
        queries = self.mask_token.expand(B, N, -1).contiguous()    # (B, N, d_model)

        # p1 — WHERE (position): Fourier encoding of easting/northing
        queries = queries + self.pos_emb(spatial_b[..., :2])       # (B, N, d_model)

        # p2 — WHERE (characteristics): MLP over topographic features
        queries = queries + self.station_emb(spatial_b[..., 2:])   # (B, N, d_model)

        # t — WHEN: Aurora Fourier temporal encoding for target step
        temp_emb = self.temporal_emb(y_hours)                       # (B, d_model)
        queries  = queries + temp_emb.unsqueeze(1)                  # (B, N, d_model)

        # Δt — LEAD: Fourier delta-time embedding
        delt_emb = self.delta_emb(delta_steps)                      # (B, d_model)
        queries  = queries + delt_emb.unsqueeze(1)                  # (B, N, d_model)

        # Normalise query tokens after summation — mirrors encoder token_norm
        queries = self.query_norm(queries)                         # (B, N, d_model)

        # ------------------------------------------------------------------
        # 2. Full sequence: encoder context + station queries
        #    Shape: (B, W*N_vis + N, d_model)
        #    The station queries are appended LAST so we can slice them out
        #    after the transformer blocks.
        # ------------------------------------------------------------------
        full_seq = torch.cat([encoded_vis, queries], dim=1)        # (B, L_ctx + N, d_model)

        # ------------------------------------------------------------------
        # 3. Decoder transformer blocks (self-attention over full sequence)
        # ------------------------------------------------------------------
        h = full_seq
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        # ------------------------------------------------------------------
        # 4. Extract station tokens (last N positions) and predict
        # ------------------------------------------------------------------
        station_tokens = h[:, -N:, :]                              # (B, N, d_model)
        preds          = self.head(station_tokens)                 # (B, N, num_vars)

        return preds
