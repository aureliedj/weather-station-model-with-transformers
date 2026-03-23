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
    SpatialEmbedding,
    TemporalEmbedding,
    DeltaTimeEmbedding,
    SPATIAL_INPUT_DIM,
    TEMPORAL_FOURIER_DIM,
    NUM_VARIABLES,
)
from .encoder import TransformerBlock   # reuse the same Pre-LN block


class StationMAEDecoder(nn.Module):
    """
    MAE decoder for weather station reconstruction and forecasting.

    Args:
        d_model:     Model dimension — must match encoder d_model.
        num_heads:   Attention heads (default 4).
        num_layers:  Transformer blocks (default 2; lighter than encoder).
        mlp_ratio:   FFN hidden-dim ratio (default 4.0).
        dropout:     Dropout rate (default 0.1).
        num_vars:    Variables to predict per station (default 6).
        spatial_dim: Static feature dimension (default 18).
        fourier_dim: Fourier dimension for TemporalEmbedding (default 32).
        max_delta:   Maximum forecast lead-time in 10-min steps (default 36 = 6 h).
    """

    def __init__(
        self,
        d_model:     int   = 128,
        num_heads:   int   = 4,
        num_layers:  int   = 2,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
        num_vars:    int   = NUM_VARIABLES,
        spatial_dim: int   = SPATIAL_INPUT_DIM,
        fourier_dim: int   = TEMPORAL_FOURIER_DIM,
        max_delta:   int   = 36,
    ):
        super().__init__()

        self.d_model  = d_model
        self.num_vars = num_vars

        # ------------------------------------------------------------------
        # Learnable mask token — shared starting point for all station queries.
        # Position is then injected via the three positional embeddings below.
        # ------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ------------------------------------------------------------------
        # Positional embeddings used ONLY in the decoder
        # ------------------------------------------------------------------
        # WHERE  — same SpatialEmbedding architecture as encoder
        self.spatial_emb  = SpatialEmbedding(d_model=d_model, input_dim=spatial_dim)

        # WHEN   — Aurora-inspired Fourier temporal encoding for the TARGET timestep
        self.temporal_emb = TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim)

        # LEAD   — learned table: step 0 = reconstruction, step k = k×10-min forecast
        self.delta_emb    = DeltaTimeEmbedding(d_model=d_model, max_steps=max_delta)

        # ------------------------------------------------------------------
        # Lightweight self-attention stack (2 layers by default)
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # Prediction head: d_model → num_vars
        # ------------------------------------------------------------------
        self.head = nn.Linear(d_model, num_vars)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoded_vis: torch.Tensor,    # (B, W*N_vis, d_model)  — encoder output
        spatial:     torch.Tensor,    # (N, 18) or (B, N, 18)  — ALL N stations
        y_hours:     torch.Tensor,    # (B,)   hours since epoch for target step
        delta_steps: torch.Tensor,    # (B,)   integer forecast lead-time
    ) -> torch.Tensor:
        """
        Predict variables for all N stations at the target time.

        Args:
            encoded_vis:  (B, W*N_vis, d_model)  encoder output (visible tokens)
            spatial:      (N, 18) or (B, N, 18)  static features for ALL N stations
            y_hours:      (B,)                   target time as hours since epoch
            delta_steps:  (B,)                   forecast horizon in 10-min steps
                                                  (0 = pure reconstruction)

        Returns:
            preds: (B, N, num_vars)  raw predictions for every station.
                   Loss is computed externally on masked stations only.
        """
        B = encoded_vis.size(0)

        # --- Resolve spatial shape ---
        if spatial.dim() == 2:
            N         = spatial.size(0)
            spatial_b = spatial.unsqueeze(0).expand(B, -1, -1)   # (B, N, 18)
        else:
            N         = spatial.size(1)
            spatial_b = spatial                                   # (B, N, 18)

        # ------------------------------------------------------------------
        # 1. Build station query tokens for the target timestep
        #    query[b, n] = mask_token + spatial[n] + temporal(y_hours[b])
        #                             + delta(delta_steps[b])
        # ------------------------------------------------------------------

        # Start with shared learnable token: expand to (B, N, d_model)
        queries = self.mask_token.expand(B, N, -1).contiguous()   # (B, N, d_model)

        # Add WHERE: spatial embedding per station
        queries = queries + self.spatial_emb(spatial_b)            # (B, N, d_model)

        # Add WHEN: Aurora Fourier temporal encoding for target step
        # y_hours: (B,) → temporal_emb: (B, d_model) → broadcast over N
        temp_emb = self.temporal_emb(y_hours)                      # (B, d_model)
        queries  = queries + temp_emb.unsqueeze(1)                 # (B, N, d_model)

        # Add LEAD: learned delta-time embedding per sample
        # delta_steps: (B,) → delta_emb: (B, d_model) → broadcast over N
        delt_emb = self.delta_emb(delta_steps)                     # (B, d_model)
        queries  = queries + delt_emb.unsqueeze(1)                 # (B, N, d_model)

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
