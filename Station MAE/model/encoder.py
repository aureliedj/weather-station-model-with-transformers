"""
model/encoder.py

Transformer encoder for Station-MAE.

Input:
    A multi-step window of station observations, with spatial and temporal
    embeddings already summed into token representations.

Design:
    - Tokens are shaped (B, W, N, d_model) on entry
    - Flattened to (B, W*N, d_model) for self-attention
    - Standard Transformer encoder blocks (MSA + FFN + LayerNorm)
    - A random subset of tokens is masked before encoding (MAE-style)
    - Only visible tokens are processed by the encoder
    - Returns encoded visible tokens + indices needed by the decoder

Masking strategy:
    Masking is applied per station across all timesteps — if station s is
    masked, it is masked for the entire window. This is more meaningful than
    random token masking because:
      1. It simulates realistic missing station scenarios (gap-filling task)
      2. It prevents the encoder from trivially reconstructing a station
         from its own recent history
"""

import torch
import torch.nn as nn
from .embeddings import (
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
    VariableProjection,
    POSITION_FOURIER_DIM,
    STATION_CHAR_DIM,
    TEMPORAL_FOURIER_DIM,
    NUM_VARIABLES,
)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer block: LayerNorm → MSA → residual,
    LayerNorm → FFN → residual.

    Pre-LN (normalisation before attention) is more stable to train than
    the original post-LN formulation, especially at larger depths.

    Args:
        d_model:     Model dimension.
        num_heads:   Number of attention heads.
        mlp_ratio:   FFN hidden dim = d_model * mlp_ratio (default 4).
        dropout:     Dropout rate applied in attention and FFN (default 0.1).
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class StationMAEEncoder(nn.Module):
    """
    MAE encoder for multi-step weather station windows.

    Builds token representations by summing:
        variable_projection(x, mask)   — what was measured
        spatial_embedding(spatial)     — where the station is
        temporal_embedding(t)          — when the measurement was taken

    Then masks a fraction of stations and runs the visible tokens through
    a stack of Transformer blocks.

    Args:
        d_model:              Model dimension (default 128).
        num_heads:            Attention heads (default 4).
        num_layers:           Number of Transformer blocks (default 4).
        mlp_ratio:            FFN expansion ratio (default 4.0).
        dropout:              Dropout rate (default 0.1).
        mask_ratio:           Fraction of stations to mask (default 0.5).
        num_vars:             Number of meteorological variables (default 6).
        station_char_dim:     Dimension of station characteristic features p2 (default 13).
        fourier_dim:          Fourier dimension for TemporalEmbedding (default 32).
        position_fourier_dim: Fourier features per coordinate for PositionalEmbedding (default 16).
    """

    def __init__(
        self,
        d_model:              int   = 128,
        num_heads:            int   = 4,
        num_layers:           int   = 4,
        mlp_ratio:            float = 4.0,
        dropout:              float = 0.1,
        mask_ratio:           float = 0.5,
        num_vars:             int   = NUM_VARIABLES,
        station_char_dim:     int   = STATION_CHAR_DIM,
        fourier_dim:          int   = TEMPORAL_FOURIER_DIM,
        position_fourier_dim: int   = POSITION_FOURIER_DIM,
    ):
        super().__init__()

        self.d_model    = d_model
        self.mask_ratio = mask_ratio

        # --- Embedding modules (four components: p1, p2, v, t) ---
        self.var_proj     = VariableProjection(num_vars=num_vars, d_model=d_model)
        self.pos_emb      = PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim)
        self.station_emb  = StationEmbedding(d_model=d_model, input_dim=station_char_dim)
        self.temporal_emb = TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim)

        # --- Post-assembly normalisation ---
        # Applied after summing var_proj + spatial_emb + temporal_emb to keep
        # the three independently-initialised components on a common scale
        # before they enter the first transformer block.
        self.token_norm = nn.LayerNorm(d_model)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def _build_tokens(
        self,
        x:       torch.Tensor,   # (B, W, N, V)
        x_mask:  torch.Tensor,   # (B, W, N, V)
        spatial: torch.Tensor,   # (N, 15)  or  (B, N, 15)
        x_hours: torch.Tensor,   # (B, W)   hours-since-epoch per input step
    ) -> torch.Tensor:
        """
        Build full token representations of shape (B, W, N, d_model).

        token[b, w, n] = var_proj(x[b,w,n], x_mask[b,w,n])          v — what
                       + pos_emb(spatial[n, :2])                      p1 — where (position)
                       + station_emb(spatial[n, 2:])                  p2 — where (characteristics)
                       + temporal_emb(x_hours[b,w])                   t  — when
        """
        B, W, N, V = x.shape

        # --- v: Variable projection ---
        x_flat      = x.view(B * W, N, V)
        mask_flat   = x_mask.view(B * W, N, V)
        var_tokens  = self.var_proj(x_flat, mask_flat)             # (B*W, N, d_model)
        var_tokens  = var_tokens.view(B, W, N, self.d_model)       # (B, W, N, d_model)

        # --- p1 + p2: split spatial tensor, embed each independently ---
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)                          # (1, N, 15)
        # p1 — Fourier positional encoding over easting/northing
        pos_e     = self.pos_emb(spatial[..., :2])                  # (1/B, N, d_model)
        pos_e     = pos_e.unsqueeze(1)                              # (1/B, 1, N, d_model)
        # p2 — MLP over topographic characteristics
        station_e = self.station_emb(spatial[..., 2:])              # (1/B, N, d_model)
        station_e = station_e.unsqueeze(1)                          # (1/B, 1, N, d_model)

        # --- t: Temporal embedding (Aurora Fourier): (B, W) → (B, W, d_model) ---
        temp_emb = self.temporal_emb(x_hours)                       # (B, W, d_model)
        temp_emb = temp_emb.unsqueeze(2)                            # (B, W, 1, d_model)

        # Sum four embeddings — all broadcast cleanly over (B, W, N, d_model)
        tokens = var_tokens + pos_e + station_e + temp_emb         # (B, W, N, d_model)

        # Normalise after summation: prevents any single component from
        # dominating the scale seen by the first transformer block.
        tokens = self.token_norm(tokens)                        # (B, W, N, d_model)
        return tokens

    def _mask_stations(
        self,
        tokens: torch.Tensor,   # (B, W, N, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask a fraction of stations uniformly across all timesteps.

        The same stations are masked for the entire window — masking is
        per-station, not per-token.

        Args:
            tokens: (B, W, N, d_model)

        Returns:
            visible_tokens  : (B, W * N_vis, d_model)  — flattened visible tokens
            masked_indices  : (B, N_masked)             — masked station indices
            visible_indices : (B, N_vis)                — visible station indices
        """
        B, W, N, D = tokens.shape
        num_masked  = int(N * self.mask_ratio)
        num_visible = N - num_masked

        # Sample a different random mask per batch item
        noise           = torch.rand(B, N, device=tokens.device)
        shuffle_idx     = torch.argsort(noise, dim=1)              # (B, N)
        visible_indices = shuffle_idx[:, num_masked:]              # (B, N_vis)
        masked_indices  = shuffle_idx[:, :num_masked]              # (B, N_masked)

        # Gather visible tokens across station dimension
        # visible_indices: (B, N_vis) → expand to (B, W, N_vis, D)
        vis_idx_exp = visible_indices \
            .unsqueeze(1) \
            .unsqueeze(-1) \
            .expand(B, W, num_visible, D)                          # (B, W, N_vis, D)

        visible_tokens = tokens.gather(2, vis_idx_exp)             # (B, W, N_vis, D)

        # Flatten W and N_vis into a single sequence dimension
        visible_tokens = visible_tokens.reshape(B, W * num_visible, D)

        return visible_tokens, masked_indices, visible_indices

    def forward(
        self,
        x:       torch.Tensor,   # (B, W, N, V)
        x_mask:  torch.Tensor,   # (B, W, N, V)
        spatial: torch.Tensor,   # (N, 15) or (B, N, 15)  — [:2]=pos, [2:]=characteristics
        x_hours: torch.Tensor,   # (B, W)  hours-since-epoch per input timestep
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode visible station tokens.

        Args:
            x:        (B, W, N, V)  normalised observations
            x_mask:   (B, W, N, V)  sensor availability mask
            spatial:  (N, 15)       normalised static station features;
                                    columns 0:2 → easting/northing (PositionalEmbedding p1)
                                    columns 2:  → topographic characteristics (StationEmbedding p2)
            x_hours:  (B, W)        hours since epoch for each input timestep
                                    (feeds TemporalEmbedding)

        Returns:
            encoded         : (B, W * N_vis, d_model)  encoded visible tokens
            masked_indices  : (B, N_masked)             which stations were masked
            visible_indices : (B, N_vis)                which stations were visible
        """
        # 1. Build full token representations
        tokens = self._build_tokens(x, x_mask, spatial, x_hours)  # (B, W, N, d_model)

        # 2. Mask stations
        visible_tokens, masked_idx, visible_idx = self._mask_stations(tokens)
        # visible_tokens: (B, W*N_vis, d_model)

        # 3. Transformer blocks over visible tokens only
        h = visible_tokens
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)                                           # (B, W*N_vis, d_model)

        return h, masked_idx, visible_idx
