"""
model/encoder.py

Station-MAE encoder.

    tokens (B, W, N, d)  ->  mask whole stations  ->  merge P steps per token
        ->  L factorised blocks (temporal attention, spatial attention, FFN)
        ->  (B, W/P * N_vis, d)

Masking is per station: a masked station is hidden for the whole input
window and contributes no token, so the decoder can only reconstruct it from
the other stations.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _cp_checkpoint

from .embeddings import (
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
    StepIndexEmbedding,
    VariableProjection,
    POSITION_FOURIER_DIM,
    STATION_CHAR_DIM,
    TEMPORAL_FOURIER_DIM,
    NUM_VARIABLES,
)


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop the whole residual branch per sample with probability ``drop_prob``."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Draw in float32: in bfloat16 the rounding of rand + keep_prob near 1.0
        # lowers the realised drop rate below the nominal one.
        random_tensor = torch.rand(shape, dtype=torch.float32, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob) / keep_prob
        return x * random_tensor.to(x.dtype)


def _ffn(d_model: int, mlp_ratio: float, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, int(d_model * mlp_ratio)),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(int(d_model * mlp_ratio), d_model),
        nn.Dropout(dropout),
    )


# ---------------------------------------------------------------------------
# Factorised (axial) transformer block
# ---------------------------------------------------------------------------

class FactorisedTransformerBlock(nn.Module):
    """
    Pre-LN block on a (B, T, N, d) token grid with three residual sub-layers:

      1. temporal attention: each station attends over its T time steps
         (reshape to (B*N, T, d));
      2. spatial attention: all N stations attend to each other at each time
         step (reshape to (B*T, N, d)); omitted when ``spatial_attn=False``,
         which makes the block station-independent;
      3. shared FFN.

    Cost O(N*T^2 + T*N^2) instead of O((T*N)^2) for joint attention.
    """

    def __init__(
        self,
        d_model:      int,
        num_heads:    int,
        mlp_ratio:    float = 4.0,
        dropout:      float = 0.1,
        spatial_attn: bool  = True,
        drop_path:    float = 0.0,
    ):
        super().__init__()
        self.spatial_attn = spatial_attn

        self.norm_t = nn.LayerNorm(d_model)
        self.attn_t = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        if spatial_attn:
            self.norm_s = nn.LayerNorm(d_model)
            self.attn_s = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                batch_first=True)

        self.norm_ff   = nn.LayerNorm(d_model)
        self.ffn       = _ffn(d_model, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, d) -> (B, T, N, d)."""
        B, T, N, D = x.shape

        # 1. temporal attention over T, per station
        xt   = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        xt_n = self.norm_t(xt)
        at, _ = self.attn_t(xt_n, xt_n, xt_n, need_weights=False)
        xt   = xt + self.drop_path(at)
        x    = xt.reshape(B, N, T, D).permute(0, 2, 1, 3)

        # 2. spatial attention over N, per time step
        if self.spatial_attn:
            xs   = x.reshape(B * T, N, D)
            xs_n = self.norm_s(xs)
            as_, _ = self.attn_s(xs_n, xs_n, xs_n, need_weights=False)
            xs   = xs + self.drop_path(as_)
            x    = xs.reshape(B, T, N, D)

        # 3. FFN
        x = x + self.drop_path(self.ffn(self.norm_ff(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class StationMAEEncoder(nn.Module):
    """
    Args:
        d_model, num_heads, num_layers, mlp_ratio, dropout: transformer size.
        mask_ratio:       fraction of stations hidden per sample (0 .. 1).
        num_vars:         input variables per station (6).
        use_checkpoint:   gradient checkpointing on every block.
        spatial_attn:     False removes the spatial sub-layer from every block
                          (station-independent encoder).
        temporal_patch:   P consecutive steps are concatenated into one token
                          after masking (W must be divisible by P).
        drop_path_rate:   maximum stochastic-depth rate (linear over depth).
        step_emb, pos_emb, station_emb, temporal_emb:
                          embedding modules shared with the decoder; built
                          locally when None.
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
        use_checkpoint:       bool  = False,
        spatial_attn:         bool  = True,
        temporal_patch:       int   = 1,
        drop_path_rate:       float = 0.0,
        step_emb:             "nn.Module | None" = None,
        pos_emb:              "nn.Module | None" = None,
        station_emb:          "nn.Module | None" = None,
        temporal_emb:         "nn.Module | None" = None,
    ):
        super().__init__()
        self.d_model        = d_model
        self.mask_ratio     = mask_ratio
        self.temporal_patch = int(temporal_patch)
        self.use_checkpoint = use_checkpoint

        # Token components. pos/station/temporal/step embeddings are shared
        # with the decoder so that a station's key and its query carry the
        # same positional vector.
        self.var_proj     = VariableProjection(num_vars=num_vars, d_model=d_model)
        self.pos_emb      = pos_emb if pos_emb is not None else \
            PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim, dropout=dropout)
        self.station_emb  = station_emb if station_emb is not None else \
            StationEmbedding(d_model=d_model, input_dim=station_char_dim, dropout=dropout)
        self.temporal_emb = temporal_emb if temporal_emb is not None else \
            TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim, dropout=dropout)
        self.step_emb     = step_emb if step_emb is not None else \
            StepIndexEmbedding(d_model=d_model, dropout=dropout)
        self.token_norm   = nn.LayerNorm(d_model)

        dp_rates = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            FactorisedTransformerBlock(d_model, num_heads, mlp_ratio, dropout,
                                       spatial_attn=spatial_attn, drop_path=dp_rates[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Temporal patch merging: concatenate P step embeddings, LayerNorm, Linear.
        if self.temporal_patch > 1:
            self.patch_norm  = nn.LayerNorm(self.temporal_patch * d_model)
            self.patch_merge = nn.Linear(self.temporal_patch * d_model, d_model)
        else:
            self.patch_norm = self.patch_merge = None

        self._fixed_eval_mask = None

    # ------------------------------------------------------------------

    def _patchify(self, t: torch.Tensor) -> torch.Tensor:
        """(B, W, N, D) -> (B, W/P, N, D)."""
        B, W, N, D = t.shape
        P = self.temporal_patch
        if P <= 1:
            return t
        assert W % P == 0, f"temporal_patch={P} must divide the window W={W}"
        t = t.reshape(B, W // P, P, N, D).permute(0, 1, 3, 2, 4).reshape(B, W // P, N, P * D)
        return self.patch_merge(self.patch_norm(t))

    def _build_tokens(self, x, x_mask, spatial, x_hours) -> torch.Tensor:
        """
        token[b, w, n] = var_proj(x[b,w,n], x_mask[b,w,n])
                       + pos_emb(spatial[n, :2]) + station_emb(spatial[n, 2:])
                       + temporal_emb(x_hours[b, w]) + step_emb(w)
        Returns (B, W, N, d_model).
        """
        B, W, N, V = x.shape
        var_tokens = self.var_proj(x.view(B * W, N, V), x_mask.view(B * W, N, V))
        var_tokens = var_tokens.view(B, W, N, self.d_model)

        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)                          # (1, N, 15)
        pos_e     = self.pos_emb(spatial[..., :2]).unsqueeze(1)     # (1/B, 1, N, d)
        station_e = self.station_emb(spatial[..., 2:]).unsqueeze(1) # (1/B, 1, N, d)
        temp_e    = self.temporal_emb(x_hours).unsqueeze(2)         # (B, W, 1, d)
        step_e    = self.step_emb(torch.arange(W, device=x.device)).view(1, W, 1, self.d_model)

        tokens = var_tokens + pos_e + station_e + temp_e + step_e
        return self.token_norm(tokens)

    def set_fixed_eval_mask(self, masked_indices, visible_indices) -> None:
        """
        Replay one fixed station mask for every batch (evaluation only).
        Pass ``None, None`` to return to random per-sample masks.
        """
        if masked_indices is None and visible_indices is None:
            self._fixed_eval_mask = None
        else:
            assert masked_indices is not None and visible_indices is not None
            self._fixed_eval_mask = (masked_indices, visible_indices)

    def _mask_stations(self, tokens: torch.Tensor):
        """
        Hide int(N * mask_ratio) stations per sample, independently per sample.

        Returns:
            visible_tokens  (B, W, N_vis, d)
            masked_indices  (B, N_masked)  sorted
            visible_indices (B, N_vis)     sorted
        """
        B, W, N, D = tokens.shape

        if self._fixed_eval_mask is not None:
            masked_indices  = self._fixed_eval_mask[0].to(tokens.device).unsqueeze(0).expand(B, -1)
            visible_indices = self._fixed_eval_mask[1].to(tokens.device).unsqueeze(0).expand(B, -1)
        else:
            num_masked  = int(N * self.mask_ratio)
            shuffle_idx = torch.argsort(torch.rand(B, N, device=tokens.device), dim=1)
            # Sorting keeps the station order canonical: at mask_ratio 0 the
            # visible sequence is exactly arange(N).
            masked_indices,  _ = torch.sort(shuffle_idx[:, :num_masked], dim=1)
            visible_indices, _ = torch.sort(shuffle_idx[:, num_masked:], dim=1)

        num_visible = visible_indices.shape[1]
        vis_idx_exp = visible_indices.unsqueeze(1).unsqueeze(-1).expand(B, W, num_visible, D)
        visible_tokens = tokens.gather(2, vis_idx_exp)
        return visible_tokens, masked_indices, visible_indices

    def forward(self, x, x_mask, spatial, x_hours):
        """
        Args:
            x:        (B, W, N, V)  normalised observations
            x_mask:   (B, W, N, V)  sensor availability
            spatial:  (N, 15) or (B, N, 15) normalised static station features
            x_hours:  (B, W)  hours since epoch per input step
        Returns:
            encoded         (B, T * N_vis, d_model), T = W / temporal_patch
            masked_indices  (B, N_masked)
            visible_indices (B, N_vis)
        """
        tokens = self._build_tokens(x, x_mask, spatial, x_hours)
        visible_tokens, masked_idx, visible_idx = self._mask_stations(tokens)
        h = self._patchify(visible_tokens)
        B, T, N_vis, D = h.shape

        for block in self.blocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                h = _cp_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        h = self.norm(h.reshape(B, T * N_vis, D))
        return h, masked_idx, visible_idx
