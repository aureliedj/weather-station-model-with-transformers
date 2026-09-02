"""
model/decoder.py

Lead-time-conditioned cross-attention decoder.

For every station n and lead time delta_k a query token is built,

    q[n, k] = mask_token
            + pos_emb(spatial[n, :2]) + station_emb(spatial[n, 2:])   where
            + station_state[hidden(n)]                                 visible / masked
            + temporal_emb(y_hours[k])                                 when (absolute)
            + delta_emb(delta_k)                                       lead time
            + step_emb(W - 1 + delta_k)                                step index

and the N*K queries pass through a small stack of blocks that self-attend
among the queries and cross-attend to the encoder output. A linear head maps
each query to the 5 target variables (plus log sigma^2 when
``predict_uncertainty=True``).

With ``station_local=True`` the station axis is folded into the batch, so
each station's queries only see that station's own encoder tokens. Together
with an encoder without spatial attention this gives a model with no
cross-station pathway.
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
from .encoder import DropPath, _ffn


class CrossAttentionBlock(nn.Module):
    """Pre-LN block: self-attention over queries, cross-attention to the encoder, FFN."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm_sa    = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_q     = nn.LayerNorm(d_model)
        self.norm_kv    = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff    = nn.LayerNorm(d_model)
        self.ffn        = _ffn(d_model, mlp_ratio, dropout)
        self.drop_path  = DropPath(drop_path)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: (B, L_q, d) queries, kv: (B, L_kv, d) encoder output -> (B, L_q, d)."""
        q_n   = self.norm_sa(q)
        sa, _ = self.self_attn(q_n, q_n, q_n, need_weights=False)
        q     = q + self.drop_path(sa)

        kv_n  = self.norm_kv(kv)
        ca, _ = self.cross_attn(self.norm_q(q), kv_n, kv_n, need_weights=False)
        q     = q + self.drop_path(ca)

        q = q + self.drop_path(self.ffn(self.norm_ff(q)))
        return q


class StationMAEDecoder(nn.Module):
    """
    Args:
        d_model, num_heads, num_layers, mlp_ratio, dropout: transformer size.
        num_target_vars:      predicted variables per station (5).
        use_checkpoint:       gradient checkpointing on every block.
        drop_path_rate:       maximum stochastic-depth rate (linear over depth).
        predict_uncertainty:  add a second head predicting log sigma^2.
        window_size:          W; decoder step indices are W - 1 + delta.
        station_local:        fold the station axis into the batch (see module doc).
        step_emb, pos_emb, station_emb, temporal_emb:
                              embedding modules shared with the encoder.
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
        use_checkpoint:       bool  = False,
        drop_path_rate:       float = 0.0,
        predict_uncertainty:  bool  = False,
        window_size:          int   = 72,
        step_emb:             "nn.Module | None" = None,
        pos_emb:              "nn.Module | None" = None,
        station_emb:          "nn.Module | None" = None,
        temporal_emb:         "nn.Module | None" = None,
        station_local:        bool  = False,
    ):
        super().__init__()
        self.d_model             = d_model
        self.num_target_vars     = num_target_vars
        self.use_checkpoint      = use_checkpoint
        self.predict_uncertainty = predict_uncertainty
        self.station_local       = bool(station_local)
        self.window_size         = window_size

        # Shared starting point of every query.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Two learned vectors: index 0 = station visible to the encoder,
        # index 1 = station masked. The decoder needs this bit because with
        # the residual head it predicts a deviation from the last observation
        # for visible stations and a full value for masked ones.
        self.station_state = nn.Parameter(torch.zeros(2, d_model))
        nn.init.trunc_normal_(self.station_state, std=0.02)

        self.pos_emb      = pos_emb if pos_emb is not None else \
            PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim, dropout=dropout)
        self.station_emb  = station_emb if station_emb is not None else \
            StationEmbedding(d_model=d_model, input_dim=station_char_dim, dropout=dropout)
        self.temporal_emb = temporal_emb if temporal_emb is not None else \
            TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim, dropout=dropout)
        self.delta_emb    = DeltaTimeEmbedding(d_model=d_model, fourier_dim=delta_fourier_dim,
                                               dropout=dropout)
        self.step_emb     = step_emb if step_emb is not None else \
            StepIndexEmbedding(d_model=d_model, dropout=dropout)
        self.query_norm   = nn.LayerNorm(d_model)

        dp_rates = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, mlp_ratio, dropout, drop_path=dp_rates[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, num_target_vars)
        if predict_uncertainty:
            # log sigma^2 head, initialised so that sigma^2 = 1 at the start.
            self.log_var_head = nn.Linear(d_model, num_target_vars)
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.zeros_(self.log_var_head.bias)

    # ------------------------------------------------------------------

    def _run_blocks(self, h: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                h = _cp_checkpoint(block, h, kv, use_reentrant=False)
            else:
                h = block(h, kv)
        return h

    def forward(
        self,
        encoded_vis:    torch.Tensor,                    # (B, T*N_vis, d)
        spatial:        torch.Tensor,                    # (N, 15) or (B, N, 15)
        y_hours:        torch.Tensor,                    # (B,) or (B, K)
        delta_steps:    torch.Tensor,                    # (B,) or (B, K)
        station_masked: "torch.Tensor | None" = None,    # (B, N) bool
    ):
        """
        Returns (B, N, V_t) for single-delta inputs or (B, K, N, V_t) for
        multi-delta inputs; a tuple (mean, log_var) when predict_uncertainty.
        """
        B = encoded_vis.size(0)
        if spatial.dim() == 2:
            N, spatial_b = spatial.size(0), spatial.unsqueeze(0).expand(B, -1, -1)
        else:
            N, spatial_b = spatial.size(1), spatial

        is_multi = (y_hours.dim() == 2)
        if not is_multi:
            y_hours, delta_steps = y_hours.unsqueeze(1), delta_steps.unsqueeze(1)
        K = y_hours.shape[1]

        # Station part of the query, shared over lead times: (B, N, d)
        spatial_q = self.mask_token.expand(B, N, -1).contiguous()
        spatial_q = spatial_q + self.pos_emb(spatial_b[..., :2])
        spatial_q = spatial_q + self.station_emb(spatial_b[..., 2:])
        if station_masked is None:
            spatial_q = spatial_q + self.station_state[0]
        else:
            spatial_q = spatial_q + self.station_state[station_masked.long()]

        # Time part, shared over stations: (B, K, d)
        time_q = (self.temporal_emb(y_hours)
                  + self.delta_emb(delta_steps)
                  + self.step_emb((self.window_size - 1) + delta_steps))

        queries = spatial_q.unsqueeze(2) + time_q.unsqueeze(1)      # (B, N, K, d)
        queries = self.query_norm(queries)

        if self.station_local:
            # Each station attends only to its own T encoder tokens.
            L = encoded_vis.shape[1]
            if L % N != 0:
                raise RuntimeError(
                    f"station_local decoder needs every station present: encoder "
                    f"returned {L} tokens, not divisible by N={N}. Use mask_ratio=0.")
            T = L // N
            kv = encoded_vis.view(B, T, N, self.d_model).permute(0, 2, 1, 3) \
                            .reshape(B * N, T, self.d_model)
            h = self._run_blocks(queries.reshape(B * N, K, self.d_model), kv)
            h = h.reshape(B, N * K, self.d_model)
        else:
            h = self._run_blocks(queries.reshape(B, N * K, self.d_model), encoded_vis)

        h = self.norm(h)

        def _out(lin):
            o = lin(h).reshape(B, N, K, self.num_target_vars).permute(0, 2, 1, 3).contiguous()
            return o if is_multi else o[:, 0]

        mean = _out(self.head)
        if self.predict_uncertainty:
            return mean, _out(self.log_var_head)
        return mean
