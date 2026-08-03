"""
masked_transformer.py — encoder-only transformer with in-place station masking.

The simplest design in the project that still does BOTH tasks: one stack of
plain self-attention blocks, one linear head, and masking as the only extra
mechanism. No decoder, no query tokens, no delta embedding, no mask-token
insertion stage.

    tokenize → corrupt masked stations → encoder → pool → multi-horizon head

Why in-place corruption instead of MAE-style removal
----------------------------------------------------
The readout pools EACH STATION'S OWN tokens, so a station that has been
removed from the sequence has nothing to pool. Masked stations therefore keep
their slot: only the VALUE component of their token is replaced by a learned
`station_mask_token`, while position, topography, absolute time and step
embeddings survive — "this is station n, at time t, contents unknown". The
encoder builds their representation purely from attention to visible
neighbours, which is exactly the gap-filling signal.

This is the BERT convention (corrupt in place) rather than the MAE one (drop
and re-insert). The trade-off is deliberate and worth stating in the report:
   • lost   — He et al.'s efficiency trick; the sequence never shrinks, so a
              50%-masked batch costs the same as an unmasked one.
   • gained — a single stack with no decoder at all, and a representation for
              every station whether observed or not.

Tasks, both from the same forward pass
--------------------------------------
    forecasting  — head slot k predicts lead delta_grid[k] for every station
    gap-filling  — for a MASKED station, the k=0 output (delta=0, the last
                   observed timestep) is the reconstruction

Loss follows the project convention: delta=0 is scored on MASKED stations
only (a visible station's k=0 answer is in its own input — supervising it
would reward copying), delta>0 on ALL stations.

Lead-time conditioning is the head's output slot rather than an additive
embedding: the k-th block of head weights is that lead's own parameterisation.
"""

import torch
import torch.nn as nn

from .encoder import TransformerBlock
from .embeddings import (
    VariableProjection,
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
    StepIndexEmbedding,
    NUM_VARIABLES,
    NUM_TARGET_VARIABLES,
)


class MaskedStationTransformer(nn.Module):
    """
    Args:
        window_steps:  Input window in 10-min steps (default 72 = 12 h).
        temporal_patch: Steps merged into one token (default 6 = hourly).
                       Must divide window_steps. The sequence is
                       (window_steps / P) x N_stations and does NOT shrink
                       with masking, so keep P >= 3 to stay out of the
                       attention-dilution regime (see run script).
        num_horizons:  K output slots (default 13 = 0, 30 min … 6 h).
        horizon_steps: The lead-times those slots mean, in 10-min steps.
        d_model / n_layers / n_heads / mlp_ratio / dropout: transformer size.
        mask_ratio:    Fraction of stations corrupted per sample (default 0.5).
        huber_delta:   Huber delta in normalised units (default 1.0; 0 = MSE).
        pool:          "last" (default) — pool each station's final time slot,
                       the attention analogue of the LSTM's h_W;
                       "mean" — average that station's slots.
    """

    def __init__(
        self,
        window_steps:   int   = 72,
        temporal_patch: int   = 6,
        num_horizons:   int   = 13,
        horizon_steps:  "list | None" = None,
        d_model:        int   = 384,
        n_layers:       int   = 8,
        n_heads:        int   = 8,
        mlp_ratio:      float = 4.0,
        dropout:        float = 0.0,
        mask_ratio:     float = 0.5,
        huber_delta:    float = 1.0,
        pool:           str   = "last",
    ):
        super().__init__()
        assert window_steps % temporal_patch == 0, "patch must divide the window"
        assert pool in ("last", "mean")
        self.W  = window_steps
        self.P  = temporal_patch
        self.T  = window_steps // temporal_patch      # temporal positions
        self.V  = NUM_VARIABLES
        self.Vt = NUM_TARGET_VARIABLES
        self.d_model     = d_model
        self.mask_ratio  = mask_ratio
        self.huber_delta = float(huber_delta)
        self.pool        = pool
        self.horizon_steps = list(horizon_steps) if horizon_steps is not None \
            else list(range(0, 3 * num_horizons, 3))
        self.K = len(self.horizon_steps)

        # ── Embeddings — the same stack as every other model here ────────
        self.var_proj     = VariableProjection(num_vars=self.V, d_model=d_model)
        self.pos_emb      = PositionalEmbedding(d_model=d_model, dropout=dropout)
        self.station_emb  = StationEmbedding(d_model=d_model, dropout=dropout)
        self.temporal_emb = TemporalEmbedding(d_model=d_model, dropout=dropout)
        self.step_emb     = StepIndexEmbedding(d_model=d_model, dropout=dropout)

        # Replaces the VALUE embedding of a corrupted station. Zero-init: at
        # step 0 a masked station contributes only its identity/time
        # embeddings, and the model learns what "unobserved" should look like.
        self.station_mask_token = nn.Parameter(torch.zeros(d_model))

        self.token_norm = nn.LayerNorm(d_model)

        # ── Patch merge (identical construction to the encoder's) ────────
        if self.P > 1:
            self.patch_norm  = nn.LayerNorm(self.P * d_model)
            self.patch_merge = nn.Linear(self.P * d_model, d_model)
        else:
            self.patch_norm = self.patch_merge = None

        # ── One stack, one block type ────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── Multi-horizon head: station vector → all K leads at once ─────
        self.head = nn.Linear(d_model, self.K * self.Vt)

    # ------------------------------------------------------------------
    def _mask_stations(self, B: int, N: int, device) -> torch.Tensor:
        """(B, N) bool — True where the station's values are corrupted."""
        n_mask = int(N * self.mask_ratio)
        if n_mask == 0:
            return torch.zeros(B, N, dtype=torch.bool, device=device)
        order = torch.rand(B, N, device=device).argsort(dim=1)
        m = torch.zeros(B, N, dtype=torch.bool, device=device)
        m.scatter_(1, order[:, :n_mask], True)
        return m

    def _patchify(self, t: torch.Tensor) -> torch.Tensor:
        if self.P <= 1:
            return t
        B, W, N, D = t.shape
        t = t.reshape(B, W // self.P, self.P, N, D).permute(0, 1, 3, 2, 4)
        t = t.reshape(B, W // self.P, N, self.P * D)
        return self.patch_merge(self.patch_norm(t))

    # ------------------------------------------------------------------
    def forward(
        self,
        x:            torch.Tensor,            # (B, W, N, V) normalised obs
        x_mask:       torch.Tensor,            # (B, W, N, V) sensor availability
        spatial:      torch.Tensor,            # (N, 15) or (B, N, 15)
        x_hours:      torch.Tensor,            # (B, W) hours since epoch
        station_mask: "torch.Tensor | None" = None,   # (B, N) bool override
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            preds:        (B, K, N, Vt)
            station_mask: (B, N) bool — True = corrupted (for the loss / eval)
        """
        B, W, N, V = x.shape
        assert W == self.W, f"window {W} != configured {self.W}"

        # 1. value embedding
        var_tokens = self.var_proj(
            x.reshape(B * W, N, V), x_mask.reshape(B * W, N, V)
        ).view(B, W, N, self.d_model)

        # 2. IN-PLACE MASKING: swap the value embedding for the learned token.
        #    Identity/time embeddings (added below) are deliberately kept.
        if station_mask is None:
            station_mask = self._mask_stations(B, N, x.device)
        sm = station_mask.view(B, 1, N, 1).to(var_tokens.dtype)
        var_tokens = var_tokens * (1 - sm) + self.station_mask_token * sm

        # 3. identity / time embeddings
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)
        where = (self.pos_emb(spatial[..., :2])
                 + self.station_emb(spatial[..., 2:])).unsqueeze(1)   # (1|B,1,N,d)
        when  = self.temporal_emb(x_hours).unsqueeze(2)               # (B,W,1,d)
        step  = self.step_emb(torch.arange(W, device=x.device)).view(1, W, 1, -1)
        tokens = self.token_norm(var_tokens + where + when + step)

        # 4. patch merge → flat sequence → one transformer stack
        h = self._patchify(tokens)                        # (B, T, N, d)
        T = h.shape[1]
        h = h.reshape(B, T * N, self.d_model)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).view(B, T, N, self.d_model)

        # 5. pool one vector per station
        s = h[:, -1] if self.pool == "last" else h.mean(dim=1)        # (B, N, d)

        # 6. multi-horizon head
        preds = self.head(s).view(B, N, self.K, self.Vt).permute(0, 2, 1, 3)
        return preds.contiguous(), station_mask                       # (B,K,N,Vt)

    # ------------------------------------------------------------------
    def loss(
        self,
        preds:        torch.Tensor,   # (B, K, N, Vt)
        y:            torch.Tensor,   # (B, K, N, V)  targets at the K leads
        y_mask:       torch.Tensor,   # (B, K, N, V)  sensor availability
        station_mask: torch.Tensor,   # (B, N) bool — True = corrupted
    ) -> torch.Tensor:
        """
        Project convention:
          k=0 (delta=0, the last observed step) → MASKED stations only.
              A visible station's answer is in its own input; supervising it
              there would reward copying, not gap-filling.
          k>0 (forecast horizons)              → ALL stations.
        Huber(delta) per element, each variable normalised by its own
        present-sensor count, then averaged over the K horizons.
        """
        t = y[..., :self.Vt]
        m = y_mask[..., :self.Vt] > 0.5                               # (B,K,N,Vt)
        scope = torch.ones_like(m)
        if int(self.horizon_steps[0]) == 0:
            scope[:, 0] = station_mask.view(-1, 1, station_mask.shape[-1], 1) \
                                      .expand_as(m[:, :1])[:, 0]
        valid = (m & scope).to(preds.dtype)

        err = preds - t
        if self.huber_delta > 0:
            d = self.huber_delta
            a = err.abs()
            elem = torch.where(a <= d, 0.5 * err.pow(2), d * (a - 0.5 * d))
        else:
            elem = err.pow(2)

        # per-variable normalisation, then mean over horizons
        num = (elem * valid).sum(dim=(0, 2))          # (K, Vt)
        den = valid.sum(dim=(0, 2)).clamp(min=1.0)    # (K, Vt)
        return (num / den).mean()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
