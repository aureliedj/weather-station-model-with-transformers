"""
simple_mae.py — the canonical Masked Autoencoder for the station grid.

One mechanism, one objective: tokens are (station × hour) patches, a mask
pattern hides some of them, the encoder sees only the visible ones, a shallow
decoder reconstructs the hidden ones, and the loss is MSE on masked tokens
only (He et al. 2022, strict MAE).  The mask pattern — not the architecture —
defines the task:

    random tokens      → pretraining signal (spatio-temporal MAE,
                         Feichtenhofer et al. 2022)
    whole stations     → gap-filling (the operational scenario)
    the last 6 hours   → forecasting (all 36 10-min leads in one pass)

Compared to StationMAE (mae.py) there is NO delta grid, NO step/delta
embedding, NO cross-attention decoder and NO input-context pathway.  A masked
token IS the query; its temporal embedding IS the lead-time conditioning.

Embeddings (kept from the existing stack, per project decision):
    PositionalEmbedding  — Fourier over LV95 easting/northing      (WHERE)
    StationEmbedding     — MLP over 13 topographic features        (WHERE)
    TemporalEmbedding    — multi-scale Fourier over absolute hours (WHEN)
    + a learned window-slot embedding (S × d) — the plain MAE positional
      embedding over the S hour-slots of the window.  This replaces BOTH
      StepIndexEmbedding and DeltaTimeEmbedding: with S ≈ 18 discrete slots a
      lookup table is simpler and sufficient, and "lead time" is just a slot
      index past the visible context.

Token layout for a window of W = S·P steps (default S=18 hours, P=6 steps):
    raw token vector = [values·mask ‖ mask] over P steps × V vars = 2·P·V dims
    token embedding  = Linear(2·P·V → d) + pos + station + temporal + slot

Targets: the NUM_TARGET_VARIABLES (=5) forecast variables at all P steps of
each scored token (precipitation stays input-only, as everywhere else in the
project).  The loss follows the PROJECT convention rather than the strict
MAE: the window always splits into context (first S − future_slots hours,
"time ≤ 0") and future (last future_slots hours).  Context tokens are scored
on MASKED positions only (reconstruction — visible copies would dilute the
gradient, mae.py's δ=0 rule); future tokens are scored for EVERYONE, masked
or visible (forecast horizon, mae.py's δ>0 rule).
"""

import torch
import torch.nn as nn

from .encoder import TransformerBlock
from .embeddings import (
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
    NUM_VARIABLES,
    NUM_TARGET_VARIABLES,
    TARGET_VARIABLE_NAMES,
)

MASK_STRATEGIES = ("random", "station", "future")


class SimpleMAE(nn.Module):
    """
    Args:
        window_steps:   Input window length in 10-min steps (default 108 = 18 h).
        patch_steps:    Steps per token (default 6 = 1 hour). Must divide window.
        d_model:        Encoder width (default 256).
        enc_layers:     Encoder depth (default 8).
        enc_heads:      Encoder heads (default 8).
        dec_dim:        Decoder width (default 128) — MAE-style narrow decoder.
        dec_layers:     Decoder depth (default 2).
        dec_heads:      Decoder heads (default 4).
        mlp_ratio:      FFN expansion (default 4.0).
        dropout:        Dropout (default 0.0).
        mask_ratio:     Token mask ratio for the "random" strategy (default 0.75).
        station_ratio:  Station mask ratio for the "station" strategy (default 0.5).
        future_slots:   Hour-slots masked by the "future" strategy (default 6 = 6 h).
        strategy_probs: Sampling probabilities for (random, station, future)
                        during training (default 0.5 / 0.25 / 0.25).
    """

    def __init__(
        self,
        window_steps:   int   = 108,
        patch_steps:    int   = 6,
        d_model:        int   = 256,
        enc_layers:     int   = 8,
        enc_heads:      int   = 8,
        dec_dim:        int   = 128,
        dec_layers:     int   = 2,
        dec_heads:      int   = 4,
        mlp_ratio:      float = 4.0,
        dropout:        float = 0.0,
        mask_ratio:     float = 0.75,
        station_ratio:  float = 0.5,
        future_slots:   int   = 6,
        strategy_probs: "tuple[float, float, float]" = (0.5, 0.25, 0.25),
        fuse_layers:    int   = 0,
    ):
        super().__init__()
        assert window_steps % patch_steps == 0, "patch_steps must divide window_steps"
        self.W  = window_steps
        self.P  = patch_steps
        self.S  = window_steps // patch_steps          # hour-slots per window
        self.V  = NUM_VARIABLES                        # 6 input vars
        self.Vt = NUM_TARGET_VARIABLES                 # 5 target vars
        self.d_model        = d_model
        self.mask_ratio     = mask_ratio
        self.station_ratio  = station_ratio
        self.future_slots   = future_slots
        self.strategy_probs = strategy_probs

        # ── Tokenizer: raw hour-patch → d_model ──────────────────────────
        # [values·mask ‖ mask] so a true zero (station mean) is distinguishable
        # from an absent sensor — the indicator channel replaces both the old
        # sensor_mask_token and var_absent_embedding machinery.
        self.tokenizer = nn.Linear(2 * self.P * self.V, d_model)

        # ── Existing embedding stack (kept per project decision) ─────────
        self.pos_emb      = PositionalEmbedding(d_model=d_model, dropout=dropout)
        self.station_emb  = StationEmbedding(d_model=d_model, dropout=dropout)
        self.temporal_emb = TemporalEmbedding(d_model=d_model, dropout=dropout)
        # Learned slot embedding — replaces StepIndex/DeltaTime embeddings.
        self.slot_emb = nn.Parameter(torch.zeros(self.S, d_model))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)

        self.token_norm = nn.LayerNorm(d_model)

        # ── Encoder: plain ViT blocks over VISIBLE tokens only ───────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, enc_heads, mlp_ratio, dropout)
            for _ in range(enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # ── Reconstruction stage — one dial, two regimes ─────────────────
        # fuse_layers == 0 (MAE-classic): narrow shallow decoder over the
        #   full grid; the forecast is computed OUTSIDE the trunk.
        # fuse_layers  > 0 (trunk forecasting / BERT-hybrid): mask tokens are
        #   inserted at FULL width after the visible-only encoder and
        #   fuse_layers more d_model blocks run over the complete grid —
        #   the forecast is computed BY THE TRANSFORMER itself.  Pure BERT is
        #   the limit enc_layers=0 (all layers full-grid).  The narrow-decoder
        #   modules are not created in this regime.
        self.fuse_layers = int(fuse_layers)
        if self.fuse_layers > 0:
            self.mask_token = nn.Parameter(torch.zeros(d_model))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.fuse_blocks = nn.ModuleList([
                TransformerBlock(d_model, enc_heads, mlp_ratio, dropout)
                for _ in range(self.fuse_layers)
            ])
            self.fuse_norm = nn.LayerNorm(d_model)
            self.head      = nn.Linear(d_model, self.P * self.Vt)
            self.enc2dec = self.emb2dec = self.dec_blocks = self.dec_norm = None
        else:
            self.enc2dec    = nn.Linear(d_model, dec_dim)
            self.emb2dec    = nn.Linear(d_model, dec_dim)   # embeddings for masked slots
            self.mask_token = nn.Parameter(torch.zeros(dec_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.dec_blocks = nn.ModuleList([
                TransformerBlock(dec_dim, dec_heads, mlp_ratio, dropout)
                for _ in range(dec_layers)
            ])
            self.dec_norm = nn.LayerNorm(dec_dim)
            self.head     = nn.Linear(dec_dim, self.P * self.Vt)

    # ------------------------------------------------------------------
    # Mask generation — (B, S, N) bool, True = HIDDEN from the encoder
    # ------------------------------------------------------------------
    def make_mask(
        self,
        B: int,
        N: int,
        device: torch.device,
        strategy: "str | None" = None,
    ) -> tuple[torch.Tensor, str]:
        """One strategy per BATCH so every sample has the same visible count
        (uniform sequence lengths — no padding needed in the encoder)."""
        if strategy is None:
            r = torch.rand((), device=device).item()
            p_r, p_s, _ = self.strategy_probs
            strategy = ("random" if r < p_r
                        else "station" if r < p_r + p_s
                        else "future")
        assert strategy in MASK_STRATEGIES, f"unknown strategy {strategy!r}"

        S, L = self.S, self.S * N
        if strategy == "random":
            n_mask = int(L * self.mask_ratio)
            noise  = torch.rand(B, L, device=device)
            order  = noise.argsort(dim=1)
            mask   = torch.zeros(B, L, dtype=torch.bool, device=device)
            mask.scatter_(1, order[:, :n_mask], True)
            mask = mask.view(B, S, N)
        elif strategy == "station":
            n_mask = int(N * self.station_ratio)
            noise  = torch.rand(B, N, device=device)
            order  = noise.argsort(dim=1)
            sel    = torch.zeros(B, N, dtype=torch.bool, device=device)
            sel.scatter_(1, order[:, :n_mask], True)
            mask = sel.unsqueeze(1).expand(B, S, N).contiguous()
        else:  # future
            mask = torch.zeros(B, S, N, dtype=torch.bool, device=device)
            mask[:, S - self.future_slots:, :] = True
        return mask, strategy

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x:        torch.Tensor,            # (B, W, N, V)  normalised observations
        x_mask:   torch.Tensor,            # (B, W, N, V)  sensor availability
        spatial:  torch.Tensor,            # (N, 15) or (B, N, 15)
        x_hours:  torch.Tensor,            # (B, W)        hours since epoch
        strategy: "str | None" = None,     # None → sampled from strategy_probs
        token_mask: "torch.Tensor | None" = None,  # (B, S, N) bool override —
                                           # custom mask pattern (evaluation
                                           # scenarios); must hide the same
                                           # count per sample in the batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            loss:       scalar — MSE on masked tokens, present sensors only
            preds:      (B, W, N, Vt) reconstruction of the FULL grid
                        (only the masked region is trained/meaningful)
            token_mask: (B, S, N) bool — True where hidden from the encoder
            strategy:   the mask strategy actually used
        """
        B, W, N, V = x.shape
        S, P = self.S, self.P
        assert W == self.W, f"window {W} != configured {self.W}"

        # ── 1. Raw hour-patch vectors:  [values·mask ‖ mask] ─────────────
        xv = (x * x_mask).view(B, S, P, N, V)
        xm = x_mask.view(B, S, P, N, V)
        raw = torch.cat([xv, xm], dim=2)                       # (B, S, 2P, N, V)
        raw = raw.permute(0, 1, 3, 2, 4).reshape(B, S, N, 2 * P * V)
        tok = self.tokenizer(raw)                              # (B, S, N, d)

        # ── 2. Additive embeddings (shared basis for encoder AND decoder) ─
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)                     # (1, N, 15)
        where = (self.pos_emb(spatial[..., :2])
                 + self.station_emb(spatial[..., 2:]))         # (1|B, N, d)
        hours = x_hours.view(B, S, P)[:, :, 0]                 # slot start hour
        when  = self.temporal_emb(hours)                       # (B, S, d)
        emb = (where.unsqueeze(1)                              # (1|B, 1, N, d)
               + when.unsqueeze(2)                             # (B, S, 1, d)
               + self.slot_emb.view(1, S, 1, -1))              # (1, S, 1, d)
        tok = self.token_norm(tok + emb)                       # (B, S, N, d)

        # ── 3. Mask & encode visible tokens only ─────────────────────────
        if token_mask is not None:
            strategy = strategy or "custom"
        else:
            token_mask, strategy = self.make_mask(B, N, x.device, strategy)
        flat_tok  = tok.view(B, S * N, -1)
        flat_mask = token_mask.view(B, S * N)
        vis_idx = (~flat_mask).float().argsort(dim=1, descending=True)
        n_vis   = int((~flat_mask[0]).sum())                   # uniform per batch
        vis_idx = vis_idx[:, :n_vis]                           # (B, L_vis)
        h = flat_tok.gather(1, vis_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        for blk in self.blocks:
            h = blk(h)
        h = self.enc_norm(h)                                   # (B, L_vis, d)

        # ── 4. Reconstruction stage over the full grid ───────────────────
        # Masked slots: mask_token + their embeddings — the temporal/slot
        # embedding of a hidden slot IS the conditioning (lead time for future
        # slots, station identity for gap-filling slots); no delta embedding.
        # BOTH task types (future forecast, hidden-station reconstruction)
        # flow through the same stage — one mechanism, two tasks.
        emb_flat = (emb.expand(B, S, N, self.d_model)
                    .reshape(B, S * N, self.d_model))
        if self.fuse_layers > 0:
            # Trunk forecasting: full-width mask tokens join the transformer.
            grid = self.mask_token + emb_flat                  # (B, S·N, d)
            grid = grid.scatter(
                1,
                vis_idx.unsqueeze(-1).expand(-1, -1, grid.shape[-1]),
                h.to(grid.dtype),          # no projection — same width
            )
            for blk in self.fuse_blocks:
                grid = blk(grid)
            grid = self.fuse_norm(grid)
        else:
            # MAE-classic narrow decoder.
            grid = self.mask_token + self.emb2dec(emb_flat)    # (B, S·N, dec)
            # .to(grid.dtype): under bf16 autocast the projections return
            # bfloat16 while grid is fp32 (fp32 mask_token promotes the sum);
            # scatter() requires exactly matching dtypes.
            grid = grid.scatter(
                1,
                vis_idx.unsqueeze(-1).expand(-1, -1, grid.shape[-1]),
                self.enc2dec(h).to(grid.dtype),
            )
            for blk in self.dec_blocks:
                grid = blk(grid)
            grid = self.dec_norm(grid)
        out = self.head(grid)                                  # (B, S·N, P·Vt)
        preds = (out.view(B, S, N, P, self.Vt)
                 .permute(0, 1, 3, 2, 4)
                 .reshape(B, W, N, self.Vt))                   # (B, W, N, Vt)

        # ── 5. Loss — project convention, not strict MAE ─────────────────
        # The window always splits into CONTEXT (first S - future_slots hours,
        # "time ≤ 0") and FUTURE (last future_slots hours, "time > 0"),
        # independent of the mask strategy:
        #   • context slots (reconstruction of observed times, the t=0 regime):
        #     scored on MASKED tokens only — visible-token identity copies
        #     would dilute the gradient (same rule as mae.py's δ=0).
        #   • future slots (the forecast horizon, +10 min … +6 h):
        #     scored on ALL tokens, masked or visible (same rule as mae.py's
        #     δ>0 — under the random/station strategies a visible future
        #     token still contributes forecast learning signal).
        f0 = S - self.future_slots
        scope = token_mask.clone()                             # (B, S, N)
        scope[:, f0:, :] = True                                # future → everyone
        step_mask = (scope.unsqueeze(2)                        # (B, S, 1, N)
                     .expand(B, S, P, N)
                     .reshape(B, W, N, 1))
        valid = step_mask * x_mask[..., :self.Vt]              # (B, W, N, Vt)
        err2  = (preds - x[..., :self.Vt]).pow(2) * valid
        loss  = err2.sum() / valid.sum().clamp(min=1.0)

        return loss, preds, token_mask, strategy

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
