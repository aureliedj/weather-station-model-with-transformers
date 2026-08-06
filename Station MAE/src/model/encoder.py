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

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    SPATIAL_INPUT_DIM,
    STATIC_SLOTS_DEFAULT,
)


# ---------------------------------------------------------------------------
# Stochastic depth (DropPath)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """
    Stochastic depth regularisation (Huang et al., 2016; used in DeiT, Swin, etc.).

    During training, the *entire* residual path is dropped with probability
    ``drop_prob`` and the output is replaced by the unchanged residual input.
    At test time the path is always kept and the output is scaled by
    ``1 - drop_prob`` implicitly via the per-sample binary mask approach
    (no separate rescaling required — the mask already divides by keep_prob).

    Drop decisions are independent per sample in the batch, so information
    can still flow through via other samples during gradient accumulation.

    Args:
        drop_prob: Probability of dropping the residual path (0 = never drop).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob    = 1.0 - self.drop_prob
        # Shape: (B, 1, 1, …) — broadcasts over all token/feature dimensions
        shape        = (x.shape[0],) + (1,) * (x.ndim - 1)
        # float32, NOT x.dtype. Under --amp --bf16 x is bfloat16, which carries
        # an 8-bit mantissa: near 1.0 the representable values are spaced ~1/256.
        # `rand + keep_prob` then ROUNDS before the floor, so with drop_prob=0.1
        # a sample at rand=0.0999 becomes 0.9999 -> 1.0 -> kept, when it should
        # have been dropped. The realised drop rate drifts below the nominal one
        # by roughly half the bf16 spacing, i.e. the regulariser is quietly
        # weaker than configured. Draw in float32 and cast after the threshold.
        random_tensor = torch.rand(shape, dtype=torch.float32, device=x.device)
        # Floor to 0/1; divide by keep_prob to maintain expected value = 1
        random_tensor = torch.floor(random_tensor + keep_prob) / keep_prob
        return x * random_tensor.to(x.dtype)


# ---------------------------------------------------------------------------
# Transformer blocks
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
        drop_path:   Stochastic depth drop probability (default 0.0 = disabled).
                     Each residual branch is dropped independently with this
                     probability during training, acting as a strong regulariser.
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
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        # Self-attention with residual
        # need_weights=False lets PyTorch route through F.scaled_dot_product_attention
        # (Flash Attention on CUDA), avoiding materialising the full O(seq²) matrix.
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path(attn_out)

        # FFN with residual
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class FactorisedTransformerBlock(nn.Module):
    """
    Factorised (axial) attention block operating on 4-D token grids (B, W, N, d).

    Instead of attending jointly over all W×N tokens (O((W·N)²) pairs), this
    block alternates two cheaper operations:

      1. Temporal attention  — each station independently attends over its W
         timesteps → reshape to (B·N, W, d), run MHA, reshape back.

         Optionally uses *local windowed attention* (``temporal_window`` > 0):
         the W-step sequence is split into non-overlapping chunks of size tw,
         and attention runs only within each chunk.  Complexity drops from
         O(N·W²) to O(N·(W/tw)·tw²) = O(N·W·tw) — e.g. tw=6, W=72 → 12×
         cheaper.  Adjacent layers use a Swin-style half-window shift so that
         information can propagate across chunk boundaries after 2 layers.

      2. Spatial attention   — all N stations attend to each other at every
         timestep → reshape to (B·W, N, d), run MHA, reshape back.
         Always active (the ``spatial_attn=False`` switch was removed
         purely temporal (station-independent).  Spatial reasoning is then
         left entirely to the decoder's cross- or self-attention.

      3. Shared FFN applied element-wise over the last dimension.

    Complexity with both sub-layers: O(N·W² + W·N²)  vs  O((W·N)²) for flat.
    Complexity with temporal_window=tw: O(N·W·tw + W·N²) — local temporal.
    At W=288, N_vis≈100: ~8.3M vs ~830M attention pairs — roughly 100× cheaper.

    Args:
        d_model:         Model dimension.
        num_heads:       Attention heads (shared across both sub-layers).
        mlp_ratio:       FFN hidden-dim ratio (default 4.0).
        dropout:         Dropout in attention and FFN (default 0.1).
                         Reduces encoder cost from O(N·W²+W·N²) to O(N·W²).
        temporal_window: Local attention window size in timesteps (0 = full).
                         W must be exactly divisible by temporal_window.
                         Example: W=72, temporal_window=6 → 12 chunks of 6
                         steps (1-hour windows at 10-min resolution).
        shift:           If True, apply a half-window circular shift before
                         chunking (Swin-style).  Alternate layers should have
                         shift=True so tokens across chunk boundaries can
                         communicate after two layers.
        drop_path:       Stochastic depth drop probability (default 0.0 = disabled).
                         Applied independently to each of the three residual
                         branches (temporal attn, spatial attn, FFN).
    """

    def __init__(
        self,
        d_model:          int,
        num_heads:        int,
        mlp_ratio:        float = 4.0,
        dropout:          float = 0.1,
        spatial_attn:     bool  = True,
        temporal_window:  int   = 0,
        shift:            bool  = False,
        drop_path:        float = 0.0,
    ):
        super().__init__()

        self.spatial_attn    = spatial_attn
        self.temporal_window = temporal_window
        # shift only makes sense when windowing is active
        self.shift           = shift and (temporal_window > 0)

        # ── Temporal sub-layer (each station over W timesteps) ────────────
        self.norm_t = nn.LayerNorm(d_model)
        self.attn_t = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # ── Spatial sub-layer (all N stations at each timestep) ──────────
        # spatial_attn=False makes the encoder STATION-INDEPENDENT: each station
        # is encoded from its own temporal window with no cross-station mixing.
        # That is the controlled study against the LSTM — if the error curves
        # coincide, neighbouring stations are not contributing.
        if spatial_attn:
            self.norm_s = nn.LayerNorm(d_model)
            self.attn_s = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )

        # ── Shared FFN ────────────────────────────────────────────────────
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

        # ── Stochastic depth — one DropPath shared across all residuals ───
        self.drop_path = DropPath(drop_path)

    def _temporal_attn(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Self-attention along the temporal axis, full or windowed.

        Args:
            xt: (B·N, W, d_model)
        Returns:
            (B·N, W, d_model)  — residual NOT yet added (caller adds it)
        """
        BN, W, D = xt.shape
        tw = self.temporal_window

        if tw == 0 or tw >= W:
            # ── Full temporal attention ───────────────────────────────────
            at, _ = self.attn_t(xt, xt, xt, need_weights=False)
            return at

        # ── Local windowed temporal attention ────────────────────────────
        # Each station's W-step sequence is split into non-overlapping chunks
        # of size tw.  Attention runs only within each chunk → O(W·tw) instead
        # of O(W²).  Swin-style half-window shift (when self.shift=True) lets
        # adjacent layers bridge chunk boundaries.
        assert W % tw == 0, (
            f"temporal_window={tw} must divide W={W} exactly.  "
            f"Choose a value that divides your --window size."
        )
        shift_size = tw // 2 if self.shift else 0

        if shift_size > 0:
            xt = torch.roll(xt, shifts=-shift_size, dims=1)

        num_chunks = W // tw
        # (BN, W, D) → (BN·num_chunks, tw, D)
        xt_c = xt.reshape(BN * num_chunks, tw, D)
        at_c, _ = self.attn_t(xt_c, xt_c, xt_c, need_weights=False)
        # (BN·num_chunks, tw, D) → (BN, W, D)
        at = at_c.reshape(BN, W, D)

        if shift_size > 0:
            at = torch.roll(at, shifts=shift_size, dims=1)

        return at

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, N, d_model)
        Returns:
            (B, W, N, d_model)
        """
        B, W, N, D = x.shape

        # ── 1. Temporal attention: (B·N, W, D) ───────────────────────────
        # Permute so N is before W, then merge B and N into a single batch dim.
        # Each of the B·N sequences has length W (the temporal axis).
        xt    = x.permute(0, 2, 1, 3).reshape(B * N, W, D)    # (B·N, W, D)
        xt_n  = self.norm_t(xt)
        xt    = xt + self.drop_path(self._temporal_attn(xt_n))
        x     = xt.reshape(B, N, W, D).permute(0, 2, 1, 3)    # (B, W, N, D)

        # ── 2. Spatial attention: (B·W, N, D) ────────────────────────────
        # Merge B and W into a single batch dim.
        # Each of the B·W sequences has length N (the station axis).
        # Attention here is FULL over the station axis — the windowing options
        # apply to the temporal axis only. That is what makes the station axis
        # permutation-equivariant, and why sorting the mask indices is free.
        if self.spatial_attn:
            xs     = x.reshape(B * W, N, D)                    # (B·W, N, D)
            xs_n   = self.norm_s(xs)
            as_, _ = self.attn_s(xs_n, xs_n, xs_n, need_weights=False)
            xs     = xs + self.drop_path(as_)
            x      = xs.reshape(B, W, N, D)                    # (B, W, N, D)

        # ── 3. FFN (applied over last dim, works on any leading shape) ────
        x = x + self.drop_path(self.ffn(self.norm_ff(x)))
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
        use_checkpoint:       If True, use gradient checkpointing on each transformer block.
                              Trades ~33% extra compute for ~66% less activation memory —
                              enables larger batches / deeper models on limited GPU VRAM.
        factorised:           If True, use FactorisedTransformerBlock (axial attention over
                              the W×N grid) instead of flat self-attention over W·N tokens.
                              Reduces complexity from O((W·N)²) to O(N·W²+W·N²) — ~100×
                              cheaper at W=288, N=100.  Tokens remain in (B,W,N,d) shape
                              through the blocks; flattened to (B,W·N_vis,d) at output to
                              preserve the decoder interface.
        temporal_window:      Only used when factorised=True.  Local attention window
                              size in timesteps (0 = full attention over all W steps).
                              W must be exactly divisible by temporal_window.
                              Odd-indexed blocks automatically use a Swin-style
                              half-window shift so tokens can communicate across
                              chunk boundaries after two layers.
                              Example: W=72, temporal_window=6 → 12 one-hour chunks
                              at 10-min resolution, 12× cheaper temporal attention.
                              Takes precedence over factorised when both are True.
                              Strongly recommended with --grad_checkpoint on limited VRAM.
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
        factorised:           bool  = False,
        spatial_attn:         bool  = True,
        temporal_window:      int   = 0,
        temporal_patch:       int   = 1,
        value_embedding:      str   = "linear",     # v18: "linear" | "fourier"
        wind_pair:            "tuple | None" = None,  # v18: e.g. (3, 4) for u,v
        static_in_token:      bool  = False,        # v21: Aurora-style statics
        drop_path_rate:       float = 0.0,
        step_emb:             "nn.Module | None" = None,
        pos_emb:              "nn.Module | None" = None,
        station_emb:          "nn.Module | None" = None,
        temporal_emb:         "nn.Module | None" = None,
    ):
        super().__init__()

        self.d_model         = d_model
        self.mask_ratio      = mask_ratio
        self.temporal_patch  = int(temporal_patch)
        self.use_checkpoint  = use_checkpoint
        self.factorised      = bool(factorised)

        # --- Learnable sensor mask token (ViT-MAE style) ---
        # NOTE: the former ``sensor_mask_token`` (learned fill-value for absent
        # sensors) has been REMOVED. It was a dead parameter: the fill was
        # applied before var_proj, which then re-multiplied contributions by the
        # sensor mask — zeroing exactly the entries the fill had written. The
        # absent-sensor signal now lives where it takes effect:
        # VariableProjection.var_absent_embedding (learned per-variable absent
        # embedding, BERT-mask-style). Checkpoints from before this change are
        # NOT numerically compatible with this encoder (token math changed).

        # --- Embedding modules (four components: p1, p2, v, t) ---
        # v21: when static_in_token, the 15 static features become extra SLOTS
        # inside VariableProjection — scaled by the same mechanism as the
        # weather — and pos_emb / station_emb drop out of the token sum.
        self.static_in_token = bool(static_in_token)
        self.var_proj     = VariableProjection(
            num_vars=num_vars, d_model=d_model,
            value_embedding=value_embedding, wind_pair=wind_pair,
            static_slots=(STATIC_SLOTS_DEFAULT if static_in_token else None),
            static_dim=SPATIAL_INPUT_DIM)
        # p1/p2/t — shared with the decoder (see mae.py), same rationale as
        # step_emb below: a station's position, topography and a given
        # absolute time are the same fact on both sides of cross-attention.
        # Two independently-constructed modules would only share their fixed
        # Fourier frequencies, not their trained MLP weights, so "this
        # station" on the encoder side and "this station" on the decoder side
        # would drift apart during training instead of being provably the
        # same vector. Falls back to its own instance when used standalone
        # (e.g. tests, notebooks). NOT applied to var_proj — the actual
        # observed VALUE has no decoder-side counterpart to share with.
        self.pos_emb      = pos_emb if pos_emb is not None else \
            PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim,
                                dropout=dropout)
        self.station_emb  = station_emb if station_emb is not None else \
            StationEmbedding(d_model=d_model, input_dim=station_char_dim,
                             dropout=dropout)
        self.temporal_emb = temporal_emb if temporal_emb is not None else \
            TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim,
                              dropout=dropout)

        # s — Step-index positional embedding (within-window relative order).
        # Encodes the INTEGER step index 0..W-1 so the model knows WHERE each
        # token sits inside the 72-step (12 h) input window, independently of
        # the absolute time encoded by temporal_emb.
        #
        # Shared with the decoder: the decoder's forecast horizons only cover a
        # sparse subset of the unified step timeline (e.g. stride-3 → indices
        # 74, 77, ..., 107 for W=72), so most step indices past W-1 are never
        # visited by either side except at the single boundary point W-1
        # (delta=0). A separate StepIndexEmbedding per module would only share
        # its fixed Fourier frequencies, not its trained MLP projection weights
        # — "encoder step k" and "decoder step k" would drift apart during
        # training despite the docstring's intent. Passing the SAME module
        # instance in from StationMAE (see mae.py) makes the two genuinely
        # identical rather than coincidentally similar. Falls back to building
        # its own instance when used standalone (e.g. tests, notebooks).
        self.step_emb = step_emb if step_emb is not None else \
            StepIndexEmbedding(d_model=d_model, dropout=dropout)

        # --- Post-assembly normalisation ---
        # Applied after summing var_proj + spatial_emb + temporal_emb to keep
        # the three independently-initialised components on a common scale
        # before they enter the first transformer block.
        self.token_norm = nn.LayerNorm(d_model)

        # --- Transformer blocks ---
        # FactorisedTransformerBlock (axial attention over the W x N grid) when
        # factorised=True; otherwise the flat TransformerBlock, kept so older
        # checkpoints stay loadable.
        #
        # temporal_window > 0 enables local windowed temporal attention inside the
        # factorised block; odd-indexed blocks take a half-window Swin shift so
        # cross-chunk communication emerges after two layers. v15 onward runs with
        # temporal_window=0 (patching replaced windowing), but the option is live.
        #
        # REMOVED (v22 cleanup): JointSpatioTemporalBlock and
        # WindowedFlatTransformerBlock, ~348 lines that no configuration has
        # selected since v14. See report/v17-v19-v20-changes.md; git has the code.
        #
        # Stochastic depth: drop_path_rate is the maximum drop probability (applied
        # to the deepest layer).  Rates increase linearly from 0.0 (layer 0) to
        # drop_path_rate (layer num_layers-1), following the standard recipe from
        # Huang et al. (2016) and used in DeiT / Swin Transformer.
        dp_rates = [
            drop_path_rate * i / max(num_layers - 1, 1)
            for i in range(num_layers)
        ]
        if factorised:
            self.blocks = nn.ModuleList([
                FactorisedTransformerBlock(
                    d_model, num_heads, mlp_ratio, dropout,
                    spatial_attn=spatial_attn,
                    temporal_window=temporal_window,
                    shift=(i % 2 == 1),
                    drop_path=dp_rates[i],
                )
                for i in range(num_layers)
            ])
        else:
            # Full flat: standard self-attention over all W·N_vis tokens.
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, mlp_ratio, dropout,
                                 drop_path=dp_rates[i])
                for i in range(num_layers)
            ])

        self.norm = nn.LayerNorm(d_model)

        # ── Temporal patch merging (ViT / PatchTST / Swin patch-merge style) ────
        # Groups P consecutive timesteps into ONE token, so the encoder sees
        # W/P temporal positions instead of W.
        #
        # Why: with W=72 raw steps and windowed attention (tw=6) the newest token
        # only reaches 4 h of the 12 h window after 8 layers — measurably too
        # short. At P=6 the sequence is 12 tokens, and FULL attention over
        # 12 x N_vis = 936 tokens costs ~0.88M score entries per layer versus
        # ~2.6M for the windowed setup it replaces. Cheaper AND complete
        # coverage from layer 1, which is why windowing is retired.
        #
        # Each merged token concatenates its P constituent embeddings and
        # projects back to d_model, so the per-step positional / temporal
        # embeddings survive the merge rather than being averaged away.
        if self.temporal_patch > 1:
            self.patch_norm  = nn.LayerNorm(self.temporal_patch * d_model)
            self.patch_merge = nn.Linear(self.temporal_patch * d_model, d_model)
        else:
            self.patch_norm = self.patch_merge = None

        # ── v15: TELESCOPIC patch schedule (overrides temporal_patch) ─────
        # Resolution matched to forecast-relevance: coarse patches for the
        # distant past, RAW steps for the recent past (WaveNet/Pyraformer
        # logic applied to tokenization). Schedule string, oldest→newest,
        # "steps x patch" segments, e.g. "48x6,18x3,6x1" for W=72:
        #     oldest 8 h @ patch 6 → 8 tokens
        #     middle 3 h @ patch 3 → 6 tokens
        #     last   1 h @ raw     → 6 tokens      (20 temporal positions)
        # One merge layer per distinct patch size (>1). This is what lets the
        # input_context side-channel be deleted: full 10-min resolution of the
        # last hour lives in the main sequence.

    def _patchify(self, t: torch.Tensor) -> torch.Tensor:
        """
        (B, W, N, D) → (B, W/P, N, D), one uniform patch size.

        The v15 TELESCOPIC mode (per-segment patch sizes via --patch_schedule,
        e.g. "48x6,18x3,6x1") was removed in the v22 cleanup: no configuration
        ever selected it. git has the code.

        Station and batch axes are untouched, so station masking (which runs
        before this) is unaffected. Per-step positional/temporal embeddings
        survive inside the concatenation rather than being averaged away.
        """
        B, W, N, D = t.shape
        P = self.temporal_patch
        if P <= 1:
            return t
        assert W % P == 0, (
            f"temporal_patch={P} must divide the window W={W} exactly. "
            f"With W=72 the usable values are 1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72.")
        t = t.reshape(B, W // P, P, N, D)          # split the time axis
        t = t.permute(0, 1, 3, 2, 4)               # (B, W/P, N, P, D)
        t = t.reshape(B, W // P, N, P * D)         # concat the P embeddings
        return self.patch_merge(self.patch_norm(t))

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
        # Absent sensors are handled INSIDE var_proj: present sensors contribute
        # their projected value, absent ones a learned per-variable absent
        # embedding (var_absent_embedding). No pre-fill needed here — x carries
        # zeros at absent slots and var_proj never reads them (mask-gated).
        x_flat      = x.view(B * W, N, V)
        mask_flat   = x_mask.view(B * W, N, V)
        _static     = spatial if self.static_in_token else None
        var_tokens  = self.var_proj(x_flat, mask_flat, static=_static)
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

        # --- s: Step-index embedding (within-window relative position) ---
        # Encodes the INTEGER position 0..W-1 of each token in the input window.
        # Provides explicit ordering information that the absolute temporal
        # embedding does not cleanly expose at the 10-min step level.
        # Consistent with the decoder: step indices use the same Fourier basis
        # so "encoder step w" and "decoder query at delta_step w" are aligned.
        step_idx = torch.arange(W, device=x.device)                 # (W,)
        step_e   = self.step_emb(step_idx)                          # (W, d_model)
        step_e   = step_e.view(1, W, 1, self.d_model)               # (1, W, 1, d_model)

        # Sum five embeddings — all broadcast cleanly over (B, W, N, d_model)
        if self.static_in_token:
            # position and topography are already inside var_tokens as slots;
            # adding them again would double-count and restore the imbalance
            # this mode exists to remove.
            tokens = var_tokens + temp_emb + step_e
        else:
            tokens = var_tokens + pos_e + station_e + temp_emb + step_e

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
            visible_tokens  : (B, W, N_vis, d_model)   — unflattened; forward() flattens after transformer
            masked_indices  : (B, N_masked)             — masked station indices
            visible_indices : (B, N_vis)                — visible station indices
        """
        B, W, N, D = tokens.shape
        num_masked  = int(N * self.mask_ratio)
        num_visible = N - num_masked

        # Sample a different random mask per batch item.
        # WHICH stations are hidden is redrawn every forward pass — that is the
        # masking objective and it stays random.
        noise           = torch.rand(B, N, device=tokens.device)
        shuffle_idx     = torch.argsort(noise, dim=1)              # (B, N)
        visible_indices = shuffle_idx[:, num_masked:]              # (B, N_vis)
        masked_indices  = shuffle_idx[:, :num_masked]              # (B, N_masked)

        # ── Restore canonical station ORDER within each group ────────────────
        # argsort returns a permutation, so the two slices above came out in
        # random order — the sequence handed to the encoder was shuffled even
        # at mask_ratio 0, where nothing is dropped at all.
        #
        # Harmless for the query decoder (the encoder output is only a key/value
        # set, and attention over the station axis is permutation-equivariant —
        # spatial attention here is FULL over N, windowing applies to the
        # temporal axis only). But fatal for any positional readout: the v22
        # direct head does encoded.view(B, T, n_stations, d) and treats axis 2
        # as station index, so a shuffled sequence would train station j's
        # target against another station's tokens, redrawn every step. The only
        # learnable solution is a station-independent one — mean collapse, for a
        # reason that has nothing to do with the architecture.
        #
        # Sorting keeps the mask random while making the ORDER deterministic:
        # at mask_ratio 0 visible_indices is exactly arange(N), and at
        # mask_ratio > 0 it is a monotone subsequence of it.
        masked_indices,  _ = torch.sort(masked_indices,  dim=1)    # (B, N_masked)
        visible_indices, _ = torch.sort(visible_indices, dim=1)    # (B, N_vis)

        # Gather visible tokens across station dimension
        # visible_indices: (B, N_vis) → expand to (B, W, N_vis, D)
        vis_idx_exp = visible_indices \
            .unsqueeze(1) \
            .unsqueeze(-1) \
            .expand(B, W, num_visible, D)                          # (B, W, N_vis, D)

        visible_tokens = tokens.gather(2, vis_idx_exp)             # (B, W, N_vis, D)

        # NOTE: returned as (B, W, N_vis, D) — NOT flattened.
        # forward() flattens to (B, W*N_vis, D) after the transformer blocks so
        # that both the flat and factorised paths share the same decoder interface.
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

        # 2. Mask stations → visible_tokens: (B, W, N_vis, d_model)  [unflattened]
        visible_tokens, masked_idx, visible_idx = self._mask_stations(tokens)

        # 2b. Temporal patch merge — uniform or telescopic (v15).
        # Done AFTER masking so the station axis is already reduced — the merge
        # then costs P·d² per surviving token rather than per original station.
        #
        # v15 NOTE: the input_context side-channel has been REMOVED. Its only
        # purpose was to smuggle raw last-step information past uniform
        # patching; with the telescopic schedule the last hour is at native
        # 10-min resolution INSIDE the main sequence, so the decoder reads
        # recency through ordinary attention. (This also deletes the subsystem
        # behind the silent-eval bug and the zero-key attention issue.)
        visible_tokens = self._patchify(visible_tokens)
        B_sz, W_sz, N_vis, D = visible_tokens.shape

        # 3. Transformer blocks
        # ── Factorised path:    keep (B, W, N_vis, d) through FactorisedTransformerBlocks
        #    then flatten to (B, W*N_vis, d) for the decoder interface.
        # ── Full flat path:     flatten first to (B, W*N_vis, d) then run TransformerBlocks.
        # Both paths use gradient checkpointing when use_checkpoint=True.
        if self.factorised:
            h = visible_tokens                                     # (B, W, N_vis, d_model)
            for block in self.blocks:
                if self.use_checkpoint and torch.is_grad_enabled():
                    h = _cp_checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            h = h.reshape(B_sz, W_sz * N_vis, D)                  # (B, W*N_vis, d_model)
        else:
            h = visible_tokens.reshape(B_sz, W_sz * N_vis, D)     # (B, W*N_vis, d_model)
            for block in self.blocks:
                if self.use_checkpoint and torch.is_grad_enabled():
                    h = _cp_checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)

        h = self.norm(h)                                           # (B, W*N_vis, d_model)
        return h, masked_idx, visible_idx
