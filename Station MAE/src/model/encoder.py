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
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        # Floor to 0/1; divide by keep_prob to maintain expected value = 1
        random_tensor = torch.floor(random_tensor + keep_prob) / keep_prob
        return x * random_tensor


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
         Can be disabled via ``spatial_attn=False`` to make the encoder
         purely temporal (station-independent).  Spatial reasoning is then
         left entirely to the decoder's cross- or self-attention.

      3. Shared FFN applied element-wise over the last dimension.

    Complexity with both sub-layers: O(N·W² + W·N²)  vs  O((W·N)²) for flat.
    Complexity with spatial_attn=False: O(N·W²) — purely temporal.
    Complexity with temporal_window=tw: O(N·W·tw + W·N²) — local temporal.
    At W=288, N_vis≈100: ~8.3M vs ~830M attention pairs — roughly 100× cheaper.

    Args:
        d_model:         Model dimension.
        num_heads:       Attention heads (shared across both sub-layers).
        mlp_ratio:       FFN hidden-dim ratio (default 4.0).
        dropout:         Dropout in attention and FFN (default 0.1).
        spatial_attn:    If False, skip the spatial sub-layer entirely.
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
        # Only instantiated when spatial_attn=True to avoid unused parameters.
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
        # Skipped entirely when spatial_attn=False.
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
# Joint spatiotemporal attention block (with temporal RoPE)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE application."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    x:   torch.Tensor,   # (B, H, L, head_dim)
    cos: torch.Tensor,   # (L, head_dim)
    sin: torch.Tensor,   # (L, head_dim)
) -> torch.Tensor:
    """Apply rotary position embedding to Q or K."""
    # Broadcast over batch (B) and heads (H)
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, L, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)   # (1, 1, L, head_dim)
    return x * cos + _rotate_half(x) * sin


class JointSpatioTemporalBlock(nn.Module):
    """
    Joint spatiotemporal self-attention block operating on (B, W, N, d_model).

    Unlike the factorised block, which attends *separately* over time then space,
    this block flattens all W×N tokens into a single sequence of length L = W·N and
    runs a single multi-head self-attention pass over all of them simultaneously.
    This means every station–timestep token can directly attend to every other
    station at every other timestep in a single layer — capturing cross-dimensional
    dependencies that factorised blocks can only approximate over multiple layers.

    **Memory / compute trade-off**
    Full attention over L = W·N tokens has O(L²) = O((W·N)²) cost.  For the default
    config (W=72, N_vis≈75), L ≈ 5400.  Using Flash Attention (via
    ``F.scaled_dot_product_attention``) reduces *memory* to O(L) — no materialised
    attention matrix — so typical GPU VRAM is the limiting resource rather than
    pure FLOPS.  Gradient checkpointing on these blocks is strongly recommended.

    **Temporal RoPE**
    Position encoding is injected via 1D Rotary Position Embeddings (RoPE,
    Su et al. 2021) applied only along the *temporal* axis.  Every token at
    timestep w receives the same rotational encoding regardless of which station it
    belongs to.  RoPE rotates Q and K vectors so that attention scores naturally
    encode relative temporal distance — without adding an explicit additive position
    tensor and without requiring an attention mask.  This makes the block compatible
    with ``F.scaled_dot_product_attention`` (Flash Attention path).

    Spatial relationships are handled implicitly through the station embeddings
    (PositionalEmbedding + StationEmbedding) that are summed into the token before
    entering the encoder, so no separate spatial position encoding is added here.

    Args:
        d_model:      Model / token dimension.
        num_heads:    Number of attention heads.  head_dim = d_model // num_heads.
        mlp_ratio:    FFN hidden-dim expansion ratio (default 4.0).
        dropout:      Dropout in FFN and attention (default 0.1).
        drop_path:    Stochastic depth drop probability (default 0.0).
        max_time:     Upper bound on W for pre-computing RoPE frequencies (default 512).
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.1,
        drop_path: float = 0.0,
        max_time:  int   = 512,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads    = num_heads
        self.head_dim     = d_model // num_heads
        self.attn_dropout = dropout

        # ── Pre-norm ────────────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ── Manual Q / K / V projections ────────────────────────────────────
        # We bypass nn.MultiheadAttention so we can apply RoPE between the
        # projection and the attention kernel — MHA's forward doesn't expose
        # Q/K between those two steps.
        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # ── RoPE frequencies (buffer — moves to device automatically) ───────
        # inv_freq[i] = 1 / 10000^(2i / head_dim)
        # Shape: (head_dim // 2,)
        inv_freq = 1.0 / (
            10000.0 ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

        # ── FFN ──────────────────────────────────────────────────────────────
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

        # ── Stochastic depth ─────────────────────────────────────────────────
        self.drop_path = DropPath(drop_path)

    def _rope_for_window(
        self, W: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-compute cos/sin RoPE tensors for a sequence of length W*N
        where N tokens share the same temporal index w.

        Since the index order in the flattened sequence is
            [w=0,n=0], [w=0,n=1], …, [w=0,n=N-1],
            [w=1,n=0], …
        the temporal index for position i is i // N.  However because N is
        not known at __init__ time, we pass W here and let the caller expand.

        Returns
        -------
        cos, sin : (W, head_dim) tensors — one entry per timestep.
        Caller must unsqueeze and expand over N.
        """
        t = torch.arange(W, device=device, dtype=torch.float32)  # (W,)
        freqs = torch.outer(t, self.inv_freq)                      # (W, head_dim//2)
        emb   = torch.cat([freqs, freqs], dim=-1)                  # (W, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, N, d_model)
        Returns:
            (B, W, N, d_model)
        """
        B, W, N, D = x.shape
        L = W * N

        # Flatten spatial and temporal into a single sequence
        h = x.reshape(B, L, D)

        # ── Attention with pre-norm ──────────────────────────────────────────
        h_n = self.norm1(h)

        # Project to Q, K, V and split into heads
        # (B, L, D) → (B, L, H, head_dim) → (B, H, L, head_dim)
        Q = self.q_proj(h_n).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h_n).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h_n).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Build temporal RoPE: cos/sin shape (W, head_dim), then tile over N
        cos_w, sin_w = self._rope_for_window(W, x.device, x.dtype)  # (W, head_dim)
        # Each timestep w covers N consecutive positions → repeat_interleave
        cos_l = cos_w.repeat_interleave(N, dim=0)  # (L, head_dim)
        sin_l = sin_w.repeat_interleave(N, dim=0)  # (L, head_dim)

        Q = _apply_rope(Q, cos_l, sin_l)
        K = _apply_rope(K, cos_l, sin_l)

        # Flash Attention via F.scaled_dot_product_attention
        # No explicit attention mask needed — RoPE encodes relative temporal distance.
        # Dropout is applied inside the kernel during training.
        dp = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)
        # (B, H, L, head_dim) → (B, L, H*head_dim) = (B, L, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.out_proj(attn_out)

        h = h + self.drop_path(attn_out)

        # ── FFN with pre-norm ────────────────────────────────────────────────
        h = h + self.drop_path(self.ffn(self.norm2(h)))

        return h.reshape(B, W, N, D)


# ---------------------------------------------------------------------------
# Windowed flat attention block
# ---------------------------------------------------------------------------

class WindowedFlatTransformerBlock(nn.Module):
    _warned_shift = False

    """
    Flat self-attention restricted to local temporal windows over (B, W, N, d).

    Instead of attending over all W×N tokens at once (O((W·N)²) pairs), the
    W timesteps are split into non-overlapping chunks of size ``temporal_window``
    (tw).  All N station tokens within each chunk attend to each other jointly —
    preserving the full cross-station context that factorised attention splits
    into two separate sub-layers.

    Chunk sequence length:  L_chunk = tw × N_vis
    Attention pairs / chunk: L_chunk² = (tw × N_vis)²
    Number of chunks:        W / tw

    Example  W=72, tw=6, N_vis=65:
        L_chunk = 390  (≈ BERT)  → 152K pairs vs 21.9M for full flat  (144× cheaper)

    Cross-chunk communication uses a Swin-style half-window circular shift:
    even layers use no shift; odd layers shift the temporal axis by tw//2 so
    that chunk boundaries alternate, letting information propagate across them
    after two successive layers.

    Args:
        d_model:          Model dimension.
        num_heads:        Attention heads.
        mlp_ratio:        FFN hidden-dim ratio (default 4.0).
        dropout:          Dropout in FFN and attention (default 0.1).
        temporal_window:  Chunk size in timesteps (tw).  W must be divisible by tw.
        shift:            If True apply a tw//2 circular shift before chunking
                          (Swin-style).  Alternate layers should alternate shift.
        drop_path:        Stochastic depth drop probability (default 0.0).
    """

    def __init__(
        self,
        d_model:          int,
        num_heads:         int,
        mlp_ratio:         float = 4.0,
        dropout:           float = 0.1,
        temporal_window:   int   = 6,
        shift:             bool  = False,
        drop_path:         float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert temporal_window > 0,      "temporal_window must be > 0"

        self.temporal_window = temporal_window
        self.shift           = shift
        self.num_heads       = num_heads
        self.head_dim        = d_model // num_heads
        self.attn_dropout    = dropout

        # ── Pre-norm ────────────────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ── Manual Q/K/V projections for Flash Attention compatibility ───────────
        # (F.scaled_dot_product_attention requires explicit tensors)
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # ── FFN ──────────────────────────────────────────────────────────────────
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

        # ── Stochastic depth ─────────────────────────────────────────────────────
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, N, d_model)
        Returns:
            (B, W, N, d_model)
        """
        B, W, N, D = x.shape
        tw = self.temporal_window
        assert W % tw == 0, (
            f"temporal_window={tw} must divide W={W} exactly. "
            f"Choose a value that divides your --window size."
        )
        num_chunks = W // tw
        shift_size = tw // 2 if self.shift else 0

        # ── Half-window circular shift along the temporal axis ───────────────────
        # WARNING — this is NOT a faithful Swin shift. Real Swin follows the roll
        # with an attention mask that partitions the straddling window, so tokens
        # which wrapped around cannot attend to each other. There is no such mask
        # below, so in the LAST chunk the newest timesteps attend to the OLDEST
        # ones as if they were adjacent: at W=72, tw=6 the token at t=71 attends
        # to t=0, twelve hours earlier.
        #
        # Measured effect (tw=6, 8 layers): the apparent temporal receptive field
        # of the newest token grows from 24 to 48 steps, but that extra 4 h is a
        # spurious wrap-around shortcut at the window seam, not genuine locality.
        # v9 / v11 / v12 were all trained with this.
        #
        # Not fixed because temporal patching retires this path: at P=6 the
        # sequence is 12 tokens and FULL attention is ~3x cheaper than this
        # windowed version while covering the whole window from layer 1.
        # If you revive windowed attention (e.g. for a 24 h input window), add
        # the Swin mask first.
        if shift_size > 0:
            if not WindowedFlatTransformerBlock._warned_shift:
                warnings.warn(
                    "WindowedFlatTransformerBlock: circular temporal shift is applied "
                    "WITHOUT a Swin attention mask, so the newest timesteps attend to "
                    "the oldest across the window seam. Prefer temporal patching "
                    "(--temporal_patch) with full attention. See the comment in "
                    "encoder.py for the measured effect.",
                    RuntimeWarning, stacklevel=2,
                )
                WindowedFlatTransformerBlock._warned_shift = True
            x = torch.roll(x, shifts=-shift_size, dims=1)

        # ── Chunk: (B, W, N, D) → (B·chunks, tw·N, D) ──────────────────────────
        # Reshape temporal dimension into (num_chunks, tw), then flatten tw and N.
        h = x.reshape(B * num_chunks, tw * N, D)

        # ── Flat self-attention within each chunk (Flash Attention) ──────────────
        h_n = self.norm1(h)
        H, hd, L = self.num_heads, self.head_dim, tw * N
        BnC = B * num_chunks

        Q = self.q_proj(h_n).reshape(BnC, L, H, hd).transpose(1, 2)   # (BnC, H, L, hd)
        K = self.k_proj(h_n).reshape(BnC, L, H, hd).transpose(1, 2)
        V = self.v_proj(h_n).reshape(BnC, L, H, hd).transpose(1, 2)

        dp = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)
        attn_out = attn_out.transpose(1, 2).reshape(BnC, L, D)
        attn_out = self.out_proj(attn_out)

        h = h + self.drop_path(attn_out)

        # ── FFN ──────────────────────────────────────────────────────────────────
        h = h + self.drop_path(self.ffn(self.norm2(h)))

        # ── Un-chunk: (B·chunks, tw·N, D) → (B, W, N, D) ───────────────────────
        x = h.reshape(B, W, N, D)

        # ── Undo temporal shift ──────────────────────────────────────────────────
        if shift_size > 0:
            x = torch.roll(x, shifts=shift_size, dims=1)

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
                              Ignored when joint=True.
        spatial_attn:         Only used when factorised=True.  If False, the spatial
                              sub-layer is removed from every FactorisedTransformerBlock —
                              each station is encoded independently from its own temporal
                              window with no cross-station mixing in the encoder.  Reduces
                              encoder cost from O(N·W²+W·N²) to O(N·W²).  All spatial
                              reasoning is then delegated to the decoder.
        temporal_window:      Only used when factorised=True.  Local attention window
                              size in timesteps (0 = full attention over all W steps).
                              W must be exactly divisible by temporal_window.
                              Odd-indexed blocks automatically use a Swin-style
                              half-window shift so tokens can communicate across
                              chunk boundaries after two layers.
                              Example: W=72, temporal_window=6 → 12 one-hour chunks
                              at 10-min resolution, 12× cheaper temporal attention.
        joint:                If True, use JointSpatioTemporalBlock — full attention over
                              all W×N tokens simultaneously, with temporal RoPE injected
                              into Q and K.  Captures cross-dimensional interactions in a
                              single layer at the cost of O((W·N)²) FLOPs; Flash Attention
                              keeps VRAM at O(W·N) so the memory is manageable.
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
        patch_schedule:       "str | None" = None,   # v15 telescopic, e.g. "48x6,18x3,6x1"
        drop_path_rate:       float = 0.0,
        joint:                bool  = False,
        step_emb:             "nn.Module | None" = None,
    ):
        super().__init__()

        self.d_model         = d_model
        self.mask_ratio      = mask_ratio
        self.temporal_patch  = int(temporal_patch)
        self.use_checkpoint  = use_checkpoint
        self.factorised      = factorised and not joint
        self.joint           = joint
        # Windowed flat: full cross-station attention within temporal chunks.
        # Active when temporal_window > 0 and neither joint nor factorised.
        self.windowed_flat   = (temporal_window > 0) and not joint and not factorised

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
        self.var_proj     = VariableProjection(num_vars=num_vars, d_model=d_model,
                                              value_embedding=value_embedding,
                                              wind_pair=wind_pair)
        self.pos_emb      = PositionalEmbedding(d_model=d_model, fourier_dim=position_fourier_dim,
                                                dropout=dropout)
        self.station_emb  = StationEmbedding(d_model=d_model, input_dim=station_char_dim,
                                             dropout=dropout)
        self.temporal_emb = TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim,
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
        # FactorisedTransformerBlock (axial attention over W×N grid) is used when
        # factorised=True; otherwise fall back to flat TransformerBlock for
        # backward compatibility with existing checkpoints.
        # spatial_attn=False removes the spatial sub-layer (station-independent encoder).
        # temporal_window > 0 enables local windowed temporal attention; odd-indexed
        # blocks use a half-window Swin shift so cross-chunk communication emerges
        # after two layers.  Ignored when factorised=False.
        #
        # Stochastic depth: drop_path_rate is the maximum drop probability (applied
        # to the deepest layer).  Rates increase linearly from 0.0 (layer 0) to
        # drop_path_rate (layer num_layers-1), following the standard recipe from
        # Huang et al. (2016) and used in DeiT / Swin Transformer.
        dp_rates = [
            drop_path_rate * i / max(num_layers - 1, 1)
            for i in range(num_layers)
        ]
        if joint:
            # Joint spatiotemporal attention with temporal RoPE.
            # Full W×N self-attention per block; Flash Attention keeps VRAM linear.
            self.blocks = nn.ModuleList([
                JointSpatioTemporalBlock(
                    d_model, num_heads, mlp_ratio, dropout,
                    drop_path=dp_rates[i],
                )
                for i in range(num_layers)
            ])
        elif factorised:
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
        elif temporal_window > 0:
            # Windowed flat: full cross-station attention within temporal chunks.
            # Each chunk contains tw × N_vis tokens; attention is O((tw·N)²) per
            # chunk vs O((W·N)²) for full flat.  Swin shift on odd layers bridges
            # chunk boundaries after two layers.
            self.blocks = nn.ModuleList([
                WindowedFlatTransformerBlock(
                    d_model, num_heads, mlp_ratio, dropout,
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
        self.patch_schedule: "list[tuple[int, int]] | None" = None
        if patch_schedule:
            segs = []
            for part in str(patch_schedule).split(","):
                n, p = part.lower().split("x")
                segs.append((int(n), int(p)))
            for n, p in segs:
                assert n % p == 0, f"segment {n}x{p}: patch must divide steps"
            self.patch_schedule = segs
            self.sched_merges = nn.ModuleDict({
                f"p{p}": nn.Sequential(nn.LayerNorm(p * d_model),
                                       nn.Linear(p * d_model, d_model))
                for p in sorted({p for _, p in segs if p > 1})
            })

    def _merge_segment(self, t: torch.Tensor, P: int) -> torch.Tensor:
        """(B, n, N, D) → (B, n/P, N, D) with the merge layer for patch size P."""
        if P <= 1:
            return t
        B, n, N, D = t.shape
        t = t.reshape(B, n // P, P, N, D).permute(0, 1, 3, 2, 4)
        t = t.reshape(B, n // P, N, P * D)
        return self.sched_merges[f"p{P}"](t)

    def _patchify(self, t: torch.Tensor) -> torch.Tensor:
        """
        Uniform mode:     (B, W, N, D) → (B, W/P, N, D), one patch size.
        Telescopic mode:  (B, W, N, D) → (B, T, N, D), per-segment patch sizes
                          (patch_schedule; T = Σ nᵢ/pᵢ, e.g. 20 for W=72).

        Station and batch axes are untouched, so station masking (which runs
        before this) is unaffected. Per-step positional/temporal embeddings
        survive inside the concatenation rather than being averaged away.
        """
        B, W, N, D = t.shape
        if self.patch_schedule is not None:
            assert W == sum(n for n, _ in self.patch_schedule), (
                f"patch_schedule covers {sum(n for n, _ in self.patch_schedule)} "
                f"steps but window W={W}")
            out, i = [], 0
            for n, p in self.patch_schedule:               # oldest → newest
                out.append(self._merge_segment(t[:, i:i + n], p))
                i += n
            return torch.cat(out, dim=1)                   # (B, T, N, D)
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
        tokens = var_tokens + pos_e + station_e + temp_emb + step_e  # (B, W, N, d_model)

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
        # ── Joint path:         keep (B, W, N_vis, d) through JointSpatioTemporalBlocks
        #    (full W×N attention with temporal RoPE), then flatten to (B, W*N_vis, d).
        # ── Factorised path:    keep (B, W, N_vis, d) through FactorisedTransformerBlocks
        #    then flatten to (B, W*N_vis, d) for the decoder interface.
        # ── Windowed flat path: keep (B, W, N_vis, d) through WindowedFlatTransformerBlocks
        #    (full attention within tw×N_vis chunks), then flatten.
        # ── Full flat path:     flatten first to (B, W*N_vis, d) then run TransformerBlocks.
        # All paths use gradient checkpointing when use_checkpoint=True.
        if self.joint or self.factorised or self.windowed_flat:
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
