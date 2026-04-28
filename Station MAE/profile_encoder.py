"""
profile_encoder.py

Measures the wall-clock time of each sub-operation inside a
FactorisedTransformerBlock under realistic tensor shapes.

Operations timed
----------------
  1. Temporal attention — full        (B·N_vis, W, D)
  2. Temporal attention — windowed tw (B·N_vis·n_chunks, tw, D)
  3. Spatial attention                (B·W, N_vis, D)
  4. FFN                              (B, W, N_vis, D)

Run with:
    python profile_encoder.py
    python profile_encoder.py --window 144 --tw 6
    python profile_encoder.py --window 288 --tw 6 --n_vis 100

Results print as a table sorted by time so you can see the bottleneck.
"""

import argparse
import time
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn, n_runs: int = 200, warmup: int = 20) -> float:
    """Return mean wall-clock time in ms over n_runs calls."""
    for _ in range(warmup):
        fn()
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    _sync()
    return (time.perf_counter() - t0) / n_runs * 1_000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Profile encoder sub-operations")
    p.add_argument("--batch",    type=int,   default=8,   help="Batch size B")
    p.add_argument("--window",   type=int,   default=72,  help="Temporal steps W")
    p.add_argument("--n_vis",    type=int,   default=50,  help="Visible stations N_vis")
    p.add_argument("--d_model",  type=int,   default=128, help="Model dimension D")
    p.add_argument("--heads",    type=int,   default=4,   help="Attention heads")
    p.add_argument("--mlp",      type=float, default=4.0, help="FFN expansion ratio")
    p.add_argument("--tw",       type=int,   default=6,   help="Temporal window size")
    p.add_argument("--n_runs",   type=int,   default=200)
    p.add_argument("--warmup",   type=int,   default=20)
    args = p.parse_args()

    B, W, N, D = args.batch, args.window, args.n_vis, args.d_model
    H, mlp, tw = args.heads, args.mlp, args.tw
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Station-MAE Encoder Sub-Operation Profiler")
    print(f"{'='*60}")
    print(f"  Device      : {device.upper()}")
    print(f"  B={B}  W={W}  N_vis={N}  D={D}  heads={H}  mlp={mlp}")
    print(f"  Temporal window : tw={tw}  →  {W//tw} chunks of {tw} steps")
    print(f"  Runs: {args.n_runs}  Warmup: {args.warmup}")
    print(f"{'='*60}\n")

    assert W % tw == 0, f"tw={tw} must divide W={W}"

    # ── Modules ─────────────────────────────────────────────────────────────
    mha_t = nn.MultiheadAttention(D, H, batch_first=True).to(device).eval()
    mha_s = nn.MultiheadAttention(D, H, batch_first=True).to(device).eval()
    ffn   = nn.Sequential(
        nn.LayerNorm(D),
        nn.Linear(D, int(D * mlp)),
        nn.GELU(),
        nn.Linear(int(D * mlp), D),
    ).to(device).eval()
    norm_t = nn.LayerNorm(D).to(device).eval()
    norm_s = nn.LayerNorm(D).to(device).eval()

    # ── Input tensors ────────────────────────────────────────────────────────
    # Temporal: (B·N, W, D)
    x_temp  = torch.randn(B * N, W, D, device=device)
    # Windowed: (B·N·n_chunks, tw, D)
    x_wind  = torch.randn(B * N * (W // tw), tw, D, device=device)
    # Spatial: (B·W, N, D)
    x_spat  = torch.randn(B * W, N, D, device=device)
    # FFN: (B, W, N, D)  — same total tokens, different shape
    x_ffn   = torch.randn(B, W, N, D, device=device)

    # ── Timed operations ─────────────────────────────────────────────────────
    with torch.no_grad():

        def _temp_full():
            n = norm_t(x_temp)
            mha_t(n, n, n, need_weights=False)

        def _temp_wind():
            n = norm_t(x_wind)
            mha_t(n, n, n, need_weights=False)

        def _temp_wind_reshape():
            # Includes the reshape overhead (realistic cost)
            xt = x_temp  # (B·N, W, D)
            # shift not timed — negligible roll
            xt_c = xt.reshape(B * N * (W // tw), tw, D)
            n = norm_t(xt_c)
            out, _ = mha_t(n, n, n, need_weights=False)
            out.reshape(B * N, W, D)

        def _spatial():
            n = norm_s(x_spat)
            mha_s(n, n, n, need_weights=False)

        def _ffn():
            ffn(x_ffn)

        results = {}
        print("  Timing... (this may take a few seconds)")

        results["Temporal attn  FULL       (B·N, W, D)"]        = time_ms(_temp_full,        args.n_runs, args.warmup)
        results["Temporal attn  WINDOWED   (B·N·chunks, tw, D)"] = time_ms(_temp_wind,        args.n_runs, args.warmup)
        results["Temporal attn  WINDOWED + reshape overhead"]    = time_ms(_temp_wind_reshape, args.n_runs, args.warmup)
        results["Spatial  attn  (B·W, N, D)"]                   = time_ms(_spatial,           args.n_runs, args.warmup)
        results["FFN            (B, W, N, D) → includes norm"]  = time_ms(_ffn,               args.n_runs, args.warmup)

    # ── Table ────────────────────────────────────────────────────────────────
    max_t  = max(results.values())
    width  = max(len(k) for k in results) + 2

    print(f"\n  {'Operation':<{width}}  {'ms / call':>10}  {'rel to max':>10}  Bar")
    print(f"  {'-'*width}  {'-'*10}  {'-'*10}  {'-'*30}")
    for label, ms in results.items():
        bar_len = int(ms / max_t * 30)
        bar = "█" * bar_len
        print(f"  {label:<{width}}  {ms:>9.3f}  {ms/max_t:>9.1%}  {bar}")

    print()

    # ── Derived estimates for a full block ───────────────────────────────────
    t_temp_full = results["Temporal attn  FULL       (B·N, W, D)"]
    t_temp_wind = results["Temporal attn  WINDOWED + reshape overhead"]
    t_spat      = results["Spatial  attn  (B·W, N, D)"]
    t_ffn       = results["FFN            (B, W, N, D) → includes norm"]

    block_full = t_temp_full + t_spat + t_ffn
    block_wind = t_temp_wind + t_spat + t_ffn

    print(f"  {'─'*60}")
    print(f"  Estimated FactorisedTransformerBlock cost per layer:")
    print(f"    Full  temporal + spatial + FFN  : {block_full:7.3f} ms")
    print(f"    Windowed(tw={tw}) + spatial + FFN: {block_wind:7.3f} ms")
    print(f"    Speedup from windowed temporal  : {block_full/block_wind:.2f}×")
    print(f"    Speedup without spatial attn    : {block_full/(t_temp_full+t_ffn):.2f}×  (temporal-only)")
    print(f"  {'─'*60}\n")

    # ── Token count and attention pair summary ────────────────────────────────
    print(f"  Attention pairs per call:")
    print(f"    Temporal full     : {B*N * W*W:>12,}   ({B*N} seqs × {W}² = {W*W} pairs each)")
    print(f"    Temporal windowed : {B*N*(W//tw) * tw*tw:>12,}   ({B*N*(W//tw)} seqs × {tw}² = {tw*tw} pairs each)")
    print(f"    Spatial           : {B*W * N*N:>12,}   ({B*W} seqs × {N}² = {N*N} pairs each)")
    print(f"    Windowed speedup  : {(B*N*W*W) / (B*N*(W//tw)*tw*tw):.1f}×  (attention pairs only)\n")


if __name__ == "__main__":
    main()
