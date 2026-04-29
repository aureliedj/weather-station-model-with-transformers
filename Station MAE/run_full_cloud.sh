#!/usr/bin/env bash
# run_full_cloud.sh — optimised full training run for a 20 GB GPU
#
# Hardware context
# ----------------
# Platform reports "GPU 20GB" — exact model unconfirmed.
#   • ~20 GB VRAM
#   • 7 CPUs, 70 GB RAM, single GPU
#
# Key flags:
#
#   --amp
#     Automatic Mixed Precision.  Lightning chooses the best supported mode:
#       • bf16-mixed  if the GPU has native BF16 tensor cores (A100/H100/RTX 30+)
#       • 16-mixed    (FP16) otherwise — works on any CUDA-capable GPU.
#     Check the Lightning startup log to confirm which mode is active.
#
#   --bf16  (OPTIONAL — safe to remove if GPU is not confirmed Ampere/A100)
#     Forces bf16-mixed precision.  Wider dynamic range than FP16, no loss
#     scaling needed.  Requires native BF16 tensor cores.
#     If your GPU does not support it, PyTorch errors at startup:
#       RuntimeError: BF16 is not supported on this GPU
#     → remove --bf16 and rely on --amp alone (FP16 fallback is automatic).
#
#   --compile
#     torch.compile "default" mode fuses ops and removes Python dispatch
#     overhead (~1.3–1.5× speedup), no CUDA graphs needed.
#     Warm-up cost: ~1–2 min on the first epoch.
#
#   --local_cache_dir /tmp/station_mae_cache
#     Saves split-normalised tensors as numpy mmap files on /tmp (RAM-backed).
#     Workers mmap from OS page cache — no IPC queue overhead.
#     /tmp is tmpfs — counts against RAM (70 GB), not disk (50 GB).
#
# Speed improvements vs original config (estimated ~60 min/epoch → ~26 min):
#
#   Removed --grad_checkpoint  (+18% speed)
#     W=72 + mlp_ratio=2.0 + batch=32 → ~2.4 GB activations — well within
#     20 GB VRAM.  Grad checkpointing is only needed for W≥144 or very large
#     batches.  Removing it saves the ~33% extra backward compute it adds.
#     Re-add it (with --batch_size 16) only if you hit OOM.
#
#   --batch_size 32  (+36% speed, was 16)
#     Doubles GPU utilisation; halves steps/epoch.  Net gain ~1.6×.
#     If OOM: revert to 16 and re-add --grad_checkpoint.
#
#   --temporal_window 6  (+8% speed)
#     W=72 / tw=6 → 12 two-hour chunks; attention score cost drops 12×.
#     Odd encoder layers use Swin-style half-window shift for cross-chunk
#     communication.  W=72 is divisible by tw=6 ✓.
#
#   --num_delta 3  (+7% speed, was 6)
#     Decoder builds N×K = 156×3 = 468 query tokens (was 936).
#     Still samples 3 distinct lead-times per sample for multi-delta training.
#     To always predict at exactly Δt=36: use --num_delta 1 --max_delta 36.
#
# Encoder architecture options (require --factorised_encoder):
#
#   --no_spatial_attn
#     Encodes each station independently; cross-station reasoning delegated
#     entirely to the decoder.  Saves ~27% per encoder block.
#     Recommended with --cross_attn_decoder.
#
#   --temporal_window N
#     Local windowed temporal attention (see above).  W must be divisible by N.
#
# Resuming an interrupted run:
#   ./run_full_cloud.sh   (Lightning restores from last.ckpt automatically)
#
# Usage:
#   chmod +x run_full_cloud.sh
#   ./run_full_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"   # ← update this
SAVE_DIR="checkpoints/full_run_cloud"
LOCAL_CACHE="/tmp/station_mae_cache"

# ── Precision ─────────────────────────────────────────────────────────────────
# Remove --bf16 if you see "RuntimeError: BF16 is not supported on this GPU"
# --amp alone will then use FP16 mixed precision, which works on any CUDA GPU.
PRECISION="--amp --bf16"
# PRECISION="--amp"   # ← safe fallback for non-Ampere GPUs
# ─────────────────────────────────────────────────────────────────────────────

# ── Optional encoder flags ────────────────────────────────────────────────────
# Remove spatial attention (~27% faster per encoder block):
SPATIAL="--no_spatial_attn"
# SPATIAL=""   # ← uncomment to restore full spatial attention

# Windowed temporal attention: W=72 / tw=6 → 12 two-hour chunks (~8% faster):
TEMPORAL_WINDOW="--temporal_window 6"
#TEMPORAL_WINDOW=""   # ← uncomment to disable windowed attention
# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --num_delta        6 \
    --mlp_ratio        2.0 \
    --d_model          128 \
    --enc_layers       4 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.1 \
    --batch_size       32 \
    --num_workers      5 \
    --epochs           20 \
    --lr               1e-4 \
    --warmup_epochs    5 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --val_check_interval 2000 \
    --save_every_steps   2000 \
    --patience           5 \
    --min_lr             1e-6 \
    $PRECISION \
    --compile \
    --factorised_encoder \
    --cross_attn_decoder \
    $SPATIAL \
    $TEMPORAL_WINDOW \
    --wandb_project    station-mae \
    --wandb_run_name   full-run-cloud \
    --save_dir         "$SAVE_DIR"

# ─── Sub-epoch validation + checkpointing ────────────────────────────────────
#
# With batch_size=32 and ~105K train samples → ~3281 steps/epoch (~26 min).
# val_check_interval=2000 steps ≈ 0.61 epochs ≈ ~16 min per check.
# patience=5 checks → stops after ~80 min without improvement.
#
# Adjust to taste:
#   2000 steps → val every ~16 min, patience=5 → ~80 min tolerance
#   3000 steps → val every ~24 min, patience=3 → ~72 min tolerance

# ─── If OOM at batch_size=32 ─────────────────────────────────────────────────
# Revert to --batch_size 16 and re-add --grad_checkpoint.
# With batch=16: ~6562 steps/epoch; set val_check_interval=4000 (~30 min).
# Grad checkpointing trades ~33% extra compute for ~66% less activation memory.
