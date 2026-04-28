#!/usr/bin/env bash
# run_full.sh — full training run on all available years (2017–2021)
#
# Key differences from run_subset.sh:
#   • No --subset flag   → uses all training years (2017–2021)
#   • --window 72        → 12-hour input window
#   • --grad_checkpoint  → gradient checkpointing: ~66% less VRAM, ~33% slower
#   • --local_cache_dir  → default /tmp/station_mae_cache (tmpfs RAM disk).
#                          First run writes split .npy files there; workers
#                          mmap them directly from OS page cache — no IPC
#                          overhead for source data.  Change to '' to disable.
#
# Encoder architecture options (require --factorised_encoder):
#
#   --no_spatial_attn
#     Removes the spatial attention sub-layer from every encoder block.
#     Each station is encoded independently from its own temporal window;
#     cross-station reasoning is delegated entirely to the decoder.
#     Saves ~27% per encoder block — the single most impactful speed flag.
#     Recommended with --cross_attn_decoder.
#
#   --temporal_window N
#     Local windowed temporal attention: splits W timesteps into chunks of N.
#     Odd layers use a Swin-style half-window shift so tokens communicate
#     across chunk boundaries after two layers. W must be divisible by N.
#     At W=72, tw=6 gives 12 one-hour chunks — ~6% block speedup.
#     More impactful at larger W (see run_full_cloud.sh at W=144).
#
# If still OOM: reduce --batch_size to 4.
#
# Device is auto-selected: CUDA (GPU) → MPS (Apple Silicon) → CPU
# --amp enables fp16 AMP; safe to leave on everywhere (no-op on CPU).
#
# WandB: credentials are picked up from ~/.netrc after `wandb login`.
#        Omit --wandb_project to fall back to CSV logging only.
#
# Resuming an interrupted run:
#   ./run_full.sh   (Lightning restores from last.ckpt automatically)
#
# Usage:
#   chmod +x run_full.sh
#   ./run_full.sh

set -euo pipefail

# Reduces CUDA memory fragmentation — helps when many differently-sized
# tensors are allocated/freed during the attention passes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
SAVE_DIR="checkpoints/full_run"

# ── Optional encoder flags ────────────────────────────────────────────────────
# Uncomment to remove spatial attention (~27% faster per encoder block):
# SPATIAL=""
SPATIAL="--no_spatial_attn"

# Uncomment to enable local windowed temporal attention:
# W=72 / tw=6 → 12 one-hour chunks; ~6% block speedup at this window size.
# TEMPORAL_WINDOW="--temporal_window 6"
TEMPORAL_WINDOW=""
# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --window           72 \
    --max_delta        0 \
    --num_delta        1 \
    --d_model          128 \
    --enc_layers       4 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.1 \
    --batch_size       8 \
    --num_workers      4 \
    --epochs           50 \
    --lr               1e-4 \
    --warmup_epochs    5 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --patience         15 \
    --save_every       2 \
    --amp \
    --grad_checkpoint \
    --factorised_encoder \
    --cross_attn_decoder \
    $SPATIAL \
    $TEMPORAL_WINDOW \
    --wandb_project    station-mae-local \
    --wandb_run_name   full-run \
    --save_dir         "$SAVE_DIR"
