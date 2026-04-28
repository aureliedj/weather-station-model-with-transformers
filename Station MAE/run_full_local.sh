#!/usr/bin/env bash
# run_full_local.sh — full training run on all years, optimised for local machine
#
# Device is auto-selected: CUDA (GPU) → MPS (Apple Silicon) → CPU
# --amp enables fp16 AMP; safe to leave on everywhere (no-op on CPU).
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
#     At W=36, tw=6 gives 6 chunks — modest ~6% block speedup.
#     More impactful at larger W (see run_full_cloud.sh).
#
# Resuming an interrupted run:
#   ./run_full_local.sh   (Lightning restores from last.ckpt automatically)
#
# Usage:
#   chmod +x run_full_local.sh
#   ./run_full_local.sh

set -euo pipefail

DATA_ROOT="/Users/aureliedejong/Documents/ETH/_DAS Project/PeakWeatherDataset"
SAVE_DIR="checkpoints/full_run_local"

# ── Optional encoder flags ────────────────────────────────────────────────────
# Uncomment to remove spatial attention (~27% faster per encoder block):
# SPATIAL=""
SPATIAL="--no_spatial_attn"

# Uncomment to enable local windowed temporal attention:
# W=36 / tw=6 → 6 chunks; modest speedup at this window size.
# TEMPORAL_WINDOW="--temporal_window 6"
TEMPORAL_WINDOW=""
# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --window           36 \
    --max_delta        36 \
    --num_delta        6 \
    --d_model          128 \
    --enc_layers       4 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.1 \
    --batch_size       16 \
    --num_workers      4 \
    --epochs           50 \
    --lr               1e-4 \
    --warmup_epochs    5 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --patience         15 \
    --save_every       2 \
    --amp \
    --factorised_encoder \
    --cross_attn_decoder \
    $SPATIAL \
    $TEMPORAL_WINDOW \
    --wandb_project    station-mae-local \
    --wandb_run_name   full-run-local \
    --save_dir         "$SAVE_DIR"
