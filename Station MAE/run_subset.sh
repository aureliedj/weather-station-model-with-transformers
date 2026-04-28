#!/usr/bin/env bash
# run_subset.sh — pipeline sanity check on 2 years of data (light local config)
#
# Device is auto-selected by Lightning: CUDA (GPU) → MPS (Apple Silicon) → CPU
# --amp enables fp16 AMP only when CUDA/MPS is active; safe to leave on everywhere.
#
# Encoder architecture options (require --factorised_encoder):
#
#   --no_spatial_attn
#     Removes the spatial attention sub-layer from every encoder block.
#     Each station is encoded independently from its own temporal window;
#     cross-station reasoning is delegated entirely to the decoder.
#     Saves ~27% per encoder block. Recommended with --cross_attn_decoder.
#
#   --temporal_window N
#     Local windowed temporal attention: splits W timesteps into chunks of N.
#     Odd layers use a Swin-style half-window shift so tokens communicate
#     across chunk boundaries after two layers. W must be divisible by N.
#     At W=12, benefit is negligible — more useful at W=72+.
#
# WandB: set --wandb_project to stream metrics to your WandB dashboard.
#        Omit the flag entirely to log to CSV only (no WandB account needed).
#
# Usage:
#   chmod +x run_subset.sh
#   ./run_subset.sh

set -euo pipefail

DATA_ROOT="/Users/aureliedejong/Documents/ETH/_DAS Project/PeakWeatherDataset"
SAVE_DIR="checkpoints/subset_run"

# ── Optional encoder flags ────────────────────────────────────────────────────
# Uncomment to remove spatial attention (~27% faster per encoder block):
# SPATIAL=""
SPATIAL="--no_spatial_attn"

# Uncomment to enable local windowed temporal attention:
# At W=12 this gives only 2 chunks of 6 — negligible benefit here.
# TEMPORAL_WINDOW="--temporal_window 6"
TEMPORAL_WINDOW=""
# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --subset \
    --window           12 \
    --max_delta        3 \
    --num_delta        1 \
    --d_model          128 \
    --enc_layers       6 \
    --dec_layers       2 \
    --batch_size       32 \
    --num_workers      2 \
    --epochs           20 \
    --patience         5 \
    --save_every       5 \
    --amp \
    --factorised_encoder \
    --cross_attn_decoder \
    $SPATIAL \
    $TEMPORAL_WINDOW \
    --wandb_project    station-mae \
    --wandb_run_name   subset-local \
    --save_dir         "$SAVE_DIR"
