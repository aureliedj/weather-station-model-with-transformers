#!/usr/bin/env bash
# run_subset.sh — pipeline sanity check on 2 years of data (light local config)
#
# Device is auto-selected by Lightning: CUDA (GPU) → MPS (Apple Silicon) → CPU
# --amp enables fp16 AMP only when CUDA/MPS is active; safe to leave on everywhere.
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

python main.py \
    --data_root   "$DATA_ROOT" \
    --cache_dir   "$DATA_ROOT" \
    --subset \
    --window      12 \
    --max_delta   3 \
    --num_delta   1 \
    --d_model     128 \
    --enc_layers  6 \
    --dec_layers  2 \
    --batch_size  32 \
    --num_workers 2 \
    --epochs      20 \
    --patience    5 \
    --save_every  5 \
    --amp \
    --factorised_encoder \
    --cross_attn_decoder \
    --wandb_project   station-mae \
    --wandb_run_name  subset-local \
    --save_dir    "$SAVE_DIR"
