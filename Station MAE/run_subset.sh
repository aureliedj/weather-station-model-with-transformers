#!/usr/bin/env bash
# run_subset.sh — pipeline sanity check on 2 years of data (light local config)
#
# Device is auto-selected by main.py: CUDA (GPU) → MPS (Apple Silicon) → CPU
# --amp is safe on all three; it enables AMP only when CUDA/MPS is active.
#
# Usage:
#   chmod +x run_subset.sh
#   ./run_subset.sh
#
# Tweak DATA_ROOT and SAVE_DIR before running.
# Use NUM_WORKERS=0 on macOS if you hit shared-memory issues (rare).

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
    --save_dir    "$SAVE_DIR"
