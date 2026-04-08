#!/usr/bin/env bash
# run_subset.sh — pipeline sanity check on 2 years of data (local MPS run)
#
# Usage:
#   chmod +x run_subset.sh
#   ./run_subset.sh
#
# Tweak DATA_ROOT and SAVE_DIR before running.
# Set NUM_WORKERS=0 on macOS to avoid shared-memory issues.

set -euo pipefail

DATA_ROOT="/Users/aureliedejong/Documents/ETH/_DAS Project/PeakWeatherDataset"
SAVE_DIR="checkpoints/subset_run"

python main.py \
    --data_root   "$DATA_ROOT" \
    --cache_dir   "$DATA_ROOT" \
    --subset \
    --window      12 \
    --max_delta   6 \
    --num_delta   3 \
    --d_model     256 \
    --enc_layers  6 \
    --dec_layers  2 \
    --batch_size  32 \
    --num_workers 2 \
    --epochs      20 \
    --patience    5 \
    --save_every  5 \
    --amp \
    --save_dir    "$SAVE_DIR"
