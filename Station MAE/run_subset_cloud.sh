#!/usr/bin/env bash
# run_subset_cloud.sh — pipeline sanity check on cloud GPU (Linux/CUDA)
#
# Usage:
#   chmod +x run_subset_cloud.sh
#   ./run_subset_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
SAVE_DIR="checkpoints/subset_run_remote"

python main.py \
    --data_root   "$DATA_ROOT" \
    --cache_dir   "$DATA_ROOT" \
    --subset \
    --window      288 \
    --max_delta   18 \
    --num_delta   3 \
    --d_model     128 \
    --enc_layers  6 \
    --dec_layers  2 \
    --batch_size  32 \
    --num_workers 4 \
    --epochs      20 \
    --patience    5 \
    --save_every  5 \
    --amp \
    --save_dir    "$SAVE_DIR"
