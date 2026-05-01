#!/usr/bin/env bash

#
# Usage:
#   chmod +x run_test_subset.sh
#   ./run_test_subset.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
CKPT_DIR="checkpoints/full_run_cloud"
SAVE_DIR="test_results/full_run_cloud"

python test.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --checkpoint       "$CKPT_DIR/best.ckpt" \
    --batch_size       16 \
    --window 72 \
    --max_delta 0 \
    --mlp_ratio 2.0 \
    --num_workers      5 \
    --d_model          128 \
    --enc_layers       6 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --factorised_encoder \
    --cross_attn_decoder \
    --gap_fill_repeats 3 \

    --wandb_project    station-mae \
    --wandb_run_name   test-full-cloud-best \
    --save_dir         "$SAVE_DIR"
