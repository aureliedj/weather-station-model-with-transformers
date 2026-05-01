#!/usr/bin/env bash
# run_test_subset.sh — test-set evaluation for the subset training run
#
# Runs three evaluation modes in a single call:
#   1. Standard eval  — all deltas (1..max_delta), all stations
#                       → test_results/subset_run/test_metrics.csv
#   2. Lead-1 eval    — fixed delta=1 (10-min forecast), all stations
#                       → test_results/subset_run/lead1_metrics.csv
#   3. Gap-filling    — delta=0 reconstruction, masked stations only
#                       → test_results/subset_run/gap_filling_metrics.csv
#
# Architecture flags must match those used in run_subset.sh.
#
# WandB: metrics from all three modes are logged under the same run with
#        eval/, lead1/, and gap/ key prefixes.  Omit --wandb_project to
#        disable WandB and log to CSV only.
#
# Usage:
#   chmod +x run_test_subset.sh
#   ./run_test_subset.sh

set -euo pipefail

DATA_ROOT="/Users/aureliedejong/Documents/ETH/_DAS Project/PeakWeatherDataset"
CKPT_DIR="checkpoints/full_run_cloud"
SAVE_DIR="test_results/full_run_cloud"

python test.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --checkpoint       "$CKPT_DIR/best.ckpt" \
    --window           72 \
    --max_delta        6 \
    --batch_size       16 \
    --num_workers      2 \
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
