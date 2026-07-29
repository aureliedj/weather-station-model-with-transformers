#!/usr/bin/env bash
# run_lstm_test_cloud.sh — evaluate a trained LSTM baseline checkpoint.
#
# Mirror of run_test_cloud.sh (the transformer evaluator), for the LSTM baseline.
# Loads best.ckpt, runs the test split, and writes — in the SAME layout as the
# transformer — so the LSTM drops into the comparison / diagnostic notebooks:
#
#   test_results/<run>/best_mr0.00/predictions.pt   (same schema as test.py)
#   test_results/<run>/test_metrics.csv             (per-variable norm+phys, per-delta, skill)
#   test_results/<run>/persistence_metrics.csv
#
# The LSTM has no masking → pure forecaster → results are mr0.00; compare against
# the transformer's mr0.00 numbers. A persistence-collapse check is printed.
#
# Usage:
#   chmod +x run_lstm_test_cloud.sh
#   ./run_lstm_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_NAME="lstm-baseline-v1"
CHECKPOINT="${SCRIPT_DIR}/checkpoints/${RUN_NAME}/best.ckpt"
SAVE_DIR="${SCRIPT_DIR}/test_results"

EXCLUDE="--exclude_stations PFA"   # same station dropped as during training

# blocks: non-overlapping windows (~1,460) — fast, clean — recommended default
# sliding: all overlapping windows — slower, most stable for final metrics
INDEX_MODE="blocks"

# NOTE on batch size: the LSTM folds all 155 stations of a window into the batch,
# so effective sequences per step = batch_size × 155. Keep batch_size modest on the
# 20 GB MIG partition (16 → ~2,480 seqs, fits; 64 → ~9,920 → cuDNN OOM). Raise only
# if you have more VRAM.
python test_lstm.py \
    --data_root   "$DATA_ROOT" \
    --cache_dir   "$DATA_ROOT" \
    --checkpoint  "$CHECKPOINT" \
    $EXCLUDE \
    --batch_size  16 \
    --num_workers 4 \
    --index_mode  "$INDEX_MODE" \
    --save_predictions 200 \
    --save_dir    "$SAVE_DIR" \
    --run_name    "$RUN_NAME"

echo ""
echo "Done. Add \"${RUN_NAME}\" to CMP_VERSIONS in Test_Results_Exploration.ipynb"
echo "(point it at best_mr0.00) to compare against the transformer's mr0.00 results."
