#!/usr/bin/env bash
# run_lstm_test_cloud.sh — evaluate a trained LSTM baseline checkpoint.
#
# Mirror of run_test_cloud.sh (the transformer dump), for the LSTM baseline.
# Loads best.ckpt and dumps raw predictions over the sliding test windows —
# predictions ONLY, no metrics files:
#
#   test_results/<run>/best_mr0.00/predictions.pt   (same schema as test.py)
#
# All metrics are computed downstream in Test_Results_Exploration.ipynb from
# predictions.pt (persistence is derived from targets[:, 0] there — no CSV).
# The LSTM has no station masking → pure forecaster → results live under
# best_mr0.00; compare against the transformer's best_mr0.00 dump.
# A persistence-collapse sanity check is printed at the end.
#
# Usage:
#   chmod +x run_lstm_test_cloud.sh
#   ./run_lstm_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src   — python entry points live here
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root — checkpoints/, test_results/, report/
cd "${SRC_DIR}"

# Fail fast if torch cannot use the GPU (wheel/driver mismatch on the older node).
source "${SCRIPT_DIR}/_cuda_preflight.sh"                                              # so `python main.py` and `from data...` resolve

# Keep in sync with run_lstm_cloud.sh — it trains lstm-baseline-v2 and calls
# THIS script at the end, so a stale name here would silently re-dump the old
# checkpoint. Set to lstm-baseline-v1 to regenerate the earlier baseline
# (its dump already exists under test_results/lstm-baseline-v1/).
RUN_NAME="lstm-baseline-v2"
CHECKPOINT="${PROJ_DIR}/checkpoints/${RUN_NAME}/best.ckpt"
SAVE_DIR="${PROJ_DIR}/test_results"

EXCLUDE="--exclude_stations PFA"   # same station dropped as during training

# blocks: non-overlapping windows (~1,460) — fast, clean — recommended default
# sliding: all overlapping windows — slower, most stable for final metrics
# Rolling-origin evaluation: sliding windows every STRIDE steps (9 = 90 min = 1h30).
INDEX_MODE="sliding"
STRIDE=9
# INDEX_MODE="blocks"   # fast non-overlapping alternative

# NOTE on batch size: the LSTM folds all 155 stations of a window into the batch,
# so effective sequences per step = batch_size × 155. Keep batch_size modest on the
# 20 GB MIG partition (16 → ~2,480 seqs, fits; 64 → ~9,920 → cuDNN OOM). Raise only
# if you have more VRAM.
python test_lstm.py \
    --data_root   "$DATA_ROOT" \
    --cache_dir   "$DATA_ROOT" \
    --checkpoint  "$CHECKPOINT" \
    $EXCLUDE \
    --batch_size  8 \
    --num_workers 4 \
    --index_mode  "$INDEX_MODE" \
    --stride      "$STRIDE" \
    --max_windows 0 `# 0 = save ALL windows (sliding/9 over 2023-24 ≈ 11k windows ≈ 1.5 GB)` \
    --save_dir    "$SAVE_DIR" \
    --run_name    "$RUN_NAME"

echo ""
echo "Done. Add \"${RUN_NAME}\" to CMP_VERSIONS in Test_Results_Exploration.ipynb"
echo "(point it at best_mr0.00) to compare against the transformer's mr0.00 results."
