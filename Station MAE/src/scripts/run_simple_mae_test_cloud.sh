#!/usr/bin/env bash
# run_simple_mae_test_cloud.sh — dump simple-MAE test predictions.
#
# Two scenarios, transformer-compatible layout (drops into the notebook):
#   test_results/<run>/best_mr0.00/predictions.pt  forecast (future 6 h masked)
#   test_results/<run>/best_mr0.50/predictions.pt  outage (50% stations hidden
#                                                  all window + future masked;
#                                                  masked_idx recorded)
# Predictions only — all metrics come from Test_Results_Exploration.ipynb.
# A persistence-skill + dispersion sanity check is printed per scenario.
#
# Δ=0 caveat: untrained for visible stations (see test_simple_mae.py docstring);
# for hidden stations in mr0.50 it IS the gap-filling output.
#
# Usage:
#   bash src/scripts/run_simple_mae_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
LOCAL_CACHE="/tmp/station_mae_cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"
cd "${SRC_DIR}"

RUN_NAME="simple-mae-v1"
CHECKPOINT="${PROJ_DIR}/checkpoints/${RUN_NAME}/best.ckpt"
SAVE_DIR="${PROJ_DIR}/test_results"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[run_simple_mae_test_cloud.sh] checkpoint not found: $CHECKPOINT"
  ls -1 "${PROJ_DIR}/checkpoints" 2>/dev/null | sed 's/^/    /' || true
  exit 1
fi

# Rolling-origin evaluation: sliding windows every 9 steps (90 min) — same
# protocol as the transformer / LSTM dumps. NOTE: the 18 h window (vs their
# 12 h) shifts the forecast origins and trims the window count slightly —
# same distribution, not literally identical windows.
INDEX_MODE="sliding"
STRIDE=9

python test_simple_mae.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --checkpoint       "$CHECKPOINT" \
    --batch_size       8 \
    --num_workers      4 \
    --index_mode       "$INDEX_MODE" \
    --stride           "$STRIDE" \
    --max_windows      0 \
    --station_ratio    0.5 \
    --seed             42 \
    --exclude_stations PFA \
    --save_dir         "$SAVE_DIR" \
    --run_name         "$RUN_NAME"

echo ""
echo "Done. Add \"${RUN_NAME}\" to CMP_RUNS in Test_Results_Exploration.ipynb."
