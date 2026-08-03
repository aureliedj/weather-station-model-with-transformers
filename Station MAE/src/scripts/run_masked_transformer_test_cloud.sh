#!/usr/bin/env bash
# run_masked_transformer_test_cloud.sh — dump masked-transformer predictions.
#
#   test_results/<run>/best_mr0.00/predictions.pt   all stations observed
#   test_results/<run>/best_mr0.50/predictions.pt   50% corrupted (masked_idx kept)
#
# Predictions only — every metric is computed in Test_Results_Exploration.ipynb,
# in the same schema as v15 / simple-MAE / LSTM, so all four overlay.
#
# Usage:
#   bash src/scripts/run_masked_transformer_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
LOCAL_CACHE="/tmp/station_mae_cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"
cd "${SRC_DIR}"

source "${SCRIPT_DIR}/_cuda_preflight.sh"

# Keep in sync with run_masked_transformer_cloud.sh.
RUN_NAME="masked-tf-v1"
CHECKPOINT="${PROJ_DIR}/checkpoints/${RUN_NAME}/best.ckpt"
SAVE_DIR="${PROJ_DIR}/test_results"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[run_masked_transformer_test_cloud.sh] checkpoint not found: $CHECKPOINT"
  ls -1 "${PROJ_DIR}/checkpoints" 2>/dev/null | sed 's/^/    /' || true
  exit 1
fi

# Same rolling-origin protocol as every other dump: sliding, stride 9 (90 min).
# The sequence does not shrink with masking here, so mr0.00 and mr0.50 cost the
# same — unlike v15, where mr0.00 is the heavy case.
python test_masked_transformer.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --checkpoint       "$CHECKPOINT" \
    --batch_size       16 \
    --num_workers      4 \
    --index_mode       sliding \
    --stride           9 \
    --max_windows      0 \
    --mask_ratio       0.5 \
    --seed             42 \
    --exclude_stations PFA \
    --save_dir         "$SAVE_DIR" \
    --run_name         "$RUN_NAME"

echo ""
echo "Done. Add \"${RUN_NAME}\" to CMP_RUNS in Test_Results_Exploration.ipynb."
