#!/usr/bin/env bash
# Write the test-set predictions of the LSTM baseline
# (-> test_results/lstm-baseline-v1/best_mr0.00/predictions.pt).
# Same windows as the Transformer: sliding, 90-min stride over 2023-2024.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO}/PeakWeatherDataset}"
RUN=lstm-baseline-v1

python "${REPO}/src/test_lstm.py" \
    --data_root "$DATA_ROOT" --cache_dir "$DATA_ROOT" \
    --checkpoint "${REPO}/checkpoints/${RUN}/best.ckpt" \
    --exclude_stations PFA \
    --index_mode sliding --stride 9 \
    --batch_size 8 --num_workers 4 \
    --save_dir "${REPO}/test_results" --run_name "$RUN"
