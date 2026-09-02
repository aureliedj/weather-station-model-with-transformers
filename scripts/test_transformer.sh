#!/usr/bin/env bash
# Write the test-set predictions of a trained Transformer variant.
#
#   bash scripts/test_transformer.sh <variant> [mask ratios]
#
#   bash scripts/test_transformer.sh mae 0.0 0.5     # -> test_results/v27/best_mr0.00, best_mr0.50
#   bash scripts/test_transformer.sh prob 0.0 0.5    # -> test_results/v30-nll/...
#   bash scripts/test_transformer.sh dense           # -> test_results/v31/best_mr0.00
#   bash scripts/test_transformer.sh blind           # -> test_results/v32-blind/best_mr0.00
#
# Protocol used for every reported number: sliding windows with a 90-min
# stride over 2023-2024 (11,684 windows), seed 42, batch size 4. Mask ratio
# 0.5 is meaningful only for the two models trained with masking (mae, prob).
# Environment: DATA_ROOT, SEED (42), BATCH_SIZE (4).
set -euo pipefail

VARIANT="${1:-mae}"; shift || true
MASK_RATIOS="${*:-0.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO}/PeakWeatherDataset}"

case "$VARIANT" in
  mae)   RUN=full_run_cloud_v27;       NAME=v27 ;;
  prob)  RUN=full_run_cloud_v30-nll;   NAME=v30-nll ;;
  dense) RUN=full_run_cloud_v31;       NAME=v31 ;;
  blind) RUN=full_run_cloud_v32-blind; NAME=v32-blind ;;
  *) echo "unknown variant '$VARIANT' (mae | prob | dense | blind)"; exit 1 ;;
esac

python "${REPO}/src/test.py" \
    --data_root "$DATA_ROOT" --cache_dir "$DATA_ROOT" \
    --checkpoint "${REPO}/checkpoints/${RUN}/best.ckpt" \
    --exclude_stations PFA \
    --index_mode sliding --stride 9 \
    --test_mask_ratios $MASK_RATIOS \
    --seed "${SEED:-42}" --batch_size "${BATCH_SIZE:-4}" --num_workers 4 \
    --save_dir "${REPO}/test_results/${NAME}"
