#!/usr/bin/env bash
# run_test_cloud.sh — test-set evaluation on the cloud
#
# Loads the best checkpoint, runs full evaluation on the 2023-2024 test split,
# and saves metrics to test_results/.
#
# All architecture settings are auto-read from the checkpoint.
# Only --data_root and --checkpoint are required.
#
# Usage:
#   chmod +x run_test_cloud.sh
#   ./run_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
CHECKPOINT="checkpoints/best.ckpt"
SAVE_DIR="test_results/$(basename $(dirname $CHECKPOINT))"

# ── Evaluation mode ──────────────────────────────────────────────────────────
# sliding (default): all overlapping windows (~105k) — slow, most stable metrics
# blocks:            non-overlapping windows only (~1,460) — fast, use for quick checks
#                    and final paper metrics
# INDEX_MODE="sliding"
INDEX_MODE="blocks"

python test.py \
    --data_root     "$DATA_ROOT" \
    --checkpoint    "$CHECKPOINT" \
    --batch_size    64 \
    --num_workers   4 \
    --index_mode    "$INDEX_MODE" \
    --save_dir      "$SAVE_DIR" \
    --wandb_project station-mae \
    --wandb_run_name "test-$(basename $CHECKPOINT .ckpt)-${INDEX_MODE}"
