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

# Resolve paths relative to this script so they work from any working directory.
# Checkpoints are saved by run_full_cloud.sh inside the project directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${SCRIPT_DIR}/checkpoints/full_run_cloud/best.ckpt"

# Save test results inside the project directory (guaranteed writable on Renku)
SAVE_DIR="${SCRIPT_DIR}/test_results"

# ── Window mode ───────────────────────────────────────────────────────────────
# blocks:  non-overlapping windows (~1,460) — fast, clean — recommended default
# sliding: all overlapping windows (~105k)  — slow, most stable — for final paper metrics
# Rolling-origin evaluation: sliding windows every STRIDE steps (9 = 90 min = 1h30).
INDEX_MODE="sliding"
STRIDE=9
# INDEX_MODE="blocks"   # fast non-overlapping alternative

# ── Mask ratio sweep ──────────────────────────────────────────────────────────
# 0.0 → all stations visible to encoder (pure temporal forecasting)
# 0.5 → trained setting: 50% masked (gap-filling + forecasting)
# The model is loaded ONCE and evaluated at each ratio in sequence.
# Results save to separate subdirs: test_results/.../best_mr0.00/, best_mr0.50/
# A comparison table is printed at the end.
# ── Normalisation mode ───────────────────────────────────────────────────────
# Use --global_norm for checkpoints trained BEFORE per-station normalisation
# was introduced (i.e. all baseline-cloud and early tw12-d1024 runs).
# Omit (default) for new checkpoints trained with per-station normalisation.
# GLOBAL_NORM="--global_norm"
GLOBAL_NORM=""

MASK_RATIOS="0.0 0.5"
# MASK_RATIOS="0.5"                 # single ratio (faster)
# MASK_RATIOS="0.0 0.25 0.5 0.75"  # full robustness sweep

python test.py \
    --data_root        "$DATA_ROOT" \
    --checkpoint       "$CHECKPOINT" \
    --batch_size       8 \
    --num_workers      4 \
    --index_mode       "$INDEX_MODE" \
    --stride           "$STRIDE" \
    --exclude_stations PFA \
    --test_mask_ratios $MASK_RATIOS \
    --gap_fill_repeats 3 \
    --save_predictions 100000 \
    --seasonal \
    $GLOBAL_NORM \
    --save_dir         "$SAVE_DIR" \
    --wandb_project    station-mae \
    --wandb_run_name   "test-$(basename $CHECKPOINT .ckpt)-${INDEX_MODE}"
