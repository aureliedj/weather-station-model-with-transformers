#!/usr/bin/env bash
# run_test_cloud.sh — prediction dump on the cloud (predictions-only)
#
# Loads the best checkpoint and runs ONE forward pass over the sliding test
# windows per mask ratio, saving raw predictions only:
#
#   test_results/best_mr0.00/predictions.pt   (all stations visible)
#   test_results/best_mr0.50/predictions.pt   (50% masked; masked_idx included)
#
# NO metrics are computed here — persistence, skill, per-station, seasonal etc.
# are all derived downstream from predictions.pt in Test_Results_Exploration.ipynb.
# Same schema as the LSTM dump (test_lstm.py), so runs drop into the same
# comparison cells.
#
# All architecture settings are auto-read from the checkpoint.
#
# Usage:
#   chmod +x run_test_cloud.sh
#   ./run_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"

# Resolve paths relative to this script so they work from any working directory.
# Checkpoints are saved by run_full_cloud.sh inside the project directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src   — python entry points live here
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root — checkpoints/, test_results/, report/
cd "${SRC_DIR}"                                              # so `python main.py` and `from data...` resolve
CHECKPOINT="${PROJ_DIR}/checkpoints/full_run_cloud_v11/best.ckpt"
# CHECKPOINT="${PROJ_DIR}/checkpoints/best.ckpt"   # ← v9 (NLL) saved checkpoint

# Save under a per-run subfolder — Test_Results_Exploration.ipynb expects
# test_results/<run>/best_mr0.00/ (same layout as the LSTM's lstm-baseline-v1/).
# Keep RUN_NAME in sync with the checkpoint being evaluated.
RUN_NAME="v11"
# RUN_NAME="v9"          # ← when evaluating checkpoints/best.ckpt (v9, NLL)
SAVE_DIR="${PROJ_DIR}/test_results/${RUN_NAME}"

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

# The model is loaded ONCE; each ratio is a single forward sweep over the same
# sliding windows. --save_predictions 0 = keep ALL windows (sliding/9 over
# 2023-24 ≈ 11k windows; expect ~1.5 GB per ratio).
# batch_size 2: the flat d1024 encoder at W=72, N=155 is memory-hungry even in
# bf16; batch 8 fp32 OOMs on an 8 GB MIG slice. Throughput is data-loading
# bound anyway, so small batches cost little. Raise to 8 on a 20 GB partition.
python test.py \
    --data_root        "$DATA_ROOT" \
    --checkpoint       "$CHECKPOINT" \
    --batch_size       4 \
    --num_workers      4 \
    --index_mode       "$INDEX_MODE" \
    --stride           "$STRIDE" \
    --exclude_stations PFA \
    --test_mask_ratios $MASK_RATIOS \
    --predictions_only \
    --save_predictions 0 \
    $GLOBAL_NORM \
    --save_dir         "$SAVE_DIR"

echo ""
echo "Done. Predictions at: ${SAVE_DIR}/best_mr0.00/ and best_mr0.50/"
echo "Compute all metrics in Test_Results_Exploration.ipynb."
