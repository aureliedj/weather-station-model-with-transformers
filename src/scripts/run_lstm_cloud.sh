#!/usr/bin/env bash
# run_lstm_cloud.sh — train + evaluate the per-station LSTM baseline.
#
# A deliberately simple, spatially-blind recurrent baseline (see model/lstm_baseline.py
# and the meeting notes).  It reuses StationMAEDataset, so data / normalisation /
# splits / forecast-horizon grid are IDENTICAL to the transformer — an apples-to-apples
# comparison.  Because it has no station masking it is a pure forecaster: compare it
# against the transformer's mr0.00 numbers.
#
# What it does:
#   1. Trains the LSTM (train_lstm.py)
#   2. Evaluates it on the test split and writes predictions.pt in the same schema
#      as the transformer, so it drops into Test_Results_Exploration.ipynb — plus a
#      persistence baseline and an explicit "did it collapse to persistence?" check.
#
# Usage:
#   chmod +x run_lstm_cloud.sh
#   ./run_lstm_cloud.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src   — python entry points live here
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root — checkpoints/, test_results/, report/
cd "${SRC_DIR}"                                              # so `python main.py` and `from data...` resolve

# Override without editing this file:
#   DATA_ROOT=/path/to/PeakWeatherDataset bash src/scripts/run_lstm_cloud.sh
# Default matches src/download.py's own default (<project root>/PeakWeatherDataset),
# not a Renku-specific path — this project's copy lives at the project root.
DATA_ROOT="${DATA_ROOT:-${PROJ_DIR}/PeakWeatherDataset}"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[run_lstm_cloud.sh] dataset not found at: $DATA_ROOT"
  echo "  Set DATA_ROOT, or fetch it with:  python src/download.py"
  exit 1
fi
LOCAL_CACHE="/tmp/station_mae_cache"

# WandB offline on Renku (flaky egress falsely marks runs "crashed"; training is
# unaffected). Backfill later: wandb sync "$(ls -dt wandb/run-* | head -1)"
source "${SCRIPT_DIR}/_wandb_preflight.sh"

RUN_NAME="lstm-baseline-v1"
SAVE_DIR="${PROJ_DIR}/checkpoints/${RUN_NAME}"

EXCLUDE="--exclude_stations PFA"   # same station dropped as the transformer

# ── Model size ────────────────────────────────────────────────────────────────
# hidden ≥ transformer d_model per the meeting note ("embedding space use
# similar or higher than transformer").  3 LSTM layers.
#
# STALE-COMMENT FIX (2026-08): this used to claim "a few M params — orders of
# magnitude smaller than the ~130M transformer". Both numbers referred to the
# old d1024 era. Actual counts for the CURRENT configs:
#   LSTM  h=1024 x3 + mask feature : ~21.1M params
#   transformer d384 (v27 family)  : ~24M core + embeddings
# i.e. the two are the SAME order of magnitude in parameters. Nominal FLOPs
# per window are actually HIGHER for the LSTM (~470 GFLOP: 155 stations x 72
# steps x ~21M MAC) than the patched transformer (~48 GFLOP encoder); the
# LSTM is only faster in wall-clock because cuDNN fuses the recurrence.
HIDDEN=1024
LAYERS=3
DROPOUT=0.1

# ── Missing-sensor handling ────────────────────────────────────────────────────
# Feed the sensor-availability mask as extra input features (6 vars → 12) so the
# LSTM can distinguish a true 0 (station mean in normalised space) from a missing
# sensor. Set to "" to fall back to plain zero-imputation.
MASK_FEATURE="--use_mask_feature"
# MASK_FEATURE=""   # ← plain: missing values are just zeros

# ── Loss: same family as the transformer (Huber δ=1.0). Uniform weights for a
# neutral baseline. Change to match a specific transformer run if desired. ──────
HUBER_DELTA=1.0
VAR_WEIGHTS="1.0 1.0 1.0 1.0 1.0"   # [temp, pressure, humidity, wind_u, wind_v]

# ── Window sampling: same random-window scheme as the transformer ──────────────
# INDEX_MODE="--index_mode random --random_epoch_size 10000"
INDEX_MODE="--index_mode sliding --train_stride 9"         # sliding, hourly stride

# ── Optimizer: adamw (default) or rmsprop (classic for RNNs — supervisor's link) ─
OPTIMIZER="adamw"
# OPTIMIZER="rmsprop"

# ─────────────────────────────────────────────────────────────────────────────
echo "=== [1/2] Training LSTM baseline ==="
python train_lstm.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --delta_grid_stride 3 \
    --hidden           "$HIDDEN" \
    --lstm_layers      "$LAYERS" \
    --lstm_dropout     "$DROPOUT" \
    $MASK_FEATURE \
    --huber_delta      "$HUBER_DELTA" \
    --var_weights      $VAR_WEIGHTS \
    --batch_size       16 \
    --num_workers      3 \
    --epochs           60 \
    --lr               1e-3 \
    --min_lr           1e-6 \
    --weight_decay     0.0 \
    --warmup_epochs    3 \
    --grad_clip        1.0 \
    --optimizer        "$OPTIMIZER" \
    --monitor          val/overall_mae \
    --patience         15 \
    --overfit_stop \
    --overfit_patience 5 \
    --amp \
    --bf16 \
    $INDEX_MODE \
    $EXCLUDE \
    --wandb_project    station-mae \
    --wandb_run_name   "$RUN_NAME" \
    --save_dir         "$SAVE_DIR"

echo "=== [2/2] Evaluating on test split (delegates to run_lstm_test_cloud.sh) ==="
# Keep RUN_NAME in sync: run_lstm_test_cloud.sh reads the same default.
bash "${SCRIPT_DIR}/run_lstm_test_cloud.sh"
