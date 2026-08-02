#!/usr/bin/env bash
# run_simple_mae_cloud.sh — train the canonical MAE (model/simple_mae.py).
#
# One mechanism, one objective: (station × hour) tokens, mask a subset, ViT
# encoder on the visible tokens, shallow decoder reconstructs the hidden ones.
# Loss follows the PROJECT convention (not strict MAE): context hours
# ("time ≤ 0") are scored on masked tokens only; the future block
# (+10 min … +6 h) is scored on ALL tokens, masked or visible.
# The mask pattern defines the task:
#   random tokens (75%)  → pretraining signal
#   whole stations (50%) → gap-filling
#   last 6 hours         → forecasting (all leads in one pass)
# Training samples one strategy per batch (0.5 / 0.25 / 0.25); validation
# always runs the FUTURE mask, so val/* is a forecast metric comparable in
# spirit to the other models (exact protocol differs — see notes below).
#
# Embeddings: the existing Fourier stack (position / topography / absolute
# time) is KEPT; StepIndex and DeltaTime embeddings are RETIRED — a masked
# token's time embedding is its lead-time conditioning; an 18-slot learned
# embedding covers within-window order.
#
# Size: ~10 M params — trains overnight, not over days.
#
# Comparability notes vs LSTM / v14 val panels:
#   • window is 18 h (12 h context + 6 h future), others use 12 h context
#   • val/overall_rmse here = RMSE over the WHOLE masked 6 h block,
#     not the single 30-min lead the other models log
#   • val/persist_skill uses the last visible step as baseline
#
# Usage:
#   bash src/scripts/run_simple_mae_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
LOCAL_CACHE="/tmp/station_mae_cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root
cd "${SRC_DIR}"

# WandB offline on Renku (flaky egress falsely marks runs "crashed").
source "${SCRIPT_DIR}/_wandb_preflight.sh"

RUN_NAME="simple-mae-v1"
SAVE_DIR="${PROJ_DIR}/checkpoints/${RUN_NAME}"

EXCLUDE="--exclude_stations PFA"   # same station dropped as everywhere else

# ── Window & masking ─────────────────────────────────────────────────────────
WINDOW=108          # 18 h = 12 h context + 6 h future (10-min steps)
PATCH=6             # 1 token = 1 station-hour (6 steps × 6 vars)
MASK_RATIO=0.75     # random-strategy token ratio (MAE optimum, He et al.)
STATION_RATIO=0.5   # station-strategy: fraction of stations fully hidden
FUTURE_SLOTS=6      # future-strategy: hours hidden at the window end
STRATEGY_PROBS="0.5 0.25 0.25"   # random / station / future per batch

# ── Subset ablation first? Uncomment for a ~1 h smoke run. ──────────────────
SUBSET="--subset --epochs 15 --random_epoch_size 3600"
# SUBSET=""

python train_simple_mae.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           $WINDOW \
    --patch_steps      $PATCH \
    --d_model          256 \
    --enc_layers       8 \
    --enc_heads        8 \
    --dec_dim          128 \
    --dec_layers       2 \
    --dec_heads        4 \
    --dropout          0.0 \
    --mask_ratio       $MASK_RATIO \
    --station_ratio    $STATION_RATIO \
    --future_slots     $FUTURE_SLOTS \
    --strategy_probs   $STRATEGY_PROBS \
    --batch_size       16 \
    --num_workers      3 \
    --epochs           80 \
    --lr               3e-4 \
    --warmup_epochs    8 \
    --min_lr           1e-6 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --bf16 \
    --monitor          val/overall_rmse \
    --patience         25 \
    --index_mode       random \
    --random_epoch_size 10000 \
    $EXCLUDE \
    $SUBSET \
    --wandb_project    station-mae \
    --wandb_run_name   "$RUN_NAME" \
    --save_dir         "$SAVE_DIR"

echo ""
echo "Done. Best checkpoint: ${SAVE_DIR}/best.ckpt"
echo "Inference = one forward with the mask you care about (future / station / both)."
