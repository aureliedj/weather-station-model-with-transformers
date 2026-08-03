#!/usr/bin/env bash
# run_masked_transformer_cloud.sh — the simplest model in the project that
# still does BOTH tasks.
#
#     tokenize → corrupt masked stations IN PLACE → one transformer stack
#              → pool one vector per station → multi-horizon head
#
# No decoder, no query tokens, no delta embedding, no mask-token insertion.
# Masking is the only extra mechanism, and it is done BERT-style (corrupt the
# value embedding, keep the station's slot) rather than MAE-style (remove and
# re-insert) — because the readout pools each station's OWN tokens, so a
# removed station would have nothing to pool.
#
# Both tasks come from one forward pass:
#   forecasting  — head slot k predicts lead delta_grid[k], every station
#   gap-filling  — for a CORRUPTED station, the k=0 slot is the reconstruction
# Loss: Huber(1), k=0 on masked stations only, k>0 on all — project convention.
#
# Trade-off to state in the report: this gives up the MAE efficiency trick
# (the sequence never shrinks with masking) in exchange for having no decoder
# at all and a representation for every station, observed or not.
#
# Usage:
#   bash src/scripts/run_masked_transformer_cloud.sh
#   SANITY=1 bash ...   # ~5 min, does it run?
#   SANITY=2 bash ...   # ~1 h,  does it learn?

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
LOCAL_CACHE="/tmp/station_mae_cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"
cd "${SRC_DIR}"

source "${SCRIPT_DIR}/_wandb_preflight.sh"

RUN_NAME="masked-tf-v1"
SAVE_DIR="${PROJ_DIR}/checkpoints/${RUN_NAME}"
EXCLUDE="--exclude_stations PFA"

# ── Tokenization ─────────────────────────────────────────────────────────────
# The sequence is (72/PATCH) x 155 and does NOT shrink with masking, because
# corrupted stations keep their slot. So PATCH controls sequence length alone:
#   PATCH=6  12 positions -> 1,860 tokens   ~48 GFLOP/pass   <- default
#   PATCH=3  24 positions -> 3,720 tokens  ~138 GFLOP/pass
# Why 6 by default: v15's raw run (5,616 tokens) collapsed — softmax over that
# many keys starts near-uniform, the model settles on "predict the mean", and
# val/horizon_sensitivity FELL to 0.06. 1,860 is in the regime that trained
# fine. Lead-time resolution here comes from the HEAD, not the tokens, so
# hourly input costs less than it did in v14.
PATCH=6

# ── Sanity / smoke runs ──────────────────────────────────────────────────────
SANITY="${SANITY:-0}"
SANITY_ARGS=()
if [[ "$SANITY" == "1" ]]; then
    SANITY_ARGS=(--epochs 2 --random_epoch_size 200 --subset --warmup_epochs 1)
    SAVE_DIR="${SAVE_DIR}-sanity"; RUN_SUFFIX="-sanity1"
elif [[ "$SANITY" == "2" ]]; then
    SANITY_ARGS=(--epochs 15 --random_epoch_size 3600 --subset --warmup_epochs 2)
    SAVE_DIR="${SAVE_DIR}-sanity"; RUN_SUFFIX="-sanity2"
else
    RUN_SUFFIX=""
fi
if [[ "$SANITY" != "0" ]]; then
    echo "[sanity] SANITY=${SANITY} — short run into ${SAVE_DIR}"
    echo "[sanity]   ${SANITY_ARGS[*]}"
fi

python train_masked_transformer.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --delta_grid_stride 3 \
    --temporal_patch   $PATCH \
    --d_model          384 \
    --n_layers         8 \
    --n_heads          8 \
    --mlp_ratio        4.0 \
    --dropout          0.0 \
    --mask_ratio       0.5 \
    --val_mask_ratio   0.0 \
    --huber_delta      1.0 \
    --pool             last \
    --batch_size       8 \
    --accumulate_grad_batches 2 \
    --num_workers      3 \
    --epochs           100 \
    --lr               3e-4 \
    --warmup_epochs    10 \
    --min_lr           1e-6 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --bf16 \
    --monitor          val/overall_mae \
    --patience         40 \
    --index_mode       random \
    --random_epoch_size 10000 \
    $EXCLUDE \
    ${SANITY_ARGS[@]+"${SANITY_ARGS[@]}"} \
    --wandb_project    station-mae \
    --wandb_run_name   "${RUN_NAME}${RUN_SUFFIX}" \
    --save_dir         "$SAVE_DIR"

echo ""
echo "Done. Best checkpoint: ${SAVE_DIR}/best.ckpt"
echo "Watch val/horizon_sensitivity: RISING = using the lead structure;"
echo "falling toward 0.05 = the constant-output collapse that killed v15-raw."
