#!/usr/bin/env bash
# Train one of the four Transformer variants reported in the report.
#
#   bash scripts/train_transformer.sh <variant>
#
#   variant   run directory              training mask ratio   objective      cross-station attention
#   mae       full_run_cloud_v27         0.5                   Huber          yes
#   prob      full_run_cloud_v30-nll     0.5                   Gaussian NLL   yes
#   dense     full_run_cloud_v31         0.0                   Huber          yes
#   blind     full_run_cloud_v32-blind   0.0                   Huber          no (station-local encoder and decoder)
#
# Environment: DATA_ROOT (default <repo>/PeakWeatherDataset), WANDB_PROJECT
# (unset = CSV logging), RESUME=1 to continue from last.ckpt.
set -euo pipefail

VARIANT="${1:-mae}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO}/PeakWeatherDataset}"
[[ -d "$DATA_ROOT" ]] || { echo "dataset not found at $DATA_ROOT (run: python src/download.py)"; exit 1; }

case "$VARIANT" in
  mae)   RUN=full_run_cloud_v27;       MASK=0.5; EXTRA="" ;;
  prob)  RUN=full_run_cloud_v30-nll;   MASK=0.5; EXTRA="--nll_loss" ;;
  dense) RUN=full_run_cloud_v31;       MASK=0.0; EXTRA="" ;;
  blind) RUN=full_run_cloud_v32-blind; MASK=0.0; EXTRA="--no_spatial_attn --station_local_decoder" ;;
  *) echo "unknown variant '$VARIANT' (mae | prob | dense | blind)"; exit 1 ;;
esac

SAVE_DIR="${REPO}/checkpoints/${RUN}"
RESUME_ARG=()
if [[ "${RESUME:-0}" == "1" && -f "${SAVE_DIR}/last.ckpt" ]]; then
  RESUME_ARG=(--resume "${SAVE_DIR}/last.ckpt")
fi
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN")
fi

python "${REPO}/src/main.py" \
    --data_root "$DATA_ROOT" --cache_dir "$DATA_ROOT" \
    --exclude_stations PFA \
    --window 72 --max_delta 36 --delta_grid_stride 3 --temporal_patch 3 \
    --d_model 384 --enc_heads 8 --dec_heads 8 --enc_layers 8 --dec_layers 2 --mlp_ratio 4.0 \
    --dropout 0.1 --drop_path_rate 0.1 \
    --mask_ratio "$MASK" --val_mask_ratio 0.0 --residual_head \
    --var_weights 1.0 1.0 1.0 1.0 1.0 \
    --index_mode random --random_epoch_size 10000 \
    --batch_size 4 --accumulate_grad_batches 4 --num_workers 3 \
    --epochs 100 --lr 1e-4 --min_lr 5e-7 --warmup_epochs 10 --weight_decay 0.05 --grad_clip 1.0 \
    --monitor val/overall_mae --patience 40 --overfit_stop --overfit_patience 20 \
    --amp --bf16 --compile --grad_checkpoint \
    $EXTRA \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
    --save_dir "$SAVE_DIR"
