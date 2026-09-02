#!/usr/bin/env bash
# Train the per-station LSTM baseline (run directory checkpoints/lstm-baseline-v1).
# Environment: DATA_ROOT, WANDB_PROJECT (unset = CSV logging).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO}/PeakWeatherDataset}"
[[ -d "$DATA_ROOT" ]] || { echo "dataset not found at $DATA_ROOT (run: python src/download.py)"; exit 1; }
RUN=lstm-baseline-v1
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN")
fi

python "${REPO}/src/train_lstm.py" \
    --data_root "$DATA_ROOT" --cache_dir "$DATA_ROOT" \
    --exclude_stations PFA \
    --window 72 --max_delta 36 --delta_grid_stride 3 \
    --hidden 1024 --lstm_layers 3 --lstm_dropout 0.1 --use_mask_feature \
    --huber_delta 1.0 --var_weights 1.0 1.0 1.0 1.0 1.0 \
    --index_mode sliding --train_stride 9 \
    --batch_size 16 --num_workers 3 \
    --epochs 60 --lr 1e-3 --min_lr 1e-6 --weight_decay 0.0 --warmup_epochs 3 --grad_clip 1.0 \
    --monitor val/overall_mae --patience 15 --overfit_stop --overfit_patience 5 \
    --amp --bf16 \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
    --save_dir "${REPO}/checkpoints/${RUN}"
