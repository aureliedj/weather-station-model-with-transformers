#!/usr/bin/env bash
# run_full.sh — full training run on all available years (2017–2024)
#
# Key differences from run_subset.sh:
#   • No --subset flag          → uses all training years (2017–2021)
#   • --window 288              → 48-hour input window (vs 12 steps / 2h in subset)
#   • --max_delta 36            → 6-hour forecast horizon (vs 30 min in subset)
#   • --num_delta 3             → 3 lead-times per sample; encoder runs once,
#                                 decoder runs 3×  — amortises O(L²) attention cost
#   • --epochs 100              → longer training budget
#   • --patience 15             → more patience on the larger dataset
#   • --save_every 10           → less frequent snapshots
#   • --num_workers 4           → more DataLoader workers for full dataset I/O
#   • --batch_size 16           → reduced from 32; larger windows use more memory
#
# Device is auto-selected: CUDA (GPU) → MPS (Apple Silicon) → CPU
# --amp enables fp16 AMP; safe to leave on everywhere (no-op on CPU).
#
# WandB: credentials are picked up from ~/.netrc after `wandb login`.
#        Omit --wandb_project to fall back to CSV logging only.
#
# Resuming an interrupted run:
#   ./run_full.sh   (just re-run; Lightning restores from last.ckpt automatically)
# or pass explicitly:
#   bash run_full.sh --resume checkpoints/full_run/last.ckpt
#
# Usage:
#   chmod +x run_full.sh
#   ./run_full.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
SAVE_DIR="checkpoints/full_run"

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --window           288 \
    --max_delta        36 \
    --num_delta        3 \
    --d_model          128 \
    --enc_layers       6 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.1 \
    --batch_size       16 \
    --num_workers      4 \
    --epochs           100 \
    --lr               1e-4 \
    --warmup_epochs    5 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --patience         15 \
    --save_every       5 \
    --amp \
    --factorised_encoder \
    --cross_attn_decoder \
    --wandb_project    station-mae \
    --wandb_run_name   full-run \
    --save_dir         "$SAVE_DIR"
