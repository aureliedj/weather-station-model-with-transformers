#!/usr/bin/env bash
# run_subset_cloud.sh — pipeline sanity check, full W=288 config
#
# Device is auto-selected by main.py: CUDA (GPU) → MPS (Apple Silicon) → CPU
# Works on cloud GPU, Apple Silicon Mac, or CPU-only — no --device flag needed.
# --amp enables AMP only when CUDA/MPS is active; safe to leave on everywhere.
#
# num_workers=0: Renku/K8s containers cap /dev/shm at ~64 MB. PyTorch workers
# pass collated batches between processes via /dev/shm IPC; with num_workers>0
# and prefetch_factor=4 up to ~700 MB of batches queue there → OOM.
# num_workers=0 runs data loading in the main process (no IPC needed).
# Cost is negligible because the GPU is the bottleneck and the dataset is cached.
#
# Usage:
#   chmod +x run_subset_cloud.sh
#   ./run_subset_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
SAVE_DIR="checkpoints/subset_run_remote"

python main.py \
    --data_root              "$DATA_ROOT" \
    --cache_dir              "$DATA_ROOT" \
    --subset \
    --window                 144 \
    --max_delta              6 \
    --num_delta              1 \
    --d_model                128 \
    --enc_layers             6 \
    --dec_layers             2 \
    --batch_size             32 \
    --num_workers            8ß \
    --epochs                 20 \
    --patience               5 \
    --save_every             5 \
    --amp \
    --factorised_encoder \
    --cross_attn_decoder \
    --save_dir               "$SAVE_DIR"
