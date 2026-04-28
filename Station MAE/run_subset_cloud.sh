#!/usr/bin/env bash
# run_subset_cloud.sh — pipeline sanity check on cloud GPU (A100 MIG 3g.20gb)
#
# Hardware context
# ----------------
# A100 80GB in MIG mode. Allocated partition: 3g.20gb
#   • 19,968 MiB (~20 GB) VRAM
#   • 28 / 108 streaming multiprocessors
#
# num_workers=0: Renku/K8s containers cap /dev/shm at ~64 MB. PyTorch workers
# pass collated batches via /dev/shm IPC; with num_workers>0 and prefetch_factor=4
# up to ~700 MB of batches queue there → OOM. num_workers=0 runs data loading in
# the main process (no IPC needed). Negligible cost when GPU is the bottleneck.
#
# Key cloud-specific flags vs run_subset.sh:
#   • --bf16         → bfloat16 AMP (native on A100, wider range than fp16)
#   • --compile      → torch.compile default mode (MIG-safe, ~1.3–1.5× faster)
#   • --window 144   → 24-hour context (larger than local subset)
#
# Encoder architecture options (require --factorised_encoder):
#
#   --no_spatial_attn
#     Removes the spatial attention sub-layer from every encoder block.
#     Each station is encoded independently from its own temporal window;
#     cross-station reasoning is delegated entirely to the decoder.
#     Saves ~27% per encoder block — the single most impactful speed flag.
#     Recommended with --cross_attn_decoder.
#
#   --temporal_window N
#     Local windowed temporal attention: splits W timesteps into chunks of N.
#     Odd layers use a Swin-style half-window shift so tokens communicate
#     across chunk boundaries after two layers. W must be divisible by N.
#     At W=144, tw=6 gives 24 one-hour chunks — more meaningful than at W=72.
#
# WandB: set --wandb_project to stream metrics to your WandB dashboard.
#
# Usage:
#   chmod +x run_subset_cloud.sh
#   ./run_subset_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
SAVE_DIR="checkpoints/subset_run_remote"
LOCAL_CACHE="/tmp/station_mae_cache"

# ── Optional encoder flags ────────────────────────────────────────────────────
# Uncomment to remove spatial attention (~27% faster per encoder block):
# SPATIAL=""
SPATIAL="--no_spatial_attn"

# Uncomment to enable local windowed temporal attention:
# W=144 / tw=6 → 24 one-hour chunks; more meaningful speedup than at W=72.
# TEMPORAL_WINDOW="--temporal_window 6"
TEMPORAL_WINDOW=""
# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --subset \
    --window           144 \
    --max_delta        6 \
    --num_delta        1 \
    --d_model          128 \
    --enc_layers       6 \
    --dec_layers       2 \
    --batch_size       32 \
    --num_workers      0 \
    --epochs           20 \
    --patience         5 \
    --save_every       5 \
    --amp \
    --bf16 \
    --compile \
    --factorised_encoder \
    --cross_attn_decoder \
    $SPATIAL \
    $TEMPORAL_WINDOW \
    --wandb_project    station-mae \
    --wandb_run_name   subset-cloud \
    --save_dir         "$SAVE_DIR"
