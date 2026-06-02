#!/usr/bin/env bash
# run_subset_cloud.sh — smoke test / fast validation run
#
# Purpose
# -------
# Mirrors run_full_cloud.sh exactly in architecture and flags but runs for
# only 1 epoch / 50 batches so you can verify the full pipeline (cache build,
# normalisation, fixed-grid deltas, forward pass, val loop) in ~2-5 min
# without needing AMP, torch.compile, or gradient checkpointing.
#
# Usage
#   chmod +x run_subset_cloud.sh
#   ./run_subset_cloud.sh
#
# Switch any ENCODER / INDEX_MODE option below to smoke-test a variant
# before committing to a full 100-epoch run.

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"   # ← update if needed
SAVE_DIR="checkpoints/smoke_test"
LOCAL_CACHE="/tmp/station_mae_cache"

# ── Station exclusion ────────────────────────────────────────────────────────
EXCLUDE="--exclude_stations 110"

# ── Encoder architecture (mirror run_full_cloud.sh choice here) ──────────────
ENCODER="--factorised_encoder"                     # full axial (temporal + spatial)
# ENCODER="--factorised_encoder --no_spatial_attn"  # temporal-only
# ENCODER="--joint_encoder"                         # joint spatiotemporal + RoPE
# ENCODER=""                                         # flat

TEMPORAL_WINDOW=""

# ── Window sampling strategy ──────────────────────────────────────────────────
INDEX_MODE="--index_mode sliding --train_stride 4"
# INDEX_MODE="--index_mode blocks"
# INDEX_MODE="--index_mode random"

# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --delta_mode       fixed_grid \
    --delta_grid_stride 3 \
    --mlp_ratio        2.0 \
    --d_model          64 \
    --enc_layers       2 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.0 \
    --drop_path_rate   0.0 \
    --batch_size       8 \
    --num_workers      2 \
    --epochs           1 \
    --limit_train_batches 50 \
    --lr               1e-4 \
    --warmup_epochs    0 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --patience         0 \
    --min_lr           1e-6 \
    --cross_attn_decoder \
    $ENCODER \
    $TEMPORAL_WINDOW \
    $INDEX_MODE \
    $EXCLUDE \
    --save_dir         "$SAVE_DIR"
