#!/usr/bin/env bash
# run_profile_cloud.sh — mini profiling run for A100 80GB PCIe (MIG 3g.20gb)
#
# Purpose
# -------
# Runs a short (400-batch) training pass with PyTorch Lightning's
# PyTorchProfiler to identify CUDA kernel bottlenecks.
#
# Profiler output
# ---------------
# Chrome trace JSON → checkpoints/profile_run/profile/pytorch_trace*.json
# Open at https://ui.perfetto.dev → "Open trace file"
#
# The profiler uses a schedule of:
#   wait=5 warmup=5 active=20  (first 30 batches)
# so torch.compile JIT warm-up is excluded from the trace.  The active
# window captures 20 representative batches — enough for stable kernel stats.
#
# Interpreting the trace
# ----------------------
# Look for:
#   • Wide CUDA kernels with no overlap → compute-bound, consider smaller model
#   • Gaps between kernels → memory transfers or CPU overhead
#   • aten::copy_ spikes → H2D data transfer latency (DataLoader bottleneck)
#   • cudnn / flash_attn kernels → attention cost breakdown
#
# Usage
# -----
#   chmod +x run_profile_cloud.sh
#   ./run_profile_cloud.sh
#   # Then open the JSON at https://ui.perfetto.dev

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"   # ← update this
SAVE_DIR="$(cd "$(dirname "$0")" && pwd)/checkpoints/profile_run"
LOCAL_CACHE="/tmp/station_mae_cache"

EXCLUDE="--exclude_stations 110"

# Same encoder settings as full run to get representative timings
SPATIAL="" #"--no_spatial_attn"
TEMPORAL_WINDOW="" #"--temporal_window 6"

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        0 \
    --num_delta        1 \
    --mlp_ratio        2.0 \
    --d_model          128 \
    --enc_layers       6 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.1 \
    --batch_size       16 \
    --num_workers      5 \
    --epochs           1 \
    --limit_train_batches 400 \
    --lr               1e-4 \
    --warmup_epochs    1 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --patience         0 \
    --min_lr           1e-6 \
    --amp \
    --bf16 \
    --compile \
    --grad_checkpoint \
    --factorised_encoder \
    --cross_attn_decoder \
    --profiler         pytorch \
    $SPATIAL \
    $TEMPORAL_WINDOW \
    $EXCLUDE \
    --wandb_project    station-mae \
    --wandb_run_name   profile-run \
    --save_dir         "$SAVE_DIR"

echo ""
echo "─────────────────────────────────────────────────────────"
echo "  Profiling complete."
echo "  Chrome trace: $SAVE_DIR/profile/pytorch_trace*.json"
echo "  Open at: https://ui.perfetto.dev  → Open trace file"
echo "─────────────────────────────────────────────────────────"
