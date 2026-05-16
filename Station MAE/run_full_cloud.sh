#!/usr/bin/env bash
# run_full_cloud.sh — optimised full training run for A100 80GB PCIe (MIG 3g.20gb)
#
# Hardware context
# ----------------
# The A100 is running in MIG mode. The allocated partition is 3g.20gb:
#   • 19,968 MiB (~20 GB) VRAM  — NOT the full 80 GB
#   • 28 / 108 streaming multiprocessors
#   • No NVLink across MIG instances → single-GPU only
#
# Key flags:
#
#   CHANGE 1 — bfloat16 AMP  (--amp --bf16)
#     A100 has native BF16 tensor cores. Same speed as FP16 but wider
#     dynamic range — no loss scaling, more numerically stable.
#
#   CHANGE 2 — torch.compile default mode  (--compile)
#     MIG partitions restrict CUDA graph capture, so "reduce-overhead" mode
#     would silently fail. "default" mode still fuses ops and removes Python
#     overhead (~1.3-1.5× speedup), no CUDA graphs needed.
#     Warm-up cost: ~1-2 min on the first epoch.
#
#   CHANGE 3 — fast local cache  (--local_cache_dir /tmp/station_mae_cache)
#     Saves split-normalised tensors as numpy mmap files on /tmp (tmpfs).
#     Workers mmap directly from OS page cache — no IPC queue overhead for
#     source data. First run writes files; all subsequent runs are instant.
#
#   CHANGE 4 — gradient checkpointing  (--grad_checkpoint)
#     With only 20 GB VRAM, grad checkpointing trades ~33% extra compute
#     for ~66% less activation memory, preventing OOM.
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
#     At W=144, tw=6 gives 24 one-hour chunks. Score computation drops 24×
#     (still modest savings vs FFN/QKV, but worthwhile at this window size).
#     At W=288, tw=6 gives 48 chunks — savings become substantial.
#
# Resuming an interrupted run:
#   ./run_full_cloud.sh   (Lightning restores from last.ckpt automatically)
#
# Usage:
#   chmod +x run_full_cloud.sh
#   ./run_full_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"   # ← update this
SAVE_DIR="checkpoints/full_run_cloud"
LOCAL_CACHE="/tmp/station_mae_cache"

# ── Station exclusion ────────────────────────────────────────────────────────
# Drop stations with insufficient historical coverage before training.
# Matched case-insensitively against the stations_table index / name / abbr.
# Add more names separated by spaces: EXCLUDE="110 BAS KLO"
EXCLUDE="--exclude_stations 110"
# EXCLUDE=""   # ← uncomment to disable exclusion
# ─────────────────────────────────────────────────────────────────────────────

# ── Optional encoder flags ────────────────────────────────────────────────────
# Uncomment to remove spatial attention (~27% faster per encoder block):
# SPATIAL=""
SPATIAL="" #"--no_spatial_attn"

# Uncomment to enable local windowed temporal attention:
# W=144 / tw=6 → 24 one-hour chunks; score computation drops 24×.
# TEMPORAL_WINDOW="--temporal_window 6"
TEMPORAL_WINDOW=""

#--masked_only_loss \        # pure gap-filling objective
# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --num_delta        6 \
    --mlp_ratio        2.0 \
    --d_model          128 \
    --enc_layers       6 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.15 \
    --train_stride     4 \
    --drop_path_rate   0.10 \
    --batch_size       32 \
    --num_workers      5 \
    --epochs           100 \
    --lr               1e-4 \
    --warmup_epochs    5 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --val_check_interval 500 \
    --save_every       2 \
    --patience         50 \
    --min_lr           1e-6 \
    --amp \
    --bf16 \
    --compile \
    --grad_checkpoint \
    --factorised_encoder \
    --cross_attn_decoder \
    $SPATIAL \
    $TEMPORAL_WINDOW \
    $EXCLUDE \
    --wandb_project    station-mae \
    --wandb_run_name   baseline-cloud \
    --save_dir         "$SAVE_DIR"

# ─── Sub-epoch validation + checkpointing (--val_check_interval) ─────────────
#
# One full epoch on this config takes ~50 min.  To get validation feedback and
# checkpoint saves every ~30 min instead of every epoch, add these flags:
#
#   --val_check_interval 4000   # run val every 4000 training steps (~30 min)
#   --save_every_steps   4000   # save step_NNNNNNN.ckpt at the same interval
#   --patience           3      # stop after 3 checks (~90 min) without improvement
#   --min_lr             1e-6   # LR floor: cosine decays to 1e-6 not 0
#
# With batch_size=16 and ~105K train samples → ~6562 steps/epoch.
# 4000 steps ≈ 0.61 epochs ≈ 30 min.  Adjust if your step-time differs.
# EarlyStopping patience now counts "validation checks" not epochs, so
# patience=3 → stop after ~90 min without improvement.

# ─── If still OOM: switch to window=72 (12 h context) ────────────────────────
# Replace --window 144 with --window 72 and remove --grad_checkpoint.
# Temporal attention drops 4×, decoder KV halves, frees ~3-4 GB VRAM.
# The model still forecasts up to 6 h ahead via DeltaTimeEmbedding.
# Epoch time roughly halves compared to window=144 without grad_checkpoint.