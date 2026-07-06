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
#     At W=72, tw=6 gives 12 one-hour chunks. Score computation drops 12×
#     (still modest savings vs FFN/QKV, but worthwhile at this window size).
#
# Resuming an interrupted run:
#   ./run_full_cloud.sh   (Lightning restores from last.ckpt automatically)
#
# Usage:
#   chmod +x run_full_cloud.sh
#   ./run_full_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"

# Checkpoints saved inside the project directory — guaranteed writable on Renku.
# Saving to /home/renku/work/ sub-directories outside this project can silently
# fail due to filesystem permissions on the Renku mount.
# Saving directly to Polybox during training is unreliable (network latency,
# lock contention) and has been removed — copy checkpoints to Polybox manually
# after training with:
#   cp checkpoints/full_run_cloud/best.ckpt /home/renku/work/polybox-capstone/checkpoints/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_DIR="${SCRIPT_DIR}/checkpoints/full_run_cloud"
LOCAL_CACHE="/tmp/station_mae_cache"

# ── Station exclusion ────────────────────────────────────────────────────────
# Drop stations with insufficient historical coverage before training.
# Matched case-insensitively against the stations_table index / name / abbr.
# Add more names separated by spaces: EXCLUDE="110 BAS KLO"
EXCLUDE="--exclude_stations PFA"   # station 110 (PFA) — insufficient historical coverage
# EXCLUDE="--exclude_stations 110"  # alternative: match by numeric ID
# EXCLUDE=""                         # ← uncomment to disable exclusion
# ─────────────────────────────────────────────────────────────────────────────

# ── Encoder architecture ──────────────────────────────────────────────────────
#
# Three mutually exclusive encoder modes (choose ONE):
#
#   JOINT encoder  (--joint_encoder)
#     Full self-attention over all W×N tokens simultaneously.
#     Temporal RoPE on Q/K — captures cross-station × cross-time interactions
#     in every layer. Flash Attention keeps VRAM linear.  Slower than factorised
#     but richer representations.  Strongly recommended with --grad_checkpoint.
#     At W=72, N_vis≈75: L≈5400 tokens, ~2.5× slower than factorised.
#     Disable --factorised_encoder and --no_spatial_attn when using this.
#
#   FACTORISED encoder  (--factorised_encoder, current default)
#     Alternates temporal then spatial attention — ~100× cheaper than flat.
#     --no_spatial_attn removes the spatial sub-layer (station-independent).
#
#   FLAT encoder  (neither flag)
#     Standard self-attention over flattened W·N_vis tokens.

# Set ENCODER to one of the three options below:
# ENCODER="--factorised_encoder"                     # default: full axial (temporal + spatial)
# ENCODER="--factorised_encoder --no_spatial_attn"  # temporal-only (station-independent encoder)
# ENCODER="--joint_encoder"                         # joint spatiotemporal + RoPE
ENCODER=""                                         # flat self-attention over W·N tokens

# ── Temporal window (flat or factorised encoder) ─────────────────────────────
#
# Splits W timesteps into non-overlapping chunks of size tw.
# Odd layers use a Swin-style half-window shift for cross-chunk communication.
#
# Flat encoder (current default):
#   Full cross-station attention within each tw×N_vis chunk.
#   W=72, tw=6, N_vis≈65 → 390 tokens/chunk  (144× cheaper than full flat)
#
TEMPORAL_WINDOW="--temporal_window 6"
# TEMPORAL_WINDOW="--temporal_window 12"  # previous setting
# TEMPORAL_WINDOW=""   # ← disable windowing (full attention)

# ── Window sampling strategy ──────────────────────────────────────────────────
#
# Training uses index_mode=random — each __getitem__ draws independently
# and uniformly from the full pool of ~260k valid windows (with replacement).
# The station mask is re-drawn per sample, so revisiting the same time window
# in a later step is NOT redundant — a different 50% of stations are masked,
# creating a genuinely different training task.
#
# Delta grid: --delta_mode fixed_grid --delta_grid_stride 3 --max_delta 36
#   K=13 horizons: [0, 3, 6, …, 36] steps = [now, 30min, 1h, …, 6h]
#   k=0 (delta=0): inpainting — loss on MASKED stations only
#   k=1..12 (delta=3..36): forecasting — loss on ALL stations
#   Decoder step indices: W-1+delta = [71, 74, 77, …, 107] (unified timeline)
#
# Epoch size rationale
# --------------------
# The non-overlapping block count for 5 training years is:
#   pool (~260,000) / W (72) ≈ 3,611 windows
# This is the minimum epoch size that gives one "effective pass" over the data.
# We set random_epoch_size=10000 (≈ 2.8× non-overlapping) as the baseline.
# Each revisit has a different random station mask, so it is NOT redundant.
#
# This is a 4× reduction from the previous 40,000-step epochs, which was
# 11× redundant within each epoch and led to:
#   • 3h/epoch → very coarse LR scheduling and monitoring granularity
#   • Only ~33 epochs in a 100h budget
# With 10,000 steps/epoch (~2500 batches at batch_size=4):
#   • ~50 min/epoch → fine-grained LR decay and early stopping
#   • ~120 epochs in a 100h budget
#
# Validation uses non-overlapping blocks (hardcoded in main.py):
#   Val year 2022 → ~728 windows → ~23 batches at batch_size=4 → ~6 s/epoch
#   No --limit_val_batches needed — the full val set is evaluated every epoch.
#
INDEX_MODE="--index_mode random --random_epoch_size 10000"
# INDEX_MODE="--index_mode random --random_epoch_size 5000"  # ~1.4× non-overlapping (fast ablation)
# INDEX_MODE="--index_mode random --random_epoch_size 3611"  # exact 1× non-overlapping
# INDEX_MODE="--index_mode random --random_epoch_size 40000" # previous (11× redundant)
# INDEX_MODE="--index_mode blocks"                           # non-overlapping train (fast ablation)
# INDEX_MODE="--index_mode sliding --train_stride 4"         # sliding, hourly stride

# ── Loss function ─────────────────────────────────────────────────────────────
#
# v10 loss: Huber(δ=1.0) for ALL variables, with per-variable weights.
#
#   Loss = (1/K) Σ_k  Σ_v  w_v · Huber(ŷ_kvn − y_kvn, δ=1.0)
#
# Weights [temperature=1.0, pressure=0.5, humidity=0.8, wind_u=1.5, wind_v=1.5]:
#   • Per-station normalisation already handles scale (std≈1 per variable).
#   • Weights correct for predictability imbalance: pressure is easy (0.5×),
#     wind is hard (1.5×).  Temperature at 1.0 — primary monitoring variable.
#   • δ=1.0 in normalised space = 1 std-dev: L2 for typical errors,
#     L1 (capped gradient) for extreme events.
#   • No σ² head — simpler decoder, stable gradients, no exploitation of
#     inflated uncertainty to artificially reduce NLL.
#   • Precipitation excluded from targets (num_target_vars=5).
#
# Previous v9 used --nll_loss (heteroscedastic Gaussian NLL). Removed here
# because: σ² head can reduce loss without improving point predictions;
# Gaussian is a poor fit for some variables; and RMSE is the primary metric.
#
# ─────────────────────────────────────────────────────────────────────────────

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --delta_mode       fixed_grid \
    --delta_grid_stride 3 \
    --mlp_ratio        4.0 \
    --d_model          1024 \
    --enc_heads        16 \
    --dec_heads        16 \
    --enc_layers       8 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.0 \
    --drop_path_rate   0.0 \
    --batch_size       4 \
    --num_workers      3 \
    --epochs           300 \
    --lr               1e-4 \
    --warmup_epochs    15 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --accumulate_grad_batches 4 \
    --input_context_cross_attn \
    --patience         40 \
    --min_lr           5e-7 \
    --amp \
    --bf16 \
    --compile \
    --grad_checkpoint \
    --cross_attn_decoder \
    $ENCODER \
    $TEMPORAL_WINDOW \
    $INDEX_MODE \
    $EXCLUDE \
    --wandb_project    station-mae \
    --wandb_run_name   tw6-d1024-v10 \
    --save_dir         "$SAVE_DIR"
# NOTE: --polybox_dir removed — Polybox writes during training are unreliable.
# After training finishes, manually copy checkpoints:
#   cp "$SAVE_DIR/best.ckpt" /home/renku/work/polybox-capstone/checkpoints/tw6-d1024-v9-best.ckpt
