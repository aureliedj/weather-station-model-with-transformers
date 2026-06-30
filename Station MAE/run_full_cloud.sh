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
#   Recommended d_model: 512  mlp_ratio: 4.0
#   TEMPORAL_WINDOW="--temporal_window 6"
#
# Factorised encoder:
#   Windowed temporal-only attention within each station's tw-step window.
#   W=72, tw=6 → 12 one-hour chunks, 12× cheaper temporal sub-layer.
#   TEMPORAL_WINDOW="--temporal_window 6"
#
TEMPORAL_WINDOW="--temporal_window 6"
# TEMPORAL_WINDOW="--temporal_window 12"  # previous setting
# TEMPORAL_WINDOW=""   # ← disable windowing (full attention)

# ── Window sampling strategy ──────────────────────────────────────────────────
#
# Controls which windows become training samples (val/test always use
# "sliding" with stride=1 for consistent evaluation).
#
#   sliding  (Strategy C — default, GraphDOP / most baselines)
#     Every contiguity-valid start, thinned by --train_stride.
#     DataLoader shuffle gives random-without-replacement epochs.
#     With W=72 and stride=4: windows every 40 min, ~94% overlap.
#     Maximum data coverage; use when overfitting is not a concern.
#
#   blocks   (Strategy B — PatchTST / iTransformer)
#     Greedy non-overlapping: no two windows share any input timestep.
#     W=72 → ~1 window per 12 h → ~3,500 train samples (vs ~260K sliding).
#     Cleanest gradient signal; useful for fast ablation runs.
#
#   random   (Strategy A — Aurora / W-MAE / VideoMAE)
#     Full pool stored; each __getitem__ samples independently at random
#     (with replacement).  Different windows every epoch, no systematic
#     coverage bias.  --train_stride is ignored in this mode.
#     Best default when combined with a large pool and many epochs.

INDEX_MODE="--index_mode random --random_epoch_size 4000"
# INDEX_MODE="--index_mode random --random_epoch_size 40000"
# ↑ 8000 samples / batch_size=4 = 2000 batches/epoch — stabler gradients than default ~914
# INDEX_MODE="--index_mode sliding --train_stride 4"
# INDEX_MODE="--index_mode blocks"

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
    --enc_layers       8 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --dropout          0.0 \
    --drop_path_rate   0.0 \
    --batch_size       4 \
    --num_workers      3 \
    --epochs           100 \
    --lr               1e-5 \
    --warmup_epochs    15 \
    --weight_decay     0.05 \
    --grad_clip        0.5 \
    --accumulate_grad_batches 4 \
    --input_context_cross_attn \
    --limit_val_batches 200 \
    --patience         15 \
    --monitor          val/temperature_rmse \
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
    --wandb_run_name   tw6-d1024-v5 \
    --save_dir         "$SAVE_DIR"
# NOTE: --polybox_dir removed — Polybox writes during training are unreliable.
# After training finishes, manually copy checkpoints:
#   cp "$SAVE_DIR/best.ckpt" /home/renku/work/polybox-capstone/checkpoints/tw6-d1024-v3-best.ckpt

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