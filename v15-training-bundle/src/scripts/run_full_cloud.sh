#!/usr/bin/env bash
# run_full_cloud.sh — Station-MAE v15 training run.
#
# Model:    masked autoencoder over a (station x time) grid. The encoder sees a
#           random subset of stations for a 12 h window; a delta-conditioned
#           query decoder predicts all 155 stations at K=13 lead times in one
#           forward pass.
# Defaults: 30-min tokens (patch 3), factorised encoder, d_model 384,
#           8 encoder / 2 decoder layers, mask ratio 0.5, Huber(1.0),
#           uniform variable weights, 100 epochs.
#
# Hardware this is tuned for: A100 80GB in MIG mode, partition 3g.20gb
#   ~20 GB VRAM (not 80), 28/108 SMs, single-GPU only (no NVLink across MIG).
#   --grad_checkpoint and the small batch below are both consequences of the
#   20 GB ceiling. --amp --bf16 uses the A100's native BF16 tensor cores
#   (no loss scaling needed). --compile is worth ~1.3x; it is a speedup, not a
#   correctness requirement, and MIG blocks CUDA-graph capture so "default"
#   mode is used rather than "reduce-overhead".
#
# Usage:
#   bash src/scripts/run_full_cloud.sh              # full run, auto-resumes
#   SANITY=1 bash src/scripts/run_full_cloud.sh     # ~5 min  — does it RUN?
#   SANITY=2 bash src/scripts/run_full_cloud.sh     # ~1 h    — does it LEARN?
#   RESUME=0 bash src/scripts/run_full_cloud.sh     # force a fresh run

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"
LOCAL_CACHE="/tmp/station_mae_cache"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src — entry points
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root
cd "${SRC_DIR}"                                              # so `from data...` resolves

# Checkpoints go inside the project directory (guaranteed writable). Give every
# run its OWN SAVE_DIR — sharing one makes Lightning write best-v1.ckpt
# alongside best.ckpt, leaving two runs in a folder with no way to tell them apart.
SAVE_DIR="${PROJ_DIR}/checkpoints/full_run_cloud_v15"

# Verifies CUDA actually initialises. A driver/wheel mismatch does not raise —
# it silently trains on CPU. Run scripts/fix_torch_cuda.sh if this complains.
source "${SCRIPT_DIR}/_cuda_preflight.sh"

# WandB defaults to ONLINE and prompts for `wandb login` if no credential is
# found, so a run never silently produces an empty dashboard. If the network
# drops mid-run the server marks it "crashed" while training continues; in that
# case re-launch with WANDB_MODE=offline and sync afterwards:
#   wandb sync "$(ls -dt wandb/run-* | head -1)"
source "${SCRIPT_DIR}/_wandb_preflight.sh"

# ── Station exclusion ────────────────────────────────────────────────────────
# Dropped before training. Matched case-insensitively against the
# stations_table index / name / abbreviation, or by numeric ID.
EXCLUDE="--exclude_stations PFA"    # insufficient historical coverage
# EXCLUDE=""                        # keep all stations

# ── Temporal tokenization ────────────────────────────────────────────────────
# P consecutive 10-min steps are merged into ONE encoder token. P must divide
# --window (72): 1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72.
#
#   P=1  raw      72 positions -> 5,616 tokens  (~273 GFLOP/pass)
#   P=3  30-min   24 positions -> 1,872 tokens  (~48 GFLOP)   <- default
#   P=6  hourly   12 positions ->   936 tokens  (~19 GFLOP)
#
# P=3 makes a token exactly 30 min, so the +30 min validation forecast is one
# token ahead rather than half a token. P=1 has collapsed in past runs: softmax
# over 5,616 keys starts near-uniform, the model settles on predicting the mean,
# and val/horizon_sensitivity falls toward its 0.05 floor. Change one variable
# at a time here — this flag alone walks the tokenization ablation.
PATCH=3

# ── Encoder attention mode ───────────────────────────────────────────────────
# Three mutually exclusive modes — pick exactly ONE.
#
#   FACTORISED (--factorised_encoder)      alternates temporal then spatial
#     attention over the W x N grid; O(N*W^2 + W*N^2). Add --no_spatial_attn to
#     drop the spatial sub-layer entirely (station-independent encoder, ~27%
#     cheaper per block, all cross-station reasoning left to the decoder).
#   JOINT (--joint_encoder)                full attention over all W x N tokens
#     at once with temporal RoPE on Q/K. Richest, ~2.5x slower than factorised.
#     Use --grad_checkpoint with it.
#   FLAT (neither flag)                    standard self-attention over the
#     flattened W*N_vis sequence.
ENCODER="--factorised_encoder"
# ENCODER="--factorised_encoder --no_spatial_attn"
# ENCODER="--joint_encoder"
# ENCODER=""

# ── Temporal windowing (flat or factorised encoder) ──────────────────────────
# Splits the W timesteps into non-overlapping chunks of size tw; odd layers
# apply a Swin-style half-window shift so information crosses chunk boundaries
# after two layers. W must be divisible by tw.
#
# Off by default: temporal patching replaces it. At P=3 the sequence is already
# short enough that full attention is cheaper than windowing AND every token
# sees the whole 12 h window from layer 1. Note the shift is applied WITHOUT a
# Swin attention mask (see encoder.py), so with windowing on, the newest
# timesteps can attend to the oldest across the window seam.
TEMPORAL_WINDOW="--temporal_window 0"
# TEMPORAL_WINDOW="--temporal_window 6"    # 12 one-hour chunks at W=72

# ── Window sampling ──────────────────────────────────────────────────────────
# index_mode=random draws each window independently from the ~260k valid pool.
# The station mask is re-drawn per sample, so revisiting a time window is NOT
# redundant — a different 50% of stations are hidden, a genuinely different task.
#
# Epoch size: the non-overlapping block count is ~260,000 / 72 ~= 3,611 windows,
# so 10,000 is ~2.8 effective passes. At batch_size 4 that is ~2,500 optimiser
# steps and roughly 50 min/epoch, which keeps LR decay and early stopping
# fine-grained. Validation uses non-overlapping blocks over 2022 (~728 windows,
# a few seconds) — the full val set runs every epoch, no --limit_val_batches.
INDEX_MODE="--index_mode random --random_epoch_size 10000"
# INDEX_MODE="--index_mode blocks"
# INDEX_MODE="--index_mode sliding --train_stride 9"

# ── Delta grid (fixed below, documented here) ────────────────────────────────
# --delta_mode fixed_grid --delta_grid_stride 3 --max_delta 36 gives K=13
# horizons [0, 3, ..., 36] steps = [now, 30 min, 1 h, ..., 6 h].
#   k=0     inpainting  — loss on MASKED stations only
#   k=1..12 forecasting — loss on ALL stations
# Decoder step indices are W-1+delta = [71, 74, ..., 107], continuing the
# encoder's 0..71 timeline so the two share one integer coordinate system.

# ── Sanity runs ──────────────────────────────────────────────────────────────
# Both write to <SAVE_DIR>-sanity, so they can never collide with a real run.
#
# SANITY=1 — does it RUN?   2 epochs x 200 windows, subset years.
#            Check: no shape/OOM error, "[cuda] ok" at the top, finite val/loss,
#            one sanity/* line per epoch, checkpoint written.
# SANITY=2 — does it LEARN? 15 epochs x 3,600 windows on a 2-year subset.
#            Check: val/overall_mae falling; val/horizon_sensitivity RISING off
#            its 0.05 floor toward 0.2+; sanity/dispersion_min climbing past
#            ~0.3. Together those say the model has left the constant-output
#            basin and is using the delta embedding. FALLING horizon_sensitivity
#            is the collapse signature — kill the job and re-tune, don't wait.
#            sanity/ctx_ratio < 0.9 means visible stations beat masked ones at
#            delta=0, i.e. recent observations are reaching the decoder; it is
#            only meaningful once dispersion is up, since a constant predictor
#            scores exactly 1.00 by arithmetic.
SANITY="${SANITY:-0}"
SANITY_ARGS=()
if [[ "$SANITY" == "1" ]]; then
    SANITY_ARGS=(--epochs 2 --random_epoch_size 200 --subset
                 --warmup_epochs 1 --patience 100 --val_check_interval 0)
    INDEX_MODE="--index_mode random"
    SAVE_DIR="${SAVE_DIR}-sanity"; RUN_SUFFIX="-sanity1"
elif [[ "$SANITY" == "2" ]]; then
    SANITY_ARGS=(--epochs 15 --random_epoch_size 3600 --subset --warmup_epochs 2)
    INDEX_MODE="--index_mode random"
    SAVE_DIR="${SAVE_DIR}-sanity"; RUN_SUFFIX="-sanity2"
else
    RUN_SUFFIX=""
fi
if [[ "$SANITY" != "0" ]]; then
    echo "[sanity] SANITY=${SANITY} — short run into ${SAVE_DIR}"
    echo "[sanity]   ${SANITY_ARGS[*]}"
    echo "[sanity]   this is NOT the real run; unset SANITY for the full budget."
fi

# ── Resume ───────────────────────────────────────────────────────────────────
# main.py only resumes when --resume is passed; a plain re-run would otherwise
# start from random init AND write into the same SAVE_DIR. If last.ckpt exists,
# full training state (weights, optimiser, LR schedule, epoch) is restored.
#
# OverfitEarlyStop keeps no state_dict, so its patience counter resets to zero
# on resume — deliberate (it gives a stopped run room to continue) but it means
# patience alone will not protect a resumed run.
RESUME="${RESUME:-1}"
RESUME_ARG=()   # array, NOT a string — the project path may contain a space
if [[ "$RESUME" == "1" && -f "${SAVE_DIR}/last.ckpt" ]]; then
    RESUME_ARG=(--resume "${SAVE_DIR}/last.ckpt")
    echo "[resume] continuing from ${SAVE_DIR}/last.ckpt"
    python - "$SAVE_DIR" <<'PYEOF'
import sys, os, torch
p = os.path.join(sys.argv[1], "last.ckpt")
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
    print(f"[resume]   epoch {c.get('epoch','?')}, global_step {c.get('global_step','?')}")
except Exception as e:
    print(f"[resume]   (could not read epoch: {type(e).__name__})")
PYEOF
elif [[ "$RESUME" == "1" ]]; then
    echo "[resume] no last.ckpt in ${SAVE_DIR} — starting fresh"
else
    echo "[resume] RESUME=0 — starting fresh"
    if [[ -f "${SAVE_DIR}/best.ckpt" ]]; then
        echo "[resume] WARNING: ${SAVE_DIR} already contains best.ckpt."
        echo "[resume]          A fresh run would write best-v1.ckpt alongside it."
        echo "[resume]          Use a new SAVE_DIR instead."
        exit 1
    fi
fi

# ── Launch ───────────────────────────────────────────────────────────────────
# Loss: Huber(delta=1.0) on all 5 target variables with uniform weights.
#   delta=1.0 in normalised space = 1 std dev — L2 for typical errors, L1
#   (capped gradient) for extremes. Precipitation is an input only, never a
#   target: its zero-inflated distribution makes a squared-error head a poor fit.
#   Uniform weights keep the per-variable comparison against the LSTM baseline
#   like-for-like. --nll_loss swaps in a heteroscedastic Gaussian head instead.
#
# Batch: 4 x 4 = effective 16. The 20 GB MIG slice is why. If you need speed,
# --batch_size 8 --accumulate_grad_batches 2 keeps the effective batch (and so
# the optimisation problem) identical. Never raise the effective batch without
# revisiting the learning rate.
python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --delta_mode       fixed_grid \
    --delta_grid_stride 3 \
    --mlp_ratio        4.0 \
    --d_model          384 \
    --enc_heads        8 \
    --dec_heads        8 \
    --enc_layers       8 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --temporal_patch   $PATCH \
    --var_weights      1.0 1.0 1.0 1.0 1.0 \
    --dropout          0.0 \
    --drop_path_rate   0.0 \
    --batch_size       4 \
    --num_workers      3 \
    --epochs           100 \
    --lr               1e-4 \
    --warmup_epochs    10 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --accumulate_grad_batches 4 \
    --patience         40 \
    --monitor          val/overall_mae \
    --overfit_stop \
    --overfit_patience 20 \
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
    --wandb_run_name   "patch${PATCH}-d384-L8-v15${RUN_SUFFIX}" \
    ${SANITY_ARGS[@]+"${SANITY_ARGS[@]}"} \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
    --save_dir         "$SAVE_DIR"

echo ""
echo "Done. Best checkpoint: ${SAVE_DIR}/best.ckpt"
