#!/usr/bin/env bash
# run_full_cloud.sh — full training run, A100 80GB PCIe (MIG 3g.20gb)
#
# CURRENT CONFIG (v13): patched encoder — P=6, full attention, d_model=512,
# 16 layers. The v9/v11/v12 settings are kept as commented alternatives.
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
# Resuming an interrupted or early-stopped run:
#   bash src/scripts/run_full_cloud.sh          (auto-resumes from last.ckpt)
#   RESUME=0 bash src/scripts/run_full_cloud.sh (force a fresh run)
#
# This is now real. It used to say Lightning resumed automatically -- it does
# NOT. main.py only resumes when --resume is passed, so a plain re-run started
# from random init AND wrote into the same SAVE_DIR, which is what produced the
# best-v1.ckpt / last-v1.ckpt pairs in full_run_cloud_v12.
#
# Usage:
#   bash src/scripts/run_full_cloud.sh

# ═══════════════════════════════════════════════════════════════════════════
# WHAT CHANGED vs v12, AND WHY
# ═══════════════════════════════════════════════════════════════════════════
#
# 1. TEMPORAL PATCHING  --temporal_patch 6   (was: none)
#    Six consecutive 10-min steps become ONE token, so the encoder sees 12
#    temporal positions instead of 72.
#    Motivation: measured receptive field. At W=72 with tw=6 windowed attention
#    and 8 layers, the NEWEST token could only reach 4 h of the 12 h window.
#    Full coverage would have needed 12 layers.
#
# 2. WINDOWING RETIRED  --temporal_window 0  (was: 6)
#    Windowing only existed to make 72x78 = 5,616 tokens affordable. After
#    patching there are 12x78 = 936, and FULL attention over those costs
#    ~0.88M score entries per layer versus ~2.63M for the windowed setup.
#    Cheaper AND every token sees the whole window from layer 1.
#    It also removes the unmasked circular shift (see encoder.py) that let
#    t=71 attend to t=0 in v9/v11/v12.
#
# 3. d_model 1024 -> 512, enc_layers 8 -> 16
#    Attention/FFN cost scales as d^2, so halving d buys 4x the layers at equal
#    FLOPs. Depth is what the receptive-field analysis said was missing.
#    No information bottleneck: each patched token carries 6 steps x 6 vars =
#    36 numbers, so 512 is still a ~14x expansion.
#
# 4. REGULARISATION REMOVED  --dropout 0.0 --drop_path_rate 0.0  (was 0.1/0.1)
#    v12 added dropout, drop-path and early stopping to a model whose problem
#    is that it IGNORES its inputs and predicts the conditional mean. Those are
#    all anti-overfitting measures applied to an underfitting failure — they
#    push the model further toward the mean. Removed until it can beat
#    persistence; re-introduce only if a genuine train/val gap appears.
#
# 5. OverfitEarlyStop now respects warmup (fixed in main.py).
#    v12 ran --overfit_patience 5 against --warmup_epochs 15, so it could stop
#    before the LR ever reached peak. v12's result should be treated as void.
#
# 6. Nearest-station normalisation for sparse (station,variable) pairs
#    (data/dataset.py). GES pressure has ZERO training observations and used to
#    borrow the altitude-blind GLOBAL mean; it now borrows from the nearest
#    station at a comparable elevation.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"

# Checkpoints saved inside the project directory — guaranteed writable on Renku.
# Saving to /home/renku/work/ sub-directories outside this project can silently
# fail due to filesystem permissions on the Renku mount.
# Saving directly to Polybox during training is unreliable (network latency,
# lock contention) and has been removed — copy checkpoints to Polybox manually
# after training with:
#   cp checkpoints/full_run_cloud/best.ckpt /home/renku/work/polybox-capstone/checkpoints/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src   — python entry points live here
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root — checkpoints/, test_results/, report/
cd "${SRC_DIR}"                                              # so `python main.py` and `from data...` resolve
SAVE_DIR="${PROJ_DIR}/checkpoints/full_run_cloud_v13"   # own dir — never share a SAVE_DIR (that is what produced the -v1 checkpoints)
LOCAL_CACHE="/tmp/station_mae_cache"

# ── WandB ────────────────────────────────────────────────────────────────────
# Defaults to ONLINE and prompts for `wandb login` if no credential is present,
# so a run never silently produces an empty dashboard.
#
# Caveat kept from earlier: Renku egress can drop for minutes at a time, and the
# wandb server then marks the run "crashed" (frozen at the last epoch it
# received) even though training continues normally. If that happens, re-launch
# with WANDB_MODE=offline and sync afterwards:
#   WANDB_MODE=offline bash src/scripts/run_full_cloud.sh
#   wandb sync "$(ls -dt wandb/run-* | head -1)"
source "${SCRIPT_DIR}/_wandb_preflight.sh"

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
TEMPORAL_WINDOW="--temporal_window 0"   # windowing retired — patching replaces it
# TEMPORAL_WINDOW="--temporal_window 6"   # v9/v11/v12 setting (unmasked shift!)
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
#INDEX_MODE="--index_mode random --random_epoch_size 10000"
# INDEX_MODE="--index_mode random --random_epoch_size 5000"  # ~1.4× non-overlapping (fast ablation)
# INDEX_MODE="--index_mode random --random_epoch_size 3611"  # exact 1× non-overlapping
# INDEX_MODE="--index_mode random --random_epoch_size 40000" # previous (11× redundant)
# INDEX_MODE="--index_mode blocks"                           # non-overlapping train (fast ablation)
INDEX_MODE="--index_mode sliding --train_stride 9"         # sliding, hourly stride

# ── Loss function ─────────────────────────────────────────────────────────────
#
# v12 loss: Huber(δ=1.0) for ALL variables — SAME as v11, only the per-variable
# WEIGHTS change.  Clean A/B on the weights alone (δ, loss family unchanged).
#
#   Loss = (1/K) Σ_k  Σ_v  w_v · Huber(ŷ_kvn − y_kvn, δ=1.0)  /  Σ_v w_v
#
# Weights [T=1.0, P=1.0, RH=0.7, u=0.5, v=0.5] (set in model/mae.py):
#   • v11 (uniform [1,1,1,1,1]) wandb curves: temperature & pressure keep
#     improving and do NOT overfit; humidity + wind_u/v bottom ~epoch 30-40 then
#     degrade.  So DOWN-weight the noisy channels (opposite of v10, which
#     UP-weighted wind 1.5× and overfit worse) → encoder spends less capacity
#     fitting unpredictable gust / saturation noise.
#   • Temperature/pressure held at 1.0 → their gradient is unchanged vs v11,
#     keeping the comparison interpretable.
#   • δ=1.0 in normalised space = 1 std-dev: L2 for typical errors,
#     L1 (capped gradient) for extreme events.
#     If wind/humidity still overfit, next loss-only lever: δ=0.5.
#   • No σ² head — simpler decoder, stable gradients, no exploitation of
#     inflated uncertainty to artificially reduce NLL.
#   • Precipitation excluded from targets (num_target_vars=5).
#
# v9 (= saved best.ckpt, all current test results) used --nll_loss
# (heteroscedastic Gaussian NLL with σ² head, uniform variable weights).
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Batch size (reviewed 2026-07-31) ─────────────────────────────────────────
# LEFT AT 4 x 4 = effective batch 16, deliberately.
#
# The slice (A100-80GB MIG 3g.20gb, 19,968 MiB, 28/108 SMs) is the same one
# these settings were tuned on, and this run already uses --grad_checkpoint
# *because* 20 GB is tight. Training memory is dominated by activations and
# optimizer state, so doubling batch_size here is a real OOM risk — and an OOM
# several hours into a multi-day run is expensive.
#
# If you want the speedup, 8 x 2 keeps the EFFECTIVE batch at 16, so the
# optimisation problem is unchanged and the run stays comparable to v9/v11/v12:
#     --batch_size 8 --accumulate_grad_batches 2
# Test it first with --epochs 1 --random_epoch_size 200 and watch nvidia-smi.
# Never raise the effective batch itself without revisiting the learning rate.
#
# Inference is different: test.py holds no activations or optimizer state, so
# run_test_cloud.sh is safe at batch_size 8 on this slice.
# ── Resume ───────────────────────────────────────────────────────────────────
# If last.ckpt exists in SAVE_DIR, continue from it: full training state
# (weights, optimiser, LR schedule, epoch counter) is restored.
#
# Without this, re-running would start from scratch and ModelCheckpoint would
# refuse to overwrite the existing files, silently creating best-v1.ckpt --
# leaving two runs in one folder with no indication which is which.
#
# NOTE: OverfitEarlyStop keeps no state_dict, so its patience counter resets to
# zero on resume. That is deliberate here (it gives a stopped run room to
# continue) but it means patience alone will not protect a resumed run --
# set --overfit_patience appropriately.
RESUME="${RESUME:-1}"
RESUME_ARG=()   # array, NOT a string: the project path contains a space
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
        echo "[resume]          A fresh run will write best-v1.ckpt alongside it."
        echo "[resume]          Use a new SAVE_DIR instead."
        exit 1
    fi
fi

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --delta_mode       fixed_grid \
    --delta_grid_stride 3 \
    --mlp_ratio        4.0 \
    --d_model          512 \
    --enc_heads        8 \
    --dec_heads        8 \
    --enc_layers       16 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --temporal_patch   6 \
    --dropout          0.0 \
    --drop_path_rate   0.0 \
    --batch_size       4 \
    --num_workers      3 \
    --epochs           100 \
    --lr               1e-4 \
    --warmup_epochs    15 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --accumulate_grad_batches 4 \
    --input_context_cross_attn \
    --patience         40 \
    --monitor          val/overall_rmse \
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
    --wandb_run_name   patch6-d512-L16-v13 \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
    --save_dir         "$SAVE_DIR"
# WandB runs ONLINE (default). If Renku blocks outbound connections, add
# --wandb_offline above and sync later with: wandb sync <run-dir>
# NOTE: --polybox_dir removed — Polybox writes during training are unreliable.
# After training finishes, manually copy checkpoints:
#   cp "$SAVE_DIR/best.ckpt" /home/renku/work/polybox-capstone/checkpoints/tw6-d1024-v12-best.ckpt
