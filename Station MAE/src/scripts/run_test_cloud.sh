#!/usr/bin/env bash
# run_test_cloud.sh — prediction dump on the cloud (predictions-only)
#
# Loads the best checkpoint and runs ONE forward pass over the sliding test
# windows per mask ratio, saving raw predictions only:
#
#   test_results/best_mr0.00/predictions.pt   (all stations visible)
#   test_results/best_mr0.50/predictions.pt   (50% masked; masked_idx included)
#
# NO metrics are computed here — persistence, skill, per-station, seasonal etc.
# are all derived downstream from predictions.pt in Test_Results_Exploration.ipynb.
# Same schema as the LSTM dump (test_lstm.py), so runs drop into the same
# comparison cells.
#
# All architecture settings are auto-read from the checkpoint.
#
# Usage:
#   chmod +x run_test_cloud.sh
#   ./run_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"

# Resolve paths relative to this script so they work from any working directory.
# Checkpoints are saved by run_full_cloud.sh inside the project directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src   — python entry points live here
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root — checkpoints/, test_results/, report/
cd "${SRC_DIR}"                                              # so `python main.py` and `from data...` resolve

# Fail fast if torch cannot use the GPU (wheel/driver mismatch on the older node).
source "${SCRIPT_DIR}/_cuda_preflight.sh"
# ── Which run to evaluate ────────────────────────────────────────────────────
# Set RUN_NAME only. The checkpoint path and the output folder are both derived
# from it, so they cannot drift apart — editing one and forgetting the other
# previously produced a run that loaded v11 while writing into test_results/v12.
#
#   RUN_NAME    checkpoint directory                notes
#   v27         checkpoints/full_run_cloud_v27      v26 config, CLEAN embeddings.py
#                                                    (final Nyquist fixes + wavelength
#                                                    placement + sharing, all landed) ← current
#   v26.1       test_results/full_run_cloud_v26.1   shared emb + temporal wavelength
#                                                    fix, but step/delta_emb still at
#                                                    the half-wrong intermediate Nyquist
#                                                    value (2.0/1.0, not 2.5/1.25) —
#                                                    saved OUTSIDE checkpoints/, see below
#   v26-res     test_results/full_run_cloud_v26-res shared-embedding fix only; predates
#                                                    every wavelength/Nyquist fix —
#                                                    saved OUTSIDE checkpoints/, see below
#   v20         checkpoints/full_run_cloud_v20      residual head + station_state
#   v19         checkpoints/full_run_cloud_v19      v17 + MLP value embedding
#   v17         checkpoints/full_run_cloud_v17      token rebalance (0.04% -> 22%)
#   v15         checkpoints/full_run_cloud_v15      patch-3 tokens, no residual
#   v14         checkpoints/full_run_cloud_v14      patch-6, input-context decoder
#   v13         checkpoints/full_run_cloud_v13      pre-audit-fix architecture
#   v12/v11/v9  checkpoints/...                     older Huber / NLL runs
#
# v26-res and v26.1 were saved into test_results/full_run_cloud_<name>/ instead
# of checkpoints/full_run_cloud_<name>/ (that's where run_full_cloud.sh's
# SAVE_DIR actually writes). CHECKPOINT below is always derived from RUN_NAME
# via checkpoints/full_run_cloud_${RUN_NAME}/best.ckpt, so evaluating either of
# those two needs the .ckpt files moved/copied into
# checkpoints/full_run_cloud_<name>/ first — there's no RUN_NAME value that
# reaches test_results/ as-is.
#
# Everything structural is read from the checkpoint's saved cfg — including,
# since this revision, value_embedding / static_in_token / direct_head. Those
# three were recorded by main.py but never read here, so a v18+ checkpoint
# rebuilt itself with the pre-v18 defaults. The load-time structural guard
# catches it (var_proj weights land in missing/unexpected and it aborts), but
# only after the dataset build, so check the [v18+] banner line matches the
# training run before letting a long sweep proceed.
#
# ⚠ PRE-v15 CHECKPOINTS CANNOT BE EVALUATED WITH THIS CODE. v15 removed the
#   decoder input-context pathway, so those checkpoints carry weights this
#   model no longer has; test.py ABORTS rather than silently dropping them
#   (that silent drop is what invalidated every v9–v13 test number). To
#   evaluate an old run, check out the commit that trained it:
#       git checkout <commit>  &&  bash src/scripts/run_test_cloud.sh
RUN_NAME="v28"

case "$RUN_NAME" in
  v9)  CKPT_DIR="run_full_cloud_v9"  ;;
  *)   CKPT_DIR="full_run_cloud_${RUN_NAME}" ;;
esac
CHECKPOINT="${PROJ_DIR}/checkpoints/${CKPT_DIR}/best.ckpt"
SAVE_DIR="${PROJ_DIR}/test_results/${RUN_NAME}"

# Fail here rather than 4 minutes into the dataset build.
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[run_test_cloud.sh] checkpoint for RUN_NAME='${RUN_NAME}' not found:"
  echo "    $CHECKPOINT"
  echo "  available runs under ${PROJ_DIR}/checkpoints:"
  ls -1 "${PROJ_DIR}/checkpoints" 2>/dev/null | sed 's/^/    /' || echo "    (no checkpoints/ directory)"
  exit 1
fi
echo "[run_test_cloud.sh] RUN_NAME=${RUN_NAME}"
echo "                    checkpoint: ${CHECKPOINT}"
echo "                    output:     ${SAVE_DIR}"

# ── Architecture preflight ───────────────────────────────────────────────────
# Print what the checkpoint says it is BEFORE the ~4 minute dataset build, so a
# cfg/code mismatch costs seconds instead of a build. test.py aborts on this
# anyway (the structural guard), just much later.
python - "$CHECKPOINT" <<'PYEOF'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = c.get("hyper_parameters", {}).get("cfg", {})
if not cfg:
    print("[preflight] no cfg in checkpoint — test.py will fall back to CLI/defaults")
    raise SystemExit(0)
print(f"[preflight] epoch {c.get('epoch','?')}, step {c.get('global_step','?')}")
for k in ("d_model", "enc_layers", "dec_layers", "temporal_patch",
          "factorised_encoder", "temporal_window", "mask_ratio",
          "value_embedding", "wind_encoder", "static_in_token",
          "residual_head", "direct_head", "readout"):
    if k in cfg:
        print(f"[preflight]   {k:20s} {cfg[k]}")
if cfg.get("direct_head"):
    print("[preflight] direct_head — only mask ratio 0.0 will be evaluated")
PYEOF

# ── Window mode ───────────────────────────────────────────────────────────────
# blocks:  non-overlapping windows (~1,460) — fast, clean — recommended default
# sliding: all overlapping windows (~105k)  — slow, most stable — for final paper metrics
# Rolling-origin evaluation: sliding windows every STRIDE steps (9 = 90 min = 1h30).
INDEX_MODE="sliding"
STRIDE=9
# INDEX_MODE="blocks"   # fast non-overlapping alternative

# ── Mask ratio sweep ──────────────────────────────────────────────────────────
# 0.0 → all stations visible to encoder (pure temporal forecasting)
# 0.5 → trained setting: 50% masked (gap-filling + forecasting)
# The model is loaded ONCE and evaluated at each ratio in sequence.
# Results save to separate subdirs: test_results/.../best_mr0.00/, best_mr0.50/
# A comparison table is printed at the end.
# ── Normalisation mode ───────────────────────────────────────────────────────
# Use --global_norm for checkpoints trained BEFORE per-station normalisation
# was introduced (i.e. all baseline-cloud and early tw12-d1024 runs).
# Omit (default) for new checkpoints trained with per-station normalisation.
# GLOBAL_NORM="--global_norm"
GLOBAL_NORM=""

MASK_RATIOS="0.0 0.5"   # mr0.00 FIRST for v20. The question v20 exists to
                        # answer is whether the residual head fixes the copy
                        # failure — v15 scored 0.1512 on pressure against a
                        # 0.0224 persistence baseline — and that comparison is
                        # only meaningful with every station visible, which is
                        # also the setting the LSTM and simple-MAE numbers use.
                        # mr0.50 is the trained setting and gives skill-vs-
                        # persistence + gap-filling; keep it second so killing
                        # the run still leaves the comparable number.
                        # For pre-v20 runs the old order (0.5 first) was right.
# MASK_RATIOS="0.0"                 # single ratio (faster)

# The model is loaded ONCE; each ratio is a single forward sweep over the same
# sliding windows. --save_predictions 0 = keep ALL windows (sliding/9 over
# 2023-24 ≈ 11k windows; expect ~1.5 GB per ratio).
# batch_size: inference keeps no optimiser state and no activations, so it can
# run larger batches than training — ON THE HARDWARE THIS WAS TUNED FOR.
#
# mr0.00 is the heaviest configuration this runs: nothing is masked, so the
# encoder carries all 155 stations instead of the ~78 it sees during training.
# At --temporal_patch 3 that is 24x155 = 3,720 tokens vs 24x78 = 1,872 at
# training time — on 11,684 windows. 4x, not 8x, if you ever evaluate a
# PATCH=1 (raw) checkpoint, where mr0.00 is 72x155 = 11,160.
# Predictions are bit-identical at any batch size; only speed changes.
#
# BATCH_SIZE WAS 16, TUNED FOR THE A100 MIG 3g.20gb (~20 GB) — HARDCODED
# ---------------------------------------------------------------------------
# OOM observed 2026-08 at mr0.00 on a smaller allocation: total capacity
# 7.82 GiB (not the ~20 GB the comment above assumed — check `nvidia-smi` /
# the OOM message's "GPU 0 has a total capacity of ..." line before assuming
# which node you have; Renku hands out different slices across sessions).
# 16 does not fit a GPU ~2.5x smaller. Dropped to 4 here as a safe default;
# if it STILL OOMs, drop further (2, then 1) — correctness is unaffected,
# only wall-clock. expandable_segments reduces the allocator fragmentation
# PyTorch's own OOM message flagged (1.43 GiB reserved-but-unallocated).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
BATCH_SIZE="${BATCH_SIZE:-4}"

python test.py \
    --data_root        "$DATA_ROOT" \
    --checkpoint       "$CHECKPOINT" \
    --batch_size       "$BATCH_SIZE" \
    --num_workers      4 \
    --index_mode       "$INDEX_MODE" \
    --stride           "$STRIDE" \
    --exclude_stations PFA \
    --test_mask_ratios $MASK_RATIOS \
    --predictions_only \
    --save_predictions 0 \
    $GLOBAL_NORM \
    --save_dir         "$SAVE_DIR"

echo ""
echo "Done. Predictions under: ${SAVE_DIR}/  (one dir per mask ratio: ${MASK_RATIOS})"
echo "Compute all metrics in Test_Results_Exploration.ipynb."
