#!/usr/bin/env bash
# run_test_cloud.sh — dump raw test-set predictions for one checkpoint.
#
# One forward pass over the sliding test windows per mask ratio, writing:
#     test_results/<RUN_NAME>/best_mr<R>/predictions.pt
#
# That file is the ONLY output. Every metric — MAE/RMSE per lead time, per
# variable, per station, masked vs visible, persistence skill — is derived
# downstream in notebooks/Test_Results_Exploration.ipynb, where the per-station
# inverse normalisation is applied correctly.
#
# predictions.pt contains:
#     preds        (M, K, N, 5)   normalised predictions
#     targets      (M, K, N, 6)   normalised targets
#     masks        (M, K, N, 6)   sensor availability
#     masked_idx   (M, n_masked)  stations hidden from the encoder (empty at MR 0)
#     delta_steps  (M, K)         lead times in 10-min steps
#     window_hours (M,)           window start, hours since epoch
#     target_hours (M, K)         target time per lead
#     spatial      (N, 15)        static station descriptors
#     log_var      (M, K, N, 5)   log sigma^2 — ONLY for NLL checkpoints
#
# All architecture settings are read from the checkpoint's saved cfg, so the
# model is always rebuilt exactly as it was trained.
#
# Usage:
#   bash src/scripts/run_test_cloud.sh
#   MASK_RATIOS="0.0 0.5" bash src/scripts/run_test_cloud.sh
#   SEED=7 BATCH_SIZE=2 bash src/scripts/run_test_cloud.sh

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root
cd "${SRC_DIR}"

# Fail fast if torch cannot use the GPU (wheel/driver mismatch).
source "${SCRIPT_DIR}/_cuda_preflight.sh"

# ── Which run to evaluate ────────────────────────────────────────────────────
# Set RUN_NAME only: the checkpoint path and the output folder are both derived
# from it, so they cannot drift apart.
#
#   RUN_NAME     encoder spatial   decoder      train MR   valid mask ratios
#   v27          yes               global       0.5        0.0, 0.5
#   v30-nll      yes               global       0.5        0.0, 0.5   (+ log_var)
#   v31          yes               global       0.0        0.0        (0.5 is OOD)
#   v32-blind    no                station-local 0.0       0.0 only   (see below)
#   lstm-*       n/a               n/a          n/a        use run_lstm_test_cloud.sh
RUN_NAME="v30-nll"

CKPT_DIR="full_run_cloud_${RUN_NAME}"
CHECKPOINT="${PROJ_DIR}/checkpoints/${CKPT_DIR}/best.ckpt"
SAVE_DIR="${PROJ_DIR}/test_results/${RUN_NAME}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[run_test_cloud.sh] checkpoint for RUN_NAME='${RUN_NAME}' not found:"
  echo "    $CHECKPOINT"
  echo "  available:"
  ls -1 "${PROJ_DIR}/checkpoints" 2>/dev/null | sed 's/^/    /' || echo "    (none)"
  exit 1
fi
echo "[run_test_cloud.sh] RUN_NAME=${RUN_NAME}"
echo "                    checkpoint: ${CHECKPOINT}"
echo "                    output:     ${SAVE_DIR}"

# ── Architecture preflight ───────────────────────────────────────────────────
# Print what the checkpoint says it is BEFORE the ~4 min dataset build, so a
# mismatch costs seconds instead of a full build.
python - "$CHECKPOINT" <<'PYEOF'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = c.get("hyper_parameters", {}).get("cfg", {})
if not cfg:
    print("[preflight] no cfg in checkpoint — test.py falls back to CLI/defaults")
    raise SystemExit(0)
print(f"[preflight] epoch {c.get('epoch','?')}, step {c.get('global_step','?')}")
for k in ("d_model", "enc_layers", "dec_layers", "temporal_patch",
          "factorised_encoder", "encoder_spatial_attn", "station_local_decoder",
          "temporal_window", "mask_ratio", "value_embedding", "static_in_token",
          "residual_head", "direct_head", "readout", "use_nll_loss"):
    if k in cfg:
        print(f"[preflight]   {k:22s} {cfg[k]}")
# The sigma head is detected from the WEIGHTS, not the cfg key (unreliable on
# v9-era runs) — mirror test.py so the banner cannot disagree with the build.
_sd = c.get("state_dict", {})
if any(k.endswith("decoder.log_var_head.weight") for k in _sd):
    print("[preflight]   sigma head           PRESENT — log_var saved in predictions.pt")
if cfg.get("direct_head") or cfg.get("station_local_decoder"):
    print("[preflight] ⚠ this model requires mask_ratio 0 — MR>0 will be refused")
PYEOF

# ── Evaluation window protocol ───────────────────────────────────────────────
# sliding + stride 9 (90 min) over 2023-24 → 11,684 windows. Keep these fixed:
# every existing dump uses them, and changing either makes runs incomparable.
INDEX_MODE="sliding"
STRIDE=9

# ── Mask ratios ──────────────────────────────────────────────────────────────
# 0.0 → all stations visible: pure forecasting. The only regime in which the
#       LSTM baseline is comparable, and the only one v31/v32 support.
# 0.5 → 50% of stations hidden: gap-filling + forecasting. Meaningful only for
#       checkpoints TRAINED with masking (v27, v30-nll); for a model trained at
#       MR 0 it is out of distribution and reports robustness, not a ranking.
#
# v30-nll was trained at MR 0.5, so both ratios are in distribution: 0.0 gives
# the LSTM-comparable forecasting numbers, 0.5 the paired masked-station
# comparison against v27. Both passes write log_var, so the sigma calibration
# can be checked on visible and hidden stations separately.
MASK_RATIOS="${MASK_RATIOS:-0.0 0.5}"

# ── Normalisation ────────────────────────────────────────────────────────────
# Empty = per-station (all current runs). --global_norm only for pre-per-station
# checkpoints; it must match how the checkpoint was trained.
GLOBAL_NORM=""

# ── Batch size ───────────────────────────────────────────────────────────────
# Inference holds no optimiser state or activations, but MR 0.0 is the heaviest
# case: nothing is masked, so the encoder carries all 155 stations
# (24x155 = 3,720 tokens vs 24x78 = 1,872 at MR 0.5). Predictions are identical
# at any batch size; only speed changes. Drop to 2 or 1 if it OOMs.
BATCH_SIZE="${BATCH_SIZE:-4}"

# ── CUDA allocator ───────────────────────────────────────────────────────────
# expandable_segments:True cuts allocator fragmentation but puts the caching
# allocator on the CUDA virtual-memory API (cuMemCreate / cuMemAddressReserve).
# Several vGPU profiles — including the A10-8Q this project is allocated — do
# not implement those, and the first host->device copy then fails with
#     RuntimeError: CUDA driver error: operation not supported
# from model.to(device), long after torch.cuda.is_available() returned True.
# Probe instead of assuming. Force with EXPANDABLE_SEGMENTS=1 / =0.
_probe_expandable() {
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python - <<'PYEOF' >/dev/null 2>&1
import torch
if torch.cuda.is_available():
    torch.zeros(8).to("cuda")
PYEOF
}
case "${EXPANDABLE_SEGMENTS:-auto}" in
  1) export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
     echo "[alloc] expandable_segments:True (forced)" ;;
  0) unset PYTORCH_CUDA_ALLOC_CONF
     echo "[alloc] expandable_segments disabled (forced)" ;;
  *) if _probe_expandable; then
         export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
         echo "[alloc] expandable_segments:True — probe passed, enabled"
     else
         unset PYTORCH_CUDA_ALLOC_CONF
         echo "[alloc] expandable_segments unsupported on this GPU — disabled"
     fi ;;
esac

# ── Station-mask seed ────────────────────────────────────────────────────────
# The mask at MR>0 is drawn from the global RNG. test.py re-seeds it ONCE PER
# MASK RATIO, so the same SEED hides the same stations across models (paired
# masked-station comparisons) and the MR 0.5 mask does not depend on whether
# MR 0.0 ran first. Use the SAME seed for every model you intend to compare.
# Reproducibility also requires the same BATCH_SIZE, INDEX_MODE and STRIDE.
# Dumps produced before seeding existed cannot be reproduced.
SEED="${SEED:-42}"

python test.py \
    --data_root        "$DATA_ROOT" \
    --checkpoint       "$CHECKPOINT" \
    --batch_size       "$BATCH_SIZE" \
    --num_workers      4 \
    --index_mode       "$INDEX_MODE" \
    --stride           "$STRIDE" \
    --exclude_stations PFA \
    --test_mask_ratios $MASK_RATIOS \
    --seed             "$SEED" \
    --save_predictions 0 \
    $GLOBAL_NORM \
    --save_dir         "$SAVE_DIR"

echo ""
echo "Done. predictions.pt under: ${SAVE_DIR}/  (one dir per mask ratio: ${MASK_RATIOS})"
echo "Compute metrics in notebooks/Test_Results_Exploration.ipynb."
