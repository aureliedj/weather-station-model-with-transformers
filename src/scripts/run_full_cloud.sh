#!/usr/bin/env bash
# run_full_cloud.sh — full training run for the Station-MAE transformer.
#
#   bash src/scripts/run_full_cloud.sh              full run, auto-resumes
#   SANITY=1 bash src/scripts/run_full_cloud.sh     ~5 min smoke test
#   SANITY=2 bash src/scripts/run_full_cloud.sh     ~1 h "does it learn?"
#   RESUME=0 bash src/scripts/run_full_cloud.sh     force a fresh run
#
# Environment overrides: DATA_ROOT, SANITY, RESUME, WANDB_API_KEY, WANDB_MODE.
#
# WHY each setting is what it is — hardware constraints, tokenisation, the
# observation encoder, the residual head, the loss family, the spatial-ablation
# arms, epoch sizing and resume semantics — is documented in
#     docs/training_configuration.md
# The fully annotated original of this script is preserved at
#     archive/removed_deadcode_2026-08-25/run_full_cloud.sh.annotated-original
# Checkpoint -> configuration -> results is in EXPERIMENTS.md.

set -euo pipefail

# ── Weights & Biases ─────────────────────────────────────────────────────────
# Key comes from the environment; it is NEVER stored in the repository.
# Without it the run logs offline and no metric is lost.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[wandb] WANDB_API_KEY not set — logging will run offline."
  echo "        export WANDB_API_KEY=...   (or run: wandb login)"
fi

# ── Paths ────────────────────────────────────────────────────────────────────
# Checkpoints go inside the project directory: guaranteed writable on Renku,
# unlike sibling paths under /home/renku/work/. Copy to Polybox by hand after
# training — writing there during a run is unreliable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src   — python entry points live here
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root — checkpoints/, test_results/, report/
cd "${SRC_DIR}"                                              # so `python main.py` and `from data...` resolve

# ── Dataset location ─────────────────────────────────────────────────────────
# Default matches src/download.py's own default (<project root>/PeakWeatherDataset),
# not a Renku-specific path — this project's copy lives at the project root.
DATA_ROOT="${DATA_ROOT:-${PROJ_DIR}/PeakWeatherDataset}"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[run_full_cloud.sh] dataset not found at: $DATA_ROOT"
  echo "  Set DATA_ROOT, or fetch it with:  python src/download.py"
  exit 1
fi
SAVE_DIR="${PROJ_DIR}/checkpoints/full_run_cloud_v32-blind"   # own dir — never share a SAVE_DIR (that is what produced the -v1 checkpoints)
LOCAL_CACHE="/tmp/station_mae_cache"

source "${SCRIPT_DIR}/_wandb_preflight.sh"

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — the only block to edit. Commented alternatives are the other arms.
#  Rationale for every choice: docs/training_configuration.md
# ═════════════════════════════════════════════════════════════════════════════

# Stations dropped for insufficient historical coverage. Matched
# case-insensitively against the stations_table index / name / abbreviation.
EXCLUDE="--exclude_stations PFA"   # station 110 (PFA) — insufficient historical coverage

# Observation encoder. NOTE: main.py defaults to "linear"; every run used "mlp".
OBS_ENCODER="--value_embedding mlp"

DELTA0=""                                        # uniform Δ weighting (every run)

# Spatial-information arms. v32 = station-blind: both flags ACTIVE, mask_ratio 0.
#SPATIAL=""                                       # <- v27/v30/v31: full axial
SPATIAL="--no_spatial_attn"                       # <- v32: station-blind arm (ACTIVE)
#DECODER_LOCAL=""                                 # <- v27/v30/v31: decoder sees all stations
DECODER_LOCAL="--station_local_decoder"           # <- v32: decoder is station-local (ACTIVE)
DIRECT=""                                          # keep the Δ-query decoder (every run)

# Loss family.
# NLL="--nll_loss"                               # <- v30: Gaussian NLL + sigma head
NLL=""                                           # <- v27/v31/v32: Huber(delta=1)

RESIDUAL="--residual_head"                       # y(t0) added outside attention (every run)

PATCH=3
ENCODER="--factorised_encoder"            # axial: temporal, then spatial, then shared FFN
TEMPORAL_WINDOW="--temporal_window 0"   # windowing retired — patching replaces it
INDEX_MODE="--index_mode random --random_epoch_size 10000" # ~2.8× non-overlapping (fast ablation)

# ── Sanity / smoke runs ──────────────────────────────────────────────────────
# Both write to <SAVE_DIR>-sanity, so they can never collide with the real run.
SANITY="${SANITY:-0}"
SANITY_ARGS=()
if [[ "$SANITY" == "1" ]]; then
    SANITY_ARGS=(--epochs 2 --random_epoch_size 200 --subset
                 --warmup_epochs 1 --patience 100 --val_check_interval 0)
    INDEX_MODE="--index_mode random"
    SAVE_DIR="${SAVE_DIR}-sanity"
    RUN_SUFFIX="-sanity1"
elif [[ "$SANITY" == "2" ]]; then
    SANITY_ARGS=(--epochs 15 --random_epoch_size 2500 --subset --warmup_epochs 2)
    INDEX_MODE="--index_mode random"
    SAVE_DIR="${SAVE_DIR}-sanity"
    RUN_SUFFIX="-sanity2"
else
    RUN_SUFFIX=""
fi
if [[ "$SANITY" != "0" ]]; then
    echo "[sanity] SANITY=${SANITY} — short run into ${SAVE_DIR}"
    echo "[sanity]   ${SANITY_ARGS[*]}"
    echo "[sanity]   this is NOT the real run; unset SANITY for the full budget."
fi

# ── Resume ───────────────────────────────────────────────────────────────────
# main.py only resumes when --resume is passed. RESUME=0 into a directory that
# already holds best.ckpt is refused, rather than writing a best-v1.ckpt beside it.
RESUME="${RESUME:-1}"
RESUME_ARG=()   # array, not a string: keeps paths with spaces safe
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

# ── Launch ───────────────────────────────────────────────────────────────────
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
    --mask_ratio       0.0 \
    --temporal_patch   $PATCH \
    --var_weights      1.0 1.0 1.0 1.0 1.0 \
    --dropout          0.1 \
    --drop_path_rate   0.1 \
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
    $OBS_ENCODER \
    $RESIDUAL \
    $NLL \
    $SPATIAL \
    $DECODER_LOCAL \
    $DIRECT \
    $DELTA0 \
    $ENCODER \
    $TEMPORAL_WINDOW \
    $INDEX_MODE \
    $EXCLUDE \
    --wandb_project    station-mae \
    --wandb_run_name   "patch${PATCH}-d384-L8-v32-station-blind${RUN_SUFFIX}" \
    ${SANITY_ARGS[@]+"${SANITY_ARGS[@]}"} \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
    --save_dir         "$SAVE_DIR"
