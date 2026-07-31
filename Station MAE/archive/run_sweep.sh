#!/usr/bin/env bash
# run_sweep.sh — WandB sweep agent launcher for Station-MAE hyperparameter search
#
# Usage:
#   # Step 1: register the sweep with WandB (run once from this directory)
#   wandb sweep sweep.yaml
#   # → WandB prints: "wandb agent <entity>/station-mae/<sweep_id>"
#
#   # Step 2: launch the agent (runs until all sweep trials are done or Ctrl+C)
#   bash run_sweep.sh <sweep_id>
#   # e.g.: bash run_sweep.sh abc123xyz
#
# The agent runs trials sequentially on the single A100 MIG partition.
# Each trial takes ~3-4 h (25 epochs × 45 min, early-stopped at patience=5).
# With 8-10 Bayesian trials the full sweep runs in ~30-40 h.
#
# Tip: You can interrupt the agent and re-launch with the same sweep_id —
# completed runs are preserved in WandB and Bayesian search resumes from them.

set -euo pipefail

SWEEP_ID="${1:-}"
if [[ -z "$SWEEP_ID" ]]; then
    echo "Usage: bash run_sweep.sh <sweep_id>"
    echo ""
    echo "Create a sweep first with:"
    echo "  wandb sweep sweep.yaml"
    exit 1
fi

ENTITY="${WANDB_ENTITY:-}"   # set WANDB_ENTITY env var or leave blank for default
PROJECT="station-mae"

if [[ -n "$ENTITY" ]]; then
    FULL_ID="${ENTITY}/${PROJECT}/${SWEEP_ID}"
else
    FULL_ID="${PROJECT}/${SWEEP_ID}"
fi

echo "Launching WandB sweep agent: ${FULL_ID}"
echo "Running from: $(pwd)"
echo ""

# Run up to 10 trials sequentially (Bayesian agent will select configs)
wandb agent --count 10 "${FULL_ID}"
