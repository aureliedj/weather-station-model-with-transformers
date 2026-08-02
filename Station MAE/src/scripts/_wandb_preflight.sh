# shellcheck shell=bash
# _wandb_preflight.sh — make sure WandB will actually log before a run starts.
#
# Source this from a launch script:  source "${SCRIPT_DIR}/_wandb_preflight.sh"
#
# Why this exists
# ---------------
# Two silent failures used to waste whole runs:
#   1. WANDB_MODE defaulted to "offline", so the dashboard stayed empty and
#      nobody noticed until they went looking for the curves.
#   2. Online mode with no API key crashed with "No API key configured" — but
#      only AFTER the dataset build, several minutes in.
#
# So: default to online, and if there is no credential, prompt for one NOW.
#
# Modes:
#   WANDB_MODE=online    (default) live dashboard; requires a key
#   WANDB_MODE=offline   log to ./wandb, sync later with `wandb sync <dir>`
#   WANDB_MODE=disabled  no logging at all

WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

_wandb_has_key() {
    # A key can come from the env, ~/.netrc (what `wandb login` writes), or the
    # wandb library itself. Check all three — any one is sufficient.
    [[ -n "${WANDB_API_KEY:-}" ]] && return 0
    grep -qs "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null && return 0
    python - <<'PY' 2>/dev/null
import sys
try:
    import wandb
    sys.exit(0 if wandb.api.api_key else 1)
except Exception:
    sys.exit(1)
PY
}

if [[ "$WANDB_MODE" == "online" ]]; then
    if _wandb_has_key; then
        echo "[wandb] online — credential found"
    else
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "[wandb] No API key found, so this run would not appear in WandB."
        echo "        Get your key from:  https://wandb.ai/authorize"
        echo "════════════════════════════════════════════════════════════════"
        if [[ -t 0 ]]; then
            # Interactive shell — prompt now rather than failing later.
            wandb login || {
                echo "[wandb] login failed."
                echo "        Re-run with WANDB_MODE=offline to log locally instead."
                exit 1
            }
            _wandb_has_key || {
                echo "[wandb] still no key after login — aborting."
                exit 1
            }
            echo "[wandb] logged in"
        else
            # Non-interactive (nohup / cron / CI): cannot prompt, so say exactly
            # what to do instead of hanging on a password prompt.
            echo "[wandb] Not an interactive shell, cannot prompt. Choose one:"
            echo "          wandb login                      # once, then re-run"
            echo "          export WANDB_API_KEY=<key>       # then re-run"
            echo "          WANDB_MODE=offline bash <script> # log locally, sync later"
            exit 1
        fi
    fi
elif [[ "$WANDB_MODE" == "offline" ]]; then
    echo "[wandb] OFFLINE — nothing will appear in the dashboard during the run."
    echo "        Sync afterwards with:  wandb sync \"\$(ls -dt wandb/run-* | head -1)\""
else
    echo "[wandb] mode=${WANDB_MODE} — logging disabled"
fi

# ── CUDA preflight (two Renku computes, two driver generations) ──────────────
# The stock torch wheel (cu130) fails CUDA init on the older node (driver
# 12.8): bare torch silently trains on CPU, Lightning hard-crashes. Refuse to
# launch instead — the fix is one script away and takes a minute.
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo ""
        echo "[cuda] ✗ GPU present but torch cannot initialise CUDA (wheel/driver mismatch)."
        echo "       Fix it with:   bash src/scripts/fix_torch_cuda.sh"
        echo "       then re-run this script."
        exit 1
    fi
    echo "[cuda] ✓ $(python -c "import torch; print(f'torch {torch.__version__} · {torch.cuda.get_device_name(0)}')" 2>/dev/null)"
fi
