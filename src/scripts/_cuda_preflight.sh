# shellcheck shell=bash
# _cuda_preflight.sh — refuse to launch on a GPU node where torch cannot use CUDA.
#
# Source this from a launch script:  source "${SCRIPT_DIR}/_cuda_preflight.sh"
#
# Why this exists
# ---------------
# The project runs on two Renku computes with different driver generations.
# When the installed torch wheel is built for a newer CUDA than the driver
# supports, the failure is NOT loud:
#   • bare torch  → silently falls back to CPU (a 100 h run becomes weeks)
#   • Lightning   → hard crash several minutes in, after the dataset build
# Both waste time; this check costs a second and names the fix.
#
# Skipped silently on machines with no nvidia-smi (laptop / CPU-only).

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
