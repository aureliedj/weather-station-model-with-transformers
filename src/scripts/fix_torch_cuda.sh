#!/usr/bin/env bash
# fix_torch_cuda.sh — make the installed torch match THIS machine's driver.
#
# Context: the project runs on two Renku computes with different NVIDIA
# driver generations. The default PyPI torch wheel (built for CUDA 13.0)
# works on the newer one but fails CUDA init on the older (driver 12.8) —
# either silently falling back to CPU (bare torch) or crashing in Lightning
# ("The NVIDIA driver on your system is too old ... found version 12080").
#
# The cu128 wheel index (https://download.pytorch.org/whl/cu128) carries
# cp313 Linux wheels only up to torch 2.11.0 — torch 2.12/2.13 exist solely
# as cu130 builds. Hence:
#     driver CUDA ≥ 13.0  → default PyPI torch (cu130) is fine
#     driver CUDA 12.8/12.9 → torch 2.11.0+cu128
#     driver CUDA 12.6/12.7 → torch 2.11.0+cu126
#
# Run once per fresh venv / new session:
#   bash src/scripts/fix_torch_cuda.sh
# Idempotent: exits immediately if CUDA already initialises.

set -euo pipefail

if python - <<'PY'
import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)
PY
then
    python -c "import torch; print(f'[fix_torch_cuda] OK: torch {torch.__version__}, CUDA available — nothing to do')"
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[fix_torch_cuda] No nvidia-smi — CPU-only machine? Leaving torch untouched."
    exit 0
fi

# Driver-supported CUDA version, e.g. "12.8" or "13.0"
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
echo "[fix_torch_cuda] CUDA init failed. Driver supports CUDA ${DRIVER_CUDA}."

MAJOR=${DRIVER_CUDA%%.*}
MINOR=${DRIVER_CUDA##*.}

if   [[ "$MAJOR" -ge 13 ]]; then
    echo "[fix_torch_cuda] Driver is CUDA 13+ but CUDA init failed anyway —"
    echo "                 not a wheel/driver mismatch. Inspect manually (nvidia-smi, dmesg)."
    exit 1
elif [[ "$MAJOR" -eq 12 && "$MINOR" -ge 8 ]]; then
    IDX="https://download.pytorch.org/whl/cu128"
elif [[ "$MAJOR" -eq 12 ]]; then
    IDX="https://download.pytorch.org/whl/cu126"
else
    echo "[fix_torch_cuda] Driver CUDA ${DRIVER_CUDA} is older than anything this project supports."
    exit 1
fi

echo "[fix_torch_cuda] Installing torch==2.11.0 from ${IDX} …"
pip install --force-reinstall "torch==2.11.0" --index-url "$IDX"

python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[fix_torch_cuda] torch {torch.__version__} · CUDA available: {ok}"
      + (f" · {torch.cuda.get_device_name(0)}" if ok else ""))
raise SystemExit(0 if ok else 1)
PY
echo "[fix_torch_cuda] Done."
