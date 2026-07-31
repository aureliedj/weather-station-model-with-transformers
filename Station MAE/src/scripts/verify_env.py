#!/usr/bin/env python
"""
verify_env.py — check the environment BEFORE starting a multi-day run.

    python src/scripts/verify_env.py

Why this exists
---------------
Two failures in this project look like success until hours or days later:

1. **Silent CPU fallback.** If the installed torch build does not match the
   host driver, `torch.cuda.is_available()` returns False and Lightning
   cheerfully trains on CPU — roughly 100x slower. A 3-day run becomes
   never-finishing, and nothing in the log says "no GPU".

2. **A changed compute tier.** Renku sessions are ephemeral: switching compute
   gives a fresh container, so pip packages installed in a previous session are
   gone, and the new node may have a different GPU, a different driver, and a
   different amount of VRAM than the one `batch_size` was tuned for.

Exits non-zero if the environment cannot train, so it can gate a launch script.
"""

import os
import shutil
import subprocess
import sys

FAIL: list[str] = []
WARN: list[str] = []


def head(t: str) -> None:
    print(f"\n{t}\n" + "-" * len(t))


head("Host GPU (nvidia-smi)")
if shutil.which("nvidia-smi") is None:
    print("  nvidia-smi not found — no NVIDIA driver visible in this container.")
    FAIL.append("no nvidia-smi: this compute tier has no GPU attached")
else:
    try:
        q = "name,driver_version,memory.total,memory.used"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        rows = [r for r in out.stdout.strip().splitlines() if r.strip()]
        if not rows:
            print("  nvidia-smi returned no GPUs.")
            FAIL.append("nvidia-smi lists no GPU")
        for r in rows:
            print(f"  {r}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  nvidia-smi failed: {type(e).__name__}: {e}")
        WARN.append("could not query nvidia-smi")

head("torch")
try:
    import torch
except ImportError:
    print("  torch is NOT installed.")
    print("  → pip install -r requirements.txt")
    raise SystemExit("\n[ABORT] torch missing — install requirements first.")

print(f"  torch version      : {torch.__version__}")
print(f"  built for CUDA     : {torch.version.cuda}")
print(f"  cuDNN              : {torch.backends.cudnn.version()}")
print(f"  cuda.is_available(): {torch.cuda.is_available()}")

if "+cu" not in torch.__version__ and sys.platform.startswith("linux"):
    WARN.append(f"torch {torch.__version__} is not a +cuXXX build — "
                "this is the generic PyPI wheel, which may not match the driver")

if not torch.cuda.is_available():
    FAIL.append(
        "torch.cuda.is_available() is False — training would silently run on CPU.\n"
        "      Most likely the installed CUDA build does not match the host driver.\n"
        "      Check the CUDA version nvidia-smi reports, then install the matching\n"
        "      build, e.g. for CUDA 12.8:\n"
        "          pip install torch --index-url https://download.pytorch.org/whl/cu128"
    )
else:
    n = torch.cuda.device_count()
    print(f"  device count       : {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024**3
        print(f"    [{i}] {p.name}  {gb:.1f} GB  sm_{p.major}{p.minor}")

    head("Live allocation test")
    try:
        a = torch.randn(4096, 4096, device="cuda")
        b = (a @ a).sum().item()
        free, total = torch.cuda.mem_get_info()
        print(f"  matmul on GPU OK   (checksum {b:.3e})")
        print(f"  free VRAM          : {free/1024**3:.1f} / {total/1024**3:.1f} GB")
        del a
        torch.cuda.empty_cache()

        # batch_size guidance — the flat d1024 encoder at W=72, N=155 is
        # memory-hungry; these numbers come from what actually fitted before.
        gb_free = free / 1024**3
        if gb_free < 10:
            rec = "batch_size 2  (test.py) / 4  (main.py)"
        elif gb_free < 24:
            rec = "batch_size 4  (test.py) / 8  (main.py)"
        else:
            rec = "batch_size 8+ (test.py) / 16 (main.py)"
        print(f"\n  suggested          : {rec}")
        print("  NOTE: if you just changed compute tier, the batch_size in")
        print("        src/scripts/*.sh was tuned for the PREVIOUS GPU.")
    except Exception as e:                                   # noqa: BLE001
        FAIL.append(f"GPU allocation failed: {type(e).__name__}: {e}")

head("Project imports")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))                   # .../src
for m in ("pytorch_lightning", "wandb", "peakweather",
          "data.dataset", "model.mae", "model.lstm_baseline"):
    try:
        __import__(m)
        print(f"  {m:<28} OK")
    except Exception as e:                                   # noqa: BLE001
        print(f"  {m:<28} FAILED: {type(e).__name__}: {e}")
        FAIL.append(f"cannot import {m}")

head("Result")
for w in WARN:
    print(f"  WARN  {w}")
if FAIL:
    for f in FAIL:
        print(f"  FAIL  {f}")
    raise SystemExit(f"\n[ABORT] {len(FAIL)} blocking problem(s) — do not start a run.")
print("  environment OK — safe to launch." if not WARN else
      "  environment usable, but see warnings above.")
