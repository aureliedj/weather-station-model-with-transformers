#!/usr/bin/env python
"""
inspect_checkpoints.py

Print the configuration stored in Lightning checkpoints.

    python src/inspect_checkpoints.py                         # every checkpoints/*/*.ckpt
    python src/inspect_checkpoints.py full_run_cloud_v27      # one run directory
    python src/inspect_checkpoints.py path/to/best.ckpt ...   # explicit files
    python src/inspect_checkpoints.py --full path/to/best.ckpt   # dump the whole cfg
"""

import glob
import os
import sys

import torch

_KEYS = ("d_model", "enc_layers", "dec_layers", "enc_heads", "temporal_patch",
         "mask_ratio", "encoder_spatial_attn", "station_local_decoder", "use_nll_loss",
         "hidden", "lstm_layers", "index_mode", "train_stride", "epochs", "lr")


def describe(path: str, full: bool = False) -> None:
    ck  = torch.load(path, map_location="cpu", weights_only=False)
    cfg = (ck.get("hyper_parameters") or {}).get("cfg", {}) or {}
    sd  = ck.get("state_dict", {})
    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    score = None
    for key, state in (ck.get("callbacks") or {}).items():
        if "ModelCheckpoint" in str(key) and isinstance(state, dict):
            score = state.get("best_model_score")
            score = float(score) if score is not None else None
    print(f"\n{path}")
    print(f"  epoch {ck.get('epoch', '?')}  step {ck.get('global_step', '?')}  "
          f"best score {score if score is None else f'{score:.4f}'}  "
          f"{n_params:,} parameters  {os.path.getsize(path) / 1e6:,.0f} MB")
    if full:
        for k in sorted(cfg):
            if not k.startswith("obs_stats"):
                print(f"    {k:24s} {cfg[k]}")
    else:
        print("  " + "  ".join(f"{k}={cfg[k]}" for k in _KEYS if k in cfg))


def main(argv) -> None:
    full = "--full" in argv
    argv = [a for a in argv if a != "--full"]
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

    if not argv:
        targets = sorted(glob.glob(os.path.join(root, "*", "*.ckpt")))
    else:
        targets = []
        for a in argv:
            if a.endswith(".ckpt"):
                targets.append(a)
            else:
                d = a if os.path.isdir(a) else os.path.join(root, a)
                targets += sorted(glob.glob(os.path.join(d, "*.ckpt")))
    if not targets:
        raise SystemExit(f"no .ckpt files found (looked under {root})")
    for p in targets:
        describe(p, full=full)


if __name__ == "__main__":
    main(sys.argv[1:])
