#!/usr/bin/env python
"""
inspect_checkpoints.py — what is actually inside a checkpoint directory.

    python inspect_checkpoints.py                       # all runs under checkpoints/
    python inspect_checkpoints.py full_run_cloud_v12    # one run
    python inspect_checkpoints.py ../checkpoints/*.ckpt # explicit files

Why this exists
---------------
Lightning's ModelCheckpoint does NOT overwrite an existing file — it appends
`-v1`, `-v2`, … So a directory containing

    best.ckpt   best-v1.ckpt   last.ckpt   last-v1.ckpt

is the signature of **two training runs writing into the same SAVE_DIR**. The
plain `best.ckpt` is then the *older* run and `best-v1.ckpt` the newer one —
the opposite of what the names suggest. Pointing an evaluation at `best.ckpt`
in that situation silently scores the wrong model, which is exactly how the
tw6/tw12 comparison went wrong earlier in this project (four checkpoints, one
md5, all d_model=128).

Timestamps alone are not conclusive — a file copy or a Polybox sync rewrites
mtime. The decisive evidence is inside: epoch, global_step, the monitored
score, and the architecture the weights actually imply.
"""

import datetime
import glob
import os
import sys

import torch


def _fmt_bytes(n: int) -> str:
    return f"{n/1e6:,.0f} MB" if n < 1e9 else f"{n/1e9:.2f} GB"


def _cfg_of(ckpt: dict) -> dict:
    hp = ckpt.get("hyper_parameters") or {}
    if isinstance(hp, dict):
        inner = hp.get("cfg")
        if isinstance(inner, dict):
            return inner
        return hp
    return {}


def _checkpoint_callback_score(ckpt: dict):
    """Pull best_model_score / best_model_path out of the ModelCheckpoint state."""
    for key, state in (ckpt.get("callbacks") or {}).items():
        if "ModelCheckpoint" in str(key) and isinstance(state, dict):
            score = state.get("best_model_score")
            if hasattr(score, "item"):
                score = score.item()
            return score, state.get("best_model_path"), state.get("monitor")
    return None, None, None


def describe(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _cfg_of(ck)
    sd = ck.get("state_dict", ck)
    score, best_path, monitor = _checkpoint_callback_score(ck)

    # Architecture inferred from the weights themselves — this cannot lie,
    # whereas cfg can be absent or stale.
    d_model = None
    for k, v in sd.items():
        if k.endswith("var_embed.weight") or k.endswith("variable_embed.weight"):
            d_model = v.shape[-1]
            break
    if d_model is None:
        for k, v in sd.items():
            if v.ndim == 2 and ("encoder" in k and k.endswith(".weight")):
                d_model = v.shape[-1]
                break
    has_nll = any(k.endswith("log_var_head.weight") for k in sd)
    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))

    return {
        "file": os.path.basename(path),
        "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path)),
        "size": os.path.getsize(path),
        "epoch": ck.get("epoch", "?"),
        "step": ck.get("global_step", "?"),
        "monitor": monitor,
        "score": score,
        "d_model": cfg.get("d_model", d_model),
        "window": cfg.get("window", "?"),
        "max_delta": cfg.get("max_delta", "?"),
        "enc_layers": cfg.get("enc_layers", "?"),
        "nll": has_nll,
        "params": n_params,
        "best_path": best_path,
    }


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(os.path.dirname(here), "checkpoints")

    if not argv:
        targets = sorted(glob.glob(os.path.join(root, "*", "*.ckpt")))
    elif all(a.endswith(".ckpt") for a in argv):
        targets = argv
    else:
        targets = []
        for a in argv:
            d = a if os.path.isdir(a) else os.path.join(root, a)
            targets += sorted(glob.glob(os.path.join(d, "*.ckpt")))

    if not targets:
        raise SystemExit(f"no .ckpt files found (looked under {root})")

    by_dir: dict[str, list] = {}
    for p in targets:
        by_dir.setdefault(os.path.dirname(p), []).append(p)

    for d, files in by_dir.items():
        print(f"\n{'='*100}\n{d}\n{'='*100}")
        rows = []
        for p in sorted(files):
            try:
                rows.append(describe(p))
            except Exception as e:                       # noqa: BLE001
                print(f"  {os.path.basename(p):18s} UNREADABLE: {type(e).__name__}: {e}")
        if not rows:
            continue

        hdr = (f"{'file':<18}{'modified':<18}{'epoch':>6}{'step':>9}"
               f"{'score':>10}  {'d_model':>7}{'win':>5}{'maxΔ':>6}{'NLL':>5}  {'size':>9}")
        print(hdr); print("-" * len(hdr))
        newest = max(rows, key=lambda r: r["mtime"])
        for r in rows:
            sc = f"{r['score']:.4f}" if isinstance(r["score"], float) else "—"
            mark = "  <- newest" if r is newest else ""
            print(f"{r['file']:<18}{r['mtime']:%Y-%m-%d %H:%M}  {str(r['epoch']):>6}"
                  f"{str(r['step']):>9}{sc:>10}  {str(r['d_model']):>7}{str(r['window']):>5}"
                  f"{str(r['max_delta']):>6}{'yes' if r['nll'] else 'no':>5}  "
                  f"{_fmt_bytes(r['size']):>9}{mark}")

        if rows[0]["monitor"]:
            print(f"\n  monitored metric: {rows[0]['monitor']}")

        # ── the warning that matters ──
        versioned = [r for r in rows if "-v" in r["file"]]
        if versioned:
            print("\n  ⚠  VERSIONED FILES PRESENT (-v1, -v2 …).")
            print("     Lightning appends -vN instead of overwriting, so this directory")
            print("     was written by MORE THAN ONE RUN. `best.ckpt` is the OLDER run.")
            print("     Compare epoch/step/score above and evaluate the one you meant —")
            print("     do not assume `best.ckpt` is current.")
            best_paths = {r["best_path"] for r in rows if r["best_path"]}
            for bp in sorted(x for x in best_paths if x):
                print(f"     ModelCheckpoint recorded its best as: {bp}")

        cfgs = {(r["d_model"], r["window"], r["max_delta"]) for r in rows}
        if len(cfgs) > 1:
            print("\n  ⚠  These checkpoints do NOT share an architecture/window config:")
            for c in sorted(map(str, cfgs)):
                print(f"     (d_model, window, max_delta) = {c}")
            print("     They are different experiments sharing a folder.")


if __name__ == "__main__":
    main(sys.argv[1:])
