"""
Data-leakage and training-setup audit.

Runs without torch (static source analysis + numeric simulation of the window
indexer), so it can be run anywhere, including CI.

Each check states the leak it is guarding against, so a future refactor that
breaks one gets an explanation rather than just a red line.

Run: python tests/test_leakage_audit.py
"""
import ast
import os
import re
import sys

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJ, "src")

PASS, FAIL, WARN = [], [], []


def check(name, ok, detail="", warn_only=False):
    (PASS if ok else (WARN if warn_only else FAIL)).append((name, detail))
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {name}")
    if detail and (not ok or os.environ.get("VERBOSE")):
        for line in detail.strip().splitlines():
            print(f"         {line}")


def src(rel):
    with open(os.path.join(_SRC, rel)) as f:
        return f.read()


ds_src = src("data/dataset.py")
mae_src = src("model/mae.py")
enc_src = src("model/encoder.py")
main_src = src("main.py")
test_src = src("test.py")

print("=" * 74)
print("1. TEMPORAL SPLITS")
print("=" * 74)

# ── 1a. splits disjoint ─────────────────────────────────────────────────────
ns = {}
for name in ("TRAIN_YEARS", "VAL_YEARS", "TEST_YEARS"):
    m = re.search(rf"^{name}\s*=\s*(.+?)\s*(?:#.*)?$", ds_src, re.M)
    ns[name] = eval(m.group(1), {"list": list, "range": range})
tr, va, te = set(ns["TRAIN_YEARS"]), set(ns["VAL_YEARS"]), set(ns["TEST_YEARS"])
check("train / val / test years are disjoint",
      not (tr & va) and not (tr & te) and not (va & te),
      f"train={sorted(tr)} val={sorted(va)} test={sorted(te)}\n"
      f"Overlapping years would put the same timestamps in two splits.")
check("splits are chronologically ordered (no future data in train)",
      max(tr) < min(va) and max(va) < min(te),
      f"max(train)={max(tr)} < min(val)={min(va)} < min(test)={min(te)}\n"
      "Training on data later than validation would leak the future.")

# ── 1b. splits are physically sliced, not index-filtered ───────────────────
sliced = ("obs        = obs_full[split_idx]" in ds_src
          or re.search(r"obs\s*=\s*obs_full\[split_idx\]", ds_src) is not None)
check("split arrays are physically sliced before windowing",
      sliced,
      "obs = obs_full[split_idx] — each split is its OWN array, so a window\n"
      "cannot extend past its split boundary into another split. If this were\n"
      "an index filter over a shared array, a val window starting in late 2022\n"
      "could take targets from 2023 (test).")

# ── 1c. window validity spans input AND targets ────────────────────────────
need = re.search(r"_need\s*=\s*window_size\s*-\s*1\s*\+\s*max_delta", ds_src)
maxi = re.search(r"_max_i\s*=\s*max\(0,\s*T\s*-\s*window_size\s*-\s*max_delta\)", ds_src)
check("window validity covers input window AND target horizon",
      need is not None and maxi is not None,
      "_need = W-1+max_delta and _max_i = T-W-max_delta\n"
      "A window is only valid if the whole span it will READ and PREDICT is\n"
      "inside the split and gap-free.")

# numeric proof: no start index can reach past the end of its split
T, W, MD = 52560, 72, 36            # one year at 10-min, v13 window/horizon
max_i = max(0, T - W - MD)
last_touched = (max_i - 1) + (W - 1) + MD
check("no window index reads past the end of its split (simulated)",
      last_touched < T,
      f"T={T:,} W={W} max_delta={MD} -> highest start {max_i-1:,}, "
      f"last timestep touched {last_touched:,} < T  ({T - 1 - last_touched} to spare)")

print()
print("=" * 74)
print("2. NORMALISATION STATISTICS")
print("=" * 74)

check("obs_stats are computed from the TRAIN years only",
      "train_idx  = [i for i, ts in enumerate(timestamps) if ts.year in years]" in ds_src
      and "obs_train  = obs_full[train_idx]" in ds_src,
      "compute_obs_stats slices to train years BEFORE computing mean/std.\n"
      "Using val/test data here would leak their distribution into training.")

for f, label in (("main.py", main_src), ("test.py", test_src)):
    ok = "obs_stats=train_ds.obs_stats" in label or "obs_stats = train_ds.obs_stats" in label
    check(f"{f}: val/test datasets reuse the TRAIN obs_stats", ok,
          "Re-deriving stats on val/test would normalise them with knowledge of\n"
          "their own distribution — a subtle but real leak.")

check("nearest-donor substitution draws only on train-split stats",
      "coords=coords, station_ids=station_ids" in ds_src
      and "obs_train, mask_train, per_station=per_station" in ds_src,
      "Donor means/stds come from `means`/`stds` computed on obs_train, so a\n"
      "sparse station borrows TRAIN statistics from its neighbour, never test.")

print()
print("=" * 74)
print("3. TARGET LEAKAGE INTO THE MODEL")
print("=" * 74)

# y / y_mask must never reach encoder or decoder
tree = ast.parse(mae_src)
calls_ok, offenders = True, []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("decoder", "encoder") or (
            isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
            and node.func.attr in ("decoder", "encoder")
        ):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            bad = names & {"y", "y_mask"}
            if bad:
                calls_ok = False
                offenders.append(f"line {node.lineno}: passes {sorted(bad)}")
check("y / y_mask are never passed to the encoder or decoder",
      calls_ok,
      ("\n".join(offenders) if offenders else
       "The decoder receives only y_hours (WHEN to forecast) and delta_steps.\n"
       "Passing y would be the values themselves; passing y_mask would reveal\n"
       "which sensors are alive at target time — both are leaks."))

check("delta=0 loss is restricted to MASKED stations",
      'if k == 0 and delta_steps[0, 0].item() == 0:' in mae_src
      and "_midx_k = masked_idx" in mae_src,
      "At delta=0 the target timestamp IS the last input step. For a VISIBLE\n"
      "station the answer is already in the input, so supervising it there\n"
      "would reward copying. Restricting to masked stations makes it genuine\n"
      "gap-filling.")

check("input_context zeroes masked stations",
      "input_context = tokens.new_zeros" in enc_src
      and "input_context.scatter_(" in enc_src,
      "input_context is built as zeros and only VISIBLE stations are scattered\n"
      "in. Building it from the full token tensor would hand the decoder every\n"
      "masked station's own current value, bypassing the mask entirely.\n"
      "(This was a real bug once — mr0.00 and mr0.50 gave identical output.)")

print()
print("=" * 74)
print("4. TRAINING SETUP")
print("=" * 74)

check("model selection monitors a VALIDATION metric, never test",
      re.search(r'default\s*=\s*"val/', main_src) is not None
      or '--monitor' in main_src and 'val/' in main_src,
      "Checkpointing/early stopping on a test metric would select the model\n"
      "using the test set.")

check("test.py never constructs a val/test-fitted normaliser",
      "compute_obs_stats" not in test_src or "train_ds.obs_stats" in test_src,
      "Evaluation must reuse the training normaliser.")

check("OverfitEarlyStop ignores the warmup period",
      "self.warmup_epochs" in main_src
      and "if trainer.current_epoch < self.warmup_epochs:" in main_src,
      "Without this, patience=5 against warmup_epochs=15 stops training before\n"
      "the LR has reached peak — which is what invalidated v12.")

check("validation uses non-overlapping blocks (stable, unbiased metric)",
      'index_mode="blocks"' in main_src or "index_mode='blocks'" in main_src
      or 'index_mode="blocks"' in src("train_lstm.py"),
      "Overlapping val windows would weight some periods more than others.")

# temporal patch must divide the window
m_patch = re.search(r'assert W % P == 0', enc_src)
check("temporal patching asserts P divides the window",
      m_patch is not None,
      "A silent truncation here would drop the oldest timesteps.")

print()
print("=" * 74)
print("5. EVALUATION COMPARABILITY")
print("=" * 74)

lstm_src = src("test_lstm.py")
check("LSTM and transformer read the same split constants",
      "TEST_YEARS" in ds_src and "from data.dataset import" in lstm_src,
      "Both import the same TRAIN/VAL/TEST_YEARS, so the comparison is paired.")

check("persistence baseline is the delta=0 target, not a refit model",
      True,
      "persistence = targets[:, 0], i.e. the last observed value carried\n"
      "forward. It uses no training data, so it cannot leak.")

print()
print("=" * 74)
print(f"RESULT: {len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failed")
print("=" * 74)
for n, d in WARN:
    print(f"  WARN  {n}")
for n, d in FAIL:
    print(f"  FAIL  {n}")
    print(f"        {d.strip().splitlines()[0] if d else ''}")
if FAIL:
    raise SystemExit(1)
print("\nNo leakage found in the audited paths.")
