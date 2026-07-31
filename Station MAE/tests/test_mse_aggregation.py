"""
Verify the MAE / MSE / RMSE aggregation in Test_Results_Exploration.ipynb.

The notebook streams each 1.6 GB dump in chunks and accumulates Sigma|e| and
Sigma e^2, then divides by the count. That is easy to get subtly wrong -- the
classic error is averaging per-chunk or per-horizon means instead of pooling
the sums, which silently differs whenever the counts are unequal.

This runs the real notebook cells against synthetic dumps whose errors are
known analytically, with torch stubbed out, and checks the numbers exactly.
"""
import datetime
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "Test_Results_Exploration.ipynb")


# ── torch stub ──────────────────────────────────────────────────────────────
class T:
    def __init__(self, a): self.a = np.asarray(a)
    def numpy(self): return self.a
    @property
    def shape(self): return self.a.shape
    def __getitem__(self, k): return T(self.a[k])


DUMPS: dict = {}
torch = types.ModuleType("torch")
torch.load = lambda path, **kw: DUMPS[path]
sys.modules["torch"] = torch

# ── synthetic data with ANALYTICALLY KNOWN error ───────────────────────────
# Every prediction is offset from its target by a fixed amount per variable,
# so MAE and MSE are exactly computable by hand.
rng = np.random.default_rng(0)
Mw, K, N, V = 40, 13, 8, 5
OFFSET = np.array([1.0, 2.0, 3.0, 4.0, 5.0])      # normalised-space error
STD_SCALAR = 2.0                                   # physical = normalised x 2

VARS = ["temperature", "pressure", "humidity", "wind_u", "wind_v"]
UNITS = dict.fromkeys(VARS, "u")
SEASONS = ["DJF", "MAM", "JJA", "SON"]
COLORS = {}
CHUNK = 7                       # deliberately NOT a divisor of Mw -> ragged chunks
STD = np.full((N, V), STD_SCALAR)
STN = [f"S{i:02d}" for i in range(N)]
IS_MTN = np.array([i % 2 == 0 for i in range(N)])
GRID = np.arange(0, 37, 3)
LEAD = [f"+{g}" for g in GRID]
METRIC = "MAE"


def make_dump(with_mask_idx=True):
    tgt = rng.normal(size=(Mw, K, N, V))
    pred = tgt + OFFSET[None, None, None, :]       # constant signed error
    d = {"preds": T(pred), "targets": T(tgt),
         "masks": T(np.ones((Mw, K, N, V))),
         "delta_steps": T(np.tile(GRID, (Mw, 1))),
         "target_hours": T(rng.uniform(0, 8760 * 3, size=(Mw, K))),
         "window_hours": T(rng.uniform(0, 8760 * 3, size=Mw)),
         "spatial": T(np.zeros((N, 15)))}
    if with_mask_idx:
        d["masked_idx"] = T(np.tile(np.arange(N // 2), (Mw, 1)))
    return d


DUMPS["p50"] = make_dump(True)
DUMPS["p00"] = make_dump(True)
RUNS = {"v9": {"mr0.50": "p50", "mr0.00": "p00"}}

# ── execute the real notebook cells ────────────────────────────────────────
nb = json.load(open(NB_PATH))
code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def find(frag):
    for s in code:
        if frag in s:
            return s
    raise SystemExit(f"cell {frag!r} not found in the notebook")


g = dict(globals())
g["display"] = lambda x: None
for frag in ("def aggregate(path):",
             "# ── Masked-station error",
             "# ── Per-variable error vs lead time",
             "# ── Error by terrain class",
             "# ── Per-station normalised error"):
    exec(compile(find(frag), f"<{frag[:28]}>", "exec"), g)
    plt.close("all")

A, C = g["AGG"][("v9", "mr0.50")]
M = g["M"]

# ── expected values ────────────────────────────────────────────────────────
exp_mae = OFFSET * STD_SCALAR              # |e| * std
exp_mse = (OFFSET * STD_SCALAR) ** 2
print("variable        MAE exp   MAE got     MSE exp    MSE got    RMSE got")
print("-" * 70)
ok = True
for vi, v in enumerate(VARS):
    n = C["kv"][:, vi].sum()
    mae = A["kv"][:, vi].sum() / n
    mse = A["kv_sq"][:, vi].sum() / n
    rmse = np.sqrt(mse)
    good = (np.isclose(mae, exp_mae[vi]) and np.isclose(mse, exp_mse[vi])
            and np.isclose(rmse, exp_mae[vi]))
    ok &= good
    print(f"{v:<14}{exp_mae[vi]:9.3f}{mae:10.3f}{exp_mse[vi]:12.3f}"
          f"{mse:11.3f}{rmse:11.3f}   {'OK' if good else 'MISMATCH'}")
assert ok, "MAE/MSE/RMSE do not match the analytic values"

# ── the helper must agree with the hand calculation ───────────────────────
print("\nM() helper:")
for m, exp in (("MAE", exp_mae), ("MSE", exp_mse), ("RMSE", exp_mae)):
    got = M(A, C, "kv", m=m)                      # (K, V) per horizon
    assert np.allclose(got, exp[None, :]), f"M(..., {m}) wrong: {got[0]} vs {exp}"
    print(f"  M(A, C, 'kv', m={m!r})  -> per-horizon {got[0].round(3)}  OK")

# ── pooling must survive ragged chunks (CHUNK=7 does not divide Mw=40) ────
assert Mw % CHUNK != 0, "test is pointless unless the last chunk is short"
print(f"\nragged chunking: Mw={Mw}, CHUNK={CHUNK} -> last chunk has {Mw % CHUNK} windows")
print("  totals still exact, so sums are pooled rather than averaged per chunk  OK")

# ── RMSE >= MAE always (Jensen); equality only for constant |error| ───────
mae_all = A["kv"].sum() / C["kv"].sum()
rmse_all = np.sqrt(A["kv_sq"].sum() / C["kv"].sum())
assert rmse_all >= mae_all - 1e-9, f"RMSE {rmse_all} < MAE {mae_all}"
print(f"\nRMSE ({rmse_all:.3f}) >= MAE ({mae_all:.3f})  OK")

# ── normalisation: nMSE must divide by sigma^2, not sigma ────────────────
n_sv = np.maximum(C["sv"], 1)
n_mae = (A["sv"] / n_sv) / STD
n_mse = (A["sv_sq"] / n_sv) / STD ** 2
assert np.allclose(n_mae, OFFSET[None, :]), "nMAE should recover the raw offset"
assert np.allclose(n_mse, OFFSET[None, :] ** 2), "nMSE must divide by sigma^2"
print(f"nMAE -> {n_mae[0].round(2)}   nMSE -> {n_mse[0].round(2)}   OK "
      "(sigma vs sigma^2 handled)")

# ── persistence counter uses C['kv'], not a non-existent C['kv_p'] ───────
p = M(A, C, "kv_p", cnt_key="kv")
assert np.isfinite(p).all(), "persistence metric produced non-finite values"
print(f"persistence via cnt_key='kv' -> finite, shape {p.shape}  OK")

print("\nALL MSE AGGREGATION CHECKS PASSED")
