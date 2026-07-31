"""Run the new §5 cells against synthetic dumps, with torch stubbed out.

Checks the numerics actually behave: a deliberately overconfident run must show
RMS(z) > 1, and a run whose sigma is pure noise must show rank_corr ~ 0.
"""
import os, sys, types, json

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "Test_Results_Exploration.ipynb")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── stub torch ──────────────────────────────────────────────────────────────
class T:                                    # minimal tensor stand-in
    def __init__(self, a): self.a = np.asarray(a)
    def numpy(self): return self.a
    @property
    def shape(self): return self.a.shape
    def __getitem__(self, k): return T(self.a[k])

DUMPS = {}
torch = types.ModuleType("torch")
torch.load = lambda path, **kw: DUMPS[path]
sys.modules["torch"] = torch

# ── globals the notebook provides in earlier cells ──────────────────────────
rng = np.random.default_rng(1)
Mw, K, N, V = 900, 13, 20, 5
VARS   = ["temperature", "pressure", "humidity", "wind_u", "wind_v"]
UNITS  = {v: "u" for v in VARS}
COLORS = {"v9": "#2E7D8C", "v11": "#D9663D"}
CHUNK  = 300
STD    = np.full((N, V), 2.0)
GRID   = np.arange(0, 37, 3)
LEAD   = [f"+{g}" for g in GRID]
display = lambda x: None                    # notebook builtin

def make_dump(sigma_scale, informative, with_lv=True):
    """Build a synthetic dump. sigma_scale<1 => overconfident."""
    tgt = rng.normal(size=(Mw, K, N, V))
    # true error sd grows with lead time and varies per station
    true_sd = (0.2 + 0.05*np.arange(K))[None,:,None,None] * (0.5 + rng.random((1,1,N,1)))
    err = rng.normal(scale=np.broadcast_to(true_sd, (Mw,K,N,V)))
    pred = tgt + err
    d = dict(preds=T(pred), targets=T(tgt), masks=T(np.ones((Mw,K,N,V))),
             delta_steps=T(np.tile(GRID, (Mw,1))),
             masked_idx=T(rng.integers(0, N, size=(Mw, N//2))))
    if with_lv:
        base = np.broadcast_to(true_sd, (Mw,K,N,V)) if informative \
               else np.full((Mw,K,N,V), float(true_sd.mean()))
        d["log_var"] = T(2*np.log(np.clip(base*sigma_scale, 1e-6, None)))
    return d

DUMPS["p_good"] = make_dump(1.0, True)      # calibrated + informative
DUMPS["p_over"] = make_dump(0.5, False)     # overconfident + uninformative
DUMPS["p_none"] = make_dump(1.0, True, with_lv=False)
RUNS = {"v9": {"mr0.50": "p_good"}, "v11": {"mr0.50": "p_over"},
        "lstm-baseline-v1": {"mr0.00": "p_none"}}

# ── exec the tagged cells ───────────────────────────────────────────────────
nb = json.load(open(sys.argv[1] if len(sys.argv)>1 else NB_PATH))
cells = [ "".join(c["source"]) for c in nb["cells"]
          if c["cell_type"] == "code" and "─UNCERTAINTY─" in "".join(c["source"]) ]
print(f"found {len(cells)} uncertainty cells\n")

g = dict(globals())
for i, src in enumerate(cells):
    print(f"── cell {i} " + "─"*50)
    exec(compile(src, f"<uncert{i}>", "exec"), g)
    plt.close("all")

# ── assertions ──────────────────────────────────────────────────────────────
cal = g["cal"]
print("\n" + "="*66)
good = cal[cal.run.str.startswith("v9")]
over = cal[cal.run.str.startswith("v11")]
print(cal[["run","variable","ratio","RMS z","cov 95%","rank_corr"]].to_string(index=False))

assert len(g["SIG"]) == 2, f"expected 2 NLL runs, got {list(g['SIG'])}"
assert "lstm-baseline-v1" not in [k[0] for k in g["SIG"]], "no-log_var run must be skipped"
assert abs(good["RMS z"].mean() - 1.0) < 0.12, f"calibrated run RMS z={good['RMS z'].mean():.3f}"
assert over["RMS z"].mean() > 1.5, f"overconfident run RMS z={over['RMS z'].mean():.3f}"
assert good["rank_corr"].mean() > 0.25, f"informative sigma rho={good['rank_corr'].mean():.3f}"
assert abs(over["rank_corr"].mean()) < 0.12, f"constant sigma rho={over['rank_corr'].mean():.3f}"
assert 0.90 < good["cov 95%"].mean() < 0.98, good["cov 95%"].mean()
print("\nALL ASSERTIONS PASSED")
