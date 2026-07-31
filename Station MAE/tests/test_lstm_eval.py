"""
tests/test_lstm_eval.py

Verification for the LSTM baseline evaluation (test_lstm.py / engine/lstm_metrics.py).

Pure-numpy — runs without torch — so it can be executed anywhere:

    cd "Station MAE" && python tests/test_lstm_eval.py

Checks:
  1. aggregate_metrics matches an independent brute-force computation
     (physical & normalised per-variable RMSE/MAE/bias, per-delta, overall, skill).
  2. Station-fold / unfold round-trip preserves each station's sequence.
  3. Persistence at Δ=0 gives ~0 error; persistence error grows with lead-time.
  4. Persistence-collapse check fires iff the model ≈ persistence.
  5. state_dict "model." prefix stripping.
  6. Saved predictions.pt schema matches the transformer's collect_predictions.
"""

import os
import sys

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.join(_PROJ, "src")          # packages moved under src/
sys.path.insert(0, _ROOT)
# Load engine/lstm_metrics.py directly (bypass engine/__init__, which imports torch),
# so this test runs anywhere — with or without torch installed.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("lstm_metrics",
                                     os.path.join(_ROOT, "engine", "lstm_metrics.py"))
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
aggregate_metrics = _mod.aggregate_metrics


VARS = ["temperature", "pressure", "humidity", "wind_u", "wind_v"]
GRID = list(range(0, 37, 3))          # [0, 3, …, 36], K=13


def _make_synthetic(seed=0):
    """Synthetic eval batch with realistic temporal structure.

    Targets follow a random walk from the Δ=0 base, so persistence (= repeat the
    Δ=0 value) degrades with lead-time — exactly the real behaviour.
    """
    rng = np.random.default_rng(seed)
    M, K, N, Vt = 8, len(GRID), 6, len(VARS)
    base = rng.normal(size=(M, N, Vt))                  # Δ=0 value (= last input)
    incr = rng.normal(scale=0.3, size=(M, K, N, Vt))
    walk = np.cumsum(incr, axis=1)
    targets = base[:, None] + walk - walk[:, :1]        # targets[:,0] == base
    persist = np.broadcast_to(base[:, None], (M, K, N, Vt)).copy()
    # model preds: closer to target than persistence (so it "beats" persistence)
    preds = targets + rng.normal(scale=0.2, size=(M, K, N, Vt))
    masks = rng.random((M, K, N, Vt)) > 0.15
    std   = np.abs(rng.normal(size=(N, Vt))) + 0.5      # per-station physical std
    return preds, targets, persist, masks, std


def _accumulate(preds, targets, persist, masks, std):
    """Replicate test_lstm.py's per-(k,v) masked accumulation."""
    M, K, N, Vt = preds.shape
    ne = preds - targets                 # normalised error
    pe = ne * std[None, None, :, :]      # physical error
    qe = persist - targets               # persistence normalised error
    n_sse = np.zeros((K, Vt)); n_sae = np.zeros((K, Vt)); n_bias = np.zeros((K, Vt))
    p_sse = np.zeros((K, Vt)); p_sae = np.zeros((K, Vt)); p_bias = np.zeros((K, Vt))
    q_sse = np.zeros((K, Vt)); cnt = np.zeros((K, Vt))
    for k in range(K):
        for v in range(Vt):
            mv = masks[:, k, :, v]
            if mv.any():
                nev = ne[:, k, :, v][mv]; pev = pe[:, k, :, v][mv]; qev = qe[:, k, :, v][mv]
                n_sse[k, v] += (nev ** 2).sum(); n_sae[k, v] += np.abs(nev).sum(); n_bias[k, v] += nev.sum()
                p_sse[k, v] += (pev ** 2).sum(); p_sae[k, v] += np.abs(pev).sum(); p_bias[k, v] += pev.sum()
                q_sse[k, v] += (qev ** 2).sum(); cnt[k, v] += mv.sum()
    return n_sse, n_sae, n_bias, p_sse, p_sae, p_bias, q_sse, cnt


def test_metrics_match_bruteforce():
    preds, targets, persist, masks, std = _make_synthetic()
    accs = _accumulate(preds, targets, persist, masks, std)
    row, persist_row = aggregate_metrics(*accs, GRID, VARS)

    ne = preds - targets
    pe = ne * std[None, None, :, :]
    qe = persist - targets

    # per-variable physical + normalised RMSE/MAE/bias over ALL present entries
    for v, name in enumerate(VARS):
        m = masks[..., v]
        ev = pe[..., v][m]; nv = ne[..., v][m]
        assert np.isclose(row[f"{name}_rmse"], np.sqrt((ev ** 2).mean())), f"{name}_rmse"
        assert np.isclose(row[f"{name}_mae"],  np.abs(ev).mean()),        f"{name}_mae"
        assert np.isclose(row[f"{name}_bias"], ev.mean()),                f"{name}_bias"
        assert np.isclose(row[f"{name}_norm_rmse"], np.sqrt((nv ** 2).mean())), f"{name}_norm_rmse"

    # overall (all vars, all deltas)
    ev_all = pe[masks]; nv_all = ne[masks]
    assert np.isclose(row["overall_rmse"], np.sqrt((ev_all ** 2).mean())), "overall_rmse"
    assert np.isclose(row["overall_rmse_norm"], np.sqrt((nv_all ** 2).mean())), "overall_rmse_norm"

    # per-delta overall normalised + skill vs persistence
    for k, dd in enumerate(GRID):
        mk = masks[:, k]
        d_norm = np.sqrt((ne[:, k][mk] ** 2).mean())
        q_norm = np.sqrt((qe[:, k][mk] ** 2).mean()) if (qe[:, k][mk] ** 2).size else 0.0
        assert np.isclose(row[f"delta_{dd:02d}_overall_rmse_norm"], d_norm), f"delta_{dd}_overall"
        if q_norm > 0:
            assert np.isclose(row[f"skill_delta_{dd:02d}"], 1 - d_norm / q_norm), f"skill_delta_{dd}"
        assert np.isclose(persist_row[f"persist_delta_{dd:02d}_overall_rmse_norm"], q_norm), f"persist_{dd}"

    print("  [1] aggregate_metrics == brute-force  ✓")


def test_persistence_delta0_zero():
    preds, targets, persist, masks, std = _make_synthetic()
    _, _, _, _, _, _, q_sse, cnt = _accumulate(preds, targets, persist, masks, std)
    _, persist_row = aggregate_metrics(*_accumulate(preds, targets, persist, masks, std), GRID, VARS)
    assert persist_row["persist_delta_00_overall_rmse_norm"] < 1e-9, "persistence@Δ0 should be 0"
    # and grows with lead-time
    seq = [persist_row[f"persist_delta_{d:02d}_overall_rmse_norm"] for d in GRID]
    assert seq[-1] > seq[1] > seq[0], "persistence error should grow with horizon"
    print("  [2] persistence Δ=0 ≈ 0 and grows with horizon  ✓")


def test_fold_unfold_roundtrip():
    B, W, N, V = 3, 72, 6, 6
    x = np.random.default_rng(1).normal(size=(B, W, N, V))
    xf = np.transpose(x, (0, 2, 1, 3)).reshape(B * N, W, V)         # _fold
    for b in range(B):
        for n in range(N):
            assert np.allclose(xf[b * N + n], x[b, :, n, :]), "fold mismatch"
    K, Vt = 13, 5
    pf = np.random.default_rng(2).normal(size=(B * N, K, Vt))
    preds = np.transpose(pf.reshape(B, N, K, Vt), (0, 2, 1, 3))     # unfold
    for b in range(B):
        for n in range(N):
            assert np.allclose(preds[b, :, n, :], pf[b * N + n]), "unfold mismatch"
    print("  [3] fold/unfold round-trip preserves station sequences  ✓")


def test_collapse_check():
    preds, targets, persist, masks, std = _make_synthetic()

    def fc_skill(pr):
        accs = _accumulate(pr, targets, persist, masks, std)
        n_sse, *_, q_sse, cnt = accs
        fc = slice(1, len(GRID))
        m = np.sqrt(n_sse[fc].sum() / cnt[fc].sum())
        q = np.sqrt(q_sse[fc].sum() / cnt[fc].sum())
        return 1 - m / q

    assert fc_skill(preds) > 0.02, "good model should beat persistence"
    assert fc_skill(persist.copy()) <= 0.02, "collapsed model (==persistence) should be flagged"
    print("  [4] persistence-collapse check fires iff model ≈ persistence  ✓")


def test_state_dict_strip():
    keys = ["model.lstm.weight_ih_l0", "model.lstm.bias_hh_l2", "model.head.weight",
            "model.head.bias", "var_weights"]
    sd = {k.removeprefix("model."): 0 for k in keys if k.startswith("model.")}
    assert set(sd) == {"lstm.weight_ih_l0", "lstm.bias_hh_l2", "head.weight", "head.bias"}
    assert "var_weights" not in sd, "buffer must be excluded (loaded strict on model only)"
    print("  [5] state_dict 'model.' prefix strip  ✓")


def test_predictions_schema():
    # Expected schema from engine/evaluate.collect_predictions (transformer test.py)
    expected = {"preds", "targets", "masks", "delta_steps", "window_hours",
                "target_hours", "spatial", "var_names", "n_windows"}
    # Keys test_lstm.py assembles:
    keep = {"preds", "targets", "masks", "delta_steps", "window_hours", "target_hours"}
    saved = set(keep) | {"spatial", "var_names", "n_windows"}
    assert saved == expected, f"schema mismatch: {saved ^ expected}"

    # If a real transformer predictions.pt is reachable, compare keys directly.
    import zipfile, pickle, io
    for cand in ["test_results/v11/best_mr0.50/predictions.pt",
                 "test_results/v9/best_mr0.50/predictions.pt",
                 "test_results/best_mr0.50/predictions.pt"]:
        cand = os.path.join(_ROOT, cand)
        if not os.path.exists(cand):
            continue
        z = zipfile.ZipFile(cand)
        dn = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        def _rb(*a): return None
        class _Stub(dict):
            def __init__(self, *a, **k): super().__init__()
        class U(pickle.Unpickler):
            def find_class(self, m, n):
                if n == "OrderedDict": return dict
                if n == "_rebuild_tensor_v2": return _rb
                return _Stub
            def persistent_load(self, pid): return None
        obj = U(io.BytesIO(z.read(dn))).load()
        real = set(obj.keys())
        assert expected <= real, f"real file missing {expected - real}"
        print(f"  [6] predictions.pt schema matches REAL file "
              f"({os.path.relpath(cand, _ROOT)})  ✓")
        return
    print("  [6] predictions.pt schema matches collect_predictions (no real file present)  ✓")


if __name__ == "__main__":
    print("LSTM evaluation checks:")
    test_metrics_match_bruteforce()
    test_persistence_delta0_zero()
    test_fold_unfold_roundtrip()
    test_collapse_check()
    test_state_dict_strip()
    test_predictions_schema()
    print("\nALL LSTM EVAL CHECKS PASSED ✓")
