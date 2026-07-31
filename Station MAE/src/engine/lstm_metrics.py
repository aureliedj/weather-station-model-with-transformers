"""
engine/lstm_metrics.py

Pure-numpy metric aggregation for the LSTM baseline evaluation (test_lstm.py).

Kept dependency-free (numpy only, no torch) so it can be unit-tested directly.
Given per-(horizon, variable) sums of squared / absolute / signed error and counts
— for the model (normalised + physical) and for persistence (normalised) — it
produces the flat metric dict written to test_metrics.csv (same field names as the
transformer's test.py) and the persistence_metrics.csv dict.
"""

import numpy as np


def aggregate_metrics(n_sse, n_sae, n_bias, p_sse, p_sae, p_bias, q_sse, cnt,
                      grid, var_names):
    """
    Args (all shape (K, Vt) unless noted):
        n_sse, n_sae, n_bias : model NORMALISED sum sq-err / abs-err / signed-err
        p_sse, p_sae, p_bias : model PHYSICAL   sum sq-err / abs-err / signed-err
        q_sse                : persistence NORMALISED sum sq-err
        cnt                  : count of supervised entries
        grid                 : list of K delta-step values (e.g. [0, 3, …, 36])
        var_names            : list of Vt target-variable names

    Returns:
        row         : dict — one flat row for test_metrics.csv
        persist_row : dict — {metric: value} for persistence_metrics.csv
    """
    K  = len(grid)
    Vt = len(var_names)

    def agg(sse, c):
        return float(np.sqrt(sse.sum() / max(c.sum(), 1)))

    row = {}
    # per-variable (over all deltas): normalised + physical
    for v, name in enumerate(var_names):
        cs = max(cnt[:, v].sum(), 1)
        row[f"{name}_norm_rmse"] = float(np.sqrt(n_sse[:, v].sum() / cs))
        row[f"{name}_norm_mae"]  = float(n_sae[:, v].sum() / cs)
        row[f"{name}_norm_bias"] = float(n_bias[:, v].sum() / cs)
        row[f"{name}_rmse"]      = float(np.sqrt(p_sse[:, v].sum() / cs))
        row[f"{name}_mae"]       = float(p_sae[:, v].sum() / cs)
        row[f"{name}_bias"]      = float(p_bias[:, v].sum() / cs)

    # overall
    row["overall_rmse_norm"] = agg(n_sse, cnt)
    row["overall_mae_norm"]  = float(n_sae.sum() / max(cnt.sum(), 1))
    row["overall_rmse"]      = agg(p_sse, cnt)
    row["overall_mae"]       = float(p_sae.sum() / max(cnt.sum(), 1))

    # per-delta overall + per-variable (normalised), and skill vs persistence
    persist_row = {}
    for k in range(K):
        dd = grid[k]
        ck = cnt[k]
        d_norm = float(np.sqrt(n_sse[k].sum() / max(ck.sum(), 1)))
        q_norm = float(np.sqrt(q_sse[k].sum() / max(ck.sum(), 1)))
        row[f"delta_{dd:02d}_overall_rmse_norm"] = d_norm
        for v, name in enumerate(var_names):
            row[f"delta_{dd:02d}_{name}_rmse_norm"] = float(
                np.sqrt(n_sse[k, v] / max(cnt[k, v], 1)))
        row[f"skill_delta_{dd:02d}"] = (1 - d_norm / q_norm) if q_norm > 0 else float("nan")
        persist_row[f"persist_delta_{dd:02d}_overall_rmse_norm"] = q_norm

    # per-variable skill + overall skill (normalised, over all deltas)
    for v, name in enumerate(var_names):
        m_n = float(np.sqrt(n_sse[:, v].sum() / max(cnt[:, v].sum(), 1)))
        q_n = float(np.sqrt(q_sse[:, v].sum() / max(cnt[:, v].sum(), 1)))
        row[f"skill_{name}"] = (1 - m_n / q_n) if q_n > 0 else float("nan")
        persist_row[f"persist_{name}_rmse_norm"] = q_n

    q_overall = agg(q_sse, cnt)
    row["skill_overall"] = (1 - row["overall_rmse_norm"] / q_overall) if q_overall > 0 else float("nan")
    persist_row["persist_overall_rmse_norm"] = q_overall

    return row, persist_row
