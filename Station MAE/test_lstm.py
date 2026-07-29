"""
test_lstm.py

Evaluate the per-station LSTM baseline on the test split, writing outputs in the
SAME layout as the transformer's test.py so the LSTM drops straight into the
comparison / diagnostic notebooks:

    <save_dir>/<run>/best_mr0.00/predictions.pt   (same schema as collect_predictions)
    <save_dir>/<run>/test_metrics.csv             (same schema as test.py — per-variable
                                                    normalised + physical RMSE/MAE/bias,
                                                    per-delta, skill vs persistence)
    <save_dir>/<run>/persistence_metrics.csv      (persist_delta_XX_overall_rmse_norm, …)

The LSTM has no masking, so it is a pure forecaster → its results live under
best_mr0.00 and should be compared against the transformer's mr0.00 numbers.
The evaluation also prints a persistence-collapse check.

Usage
-----
    python test_lstm.py --data_root /path/to/PeakWeatherDataset \
        --checkpoint checkpoints/lstm-baseline-v1/best.ckpt \
        --exclude_stations PFA --save_dir test_results --run_name lstm-baseline-v1
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM baseline evaluation")
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--cache_dir",  type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--index_mode", type=str, default="blocks", choices=["blocks", "sliding"])
    p.add_argument("--save_predictions", type=int, default=200,
                   help="Number of test windows to dump to predictions.pt (matches test.py).")
    p.add_argument("--save_dir",  type=str, default="test_results")
    p.add_argument("--run_name",  type=str, default="lstm-baseline-v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get("hyper_parameters", {}).get("cfg", {})
    def g(k, d): return cfg.get(k, d)

    # ── Data (must match training exactly) ────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset
    from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES, TARGET_VARIABLE_NAMES
    VARS = list(TARGET_VARIABLE_NAMES)
    Vt   = NUM_TARGET_VARIABLES
    cache_dir = args.cache_dir or args.data_root
    window    = g("window", 72)
    max_delta = g("max_delta", 36)
    exclude   = g("exclude_stations", None) or args.exclude_stations

    print("Loading PeakWeather …")
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=window, delta_steps=max_delta, max_delta_steps=max_delta,
                  cache_dir=cache_dir, exclude_stations=exclude, delta_mode="fixed_grid",
                  delta_grid_stride=g("delta_grid_stride", 3))
    train_ds = StationMAEDataset(ds, split="train", **common)
    test_ds  = StationMAEDataset(ds, split="test", obs_stats=train_ds.obs_stats,
                                 index_mode=args.index_mode, **common)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    K    = len(test_ds.delta_grid)
    GRID = [int(d) for d in test_ds.delta_grid]

    # per-station physical std, aligned to the prediction station axis.
    _obs = train_ds.obs_stats
    _keep = getattr(test_ds, "_keep_indices", None)
    if _obs["std"].dim() == 2:                                   # (N_full, V) per-station
        _rows = _keep if _keep is not None else slice(None)      # slice kept stations (or all)
        std_ps = _obs["std"][_rows][:, :Vt].to(device)          # (N, Vt)
    else:                                                        # (V,) global 1-D
        std_ps = _obs["std"][:Vt].to(device)                    # (Vt,)

    # ── Model ─────────────────────────────────────────────────────────────
    from model.lstm_baseline import StationLSTM
    model = StationLSTM(num_vars=NUM_VARIABLES, num_target_vars=Vt,
                        hidden=g("hidden", 256), num_layers=g("lstm_layers", 3),
                        dropout=0.0, num_horizons=g("num_horizons", K),
                        use_mask_feature=g("use_mask_feature", False))
    sd = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f"Loaded LSTM: {model.count_parameters():,} params  (hidden={g('hidden',256)}, "
          f"layers={g('lstm_layers',3)}, K={K})")

    # ── Accumulators: per (delta k, variable v) ───────────────────────────
    n_sse = np.zeros((K, Vt)); n_sae = np.zeros((K, Vt)); n_bias = np.zeros((K, Vt))  # LSTM norm
    p_sse = np.zeros((K, Vt)); p_sae = np.zeros((K, Vt)); p_bias = np.zeros((K, Vt))  # LSTM phys
    q_sse = np.zeros((K, Vt))                                                          # persist norm
    cnt   = np.zeros((K, Vt))
    keep = {k: [] for k in ("preds", "targets", "masks", "delta_steps",
                            "window_hours", "target_hours")}
    spatial_saved = None
    collected = 0

    with torch.no_grad():
        for batch in loader:
            x  = batch["x"].to(device)          # (B, W, N, V)
            xm = batch["x_mask"].to(device)
            y  = batch["y"].to(device)          # (B, K, N, V)
            ym = batch["y_mask"].to(device)
            B, W, N, V = x.shape

            xf  = x.permute(0, 2, 1, 3).reshape(B * N, W, V)
            xmf = xm.permute(0, 2, 1, 3).reshape(B * N, W, V)
            preds = model(xf, xmf).view(B, N, K, Vt).permute(0, 2, 1, 3)   # (B,K,N,Vt)
            persist = x[:, -1, :, :Vt].unsqueeze(1).expand(B, K, N, Vt)

            tt  = y[:, :, :, :Vt]
            mm  = ym[:, :, :, :Vt].bool()
            std = std_ps.view(1, 1, N, Vt) if std_ps.dim() == 2 else std_ps.view(1, 1, 1, Vt)
            ne  = preds - tt                    # normalised error
            pe  = ne * std                      # physical error
            qe  = persist - tt                  # persistence normalised error

            for k in range(K):
                for v in range(Vt):
                    mv = mm[:, k, :, v]
                    if mv.any():
                        nev = ne[:, k, :, v][mv]; pev = pe[:, k, :, v][mv]; qev = qe[:, k, :, v][mv]
                        n_sse[k, v] += float((nev ** 2).sum()); n_sae[k, v] += float(nev.abs().sum())
                        n_bias[k, v] += float(nev.sum())
                        p_sse[k, v] += float((pev ** 2).sum()); p_sae[k, v] += float(pev.abs().sum())
                        p_bias[k, v] += float(pev.sum())
                        q_sse[k, v] += float((qev ** 2).sum())
                        cnt[k, v]   += int(mv.sum())

            if collected < args.save_predictions:
                take = min(B, args.save_predictions - collected)
                keep["preds"].append(preds[:take].cpu())
                keep["targets"].append(y[:take].cpu())
                keep["masks"].append(ym[:take].cpu())
                keep["delta_steps"].append(batch["delta_steps"][:take])
                keep["window_hours"].append(batch["x_hours"][:take, 0])
                keep["target_hours"].append(batch["y_hours"][:take])
                if spatial_saved is None:
                    sp = batch["spatial"]
                    spatial_saved = (sp[0] if sp.dim() == 3 else sp).cpu()
                collected += take

    # ── Save predictions.pt (same schema as collect_predictions) ──────────
    run_dir = os.path.join(args.save_dir, args.run_name)
    out_dir = os.path.join(run_dir, "best_mr0.00")
    os.makedirs(out_dir, exist_ok=True)
    result = {k: torch.cat(v, dim=0) for k, v in keep.items()}
    result["spatial"] = spatial_saved
    result["var_names"] = VARS
    result["n_windows"] = collected
    torch.save(result, os.path.join(out_dir, "predictions.pt"))
    print(f"Saved {collected} windows → {os.path.join(out_dir, 'predictions.pt')}")

    # ── Aggregate metrics (same field names as test.py) ───────────────────
    def agg(sse, c):  return float(np.sqrt(sse.sum() / max(c.sum(), 1)))

    row = {"checkpoint": args.run_name}
    # per-variable (over all deltas): normalised + physical
    for v, name in enumerate(VARS):
        c = cnt[:, v]
        row[f"{name}_norm_rmse"] = float(np.sqrt(n_sse[:, v].sum() / max(c.sum(), 1)))
        row[f"{name}_norm_mae"]  = float(n_sae[:, v].sum() / max(c.sum(), 1))
        row[f"{name}_norm_bias"] = float(n_bias[:, v].sum() / max(c.sum(), 1))
        row[f"{name}_rmse"]      = float(np.sqrt(p_sse[:, v].sum() / max(c.sum(), 1)))
        row[f"{name}_mae"]       = float(p_sae[:, v].sum() / max(c.sum(), 1))
        row[f"{name}_bias"]      = float(p_bias[:, v].sum() / max(c.sum(), 1))
    # overall
    row["overall_rmse_norm"] = agg(n_sse, cnt)
    row["overall_mae_norm"]  = float(n_sae.sum() / max(cnt.sum(), 1))
    row["overall_rmse"]      = agg(p_sse, cnt)
    row["overall_mae"]       = float(p_sae.sum() / max(cnt.sum(), 1))
    # per-delta overall + per-variable (normalised), and skill vs persistence
    persist_row = {}
    for k in range(K):
        dd = GRID[k]; ck = cnt[k]
        d_norm = float(np.sqrt(n_sse[k].sum() / max(ck.sum(), 1)))
        q_norm = float(np.sqrt(q_sse[k].sum() / max(ck.sum(), 1)))
        row[f"delta_{dd:02d}_overall_rmse_norm"] = d_norm
        for v, name in enumerate(VARS):
            row[f"delta_{dd:02d}_{name}_rmse_norm"] = float(
                np.sqrt(n_sse[k, v] / max(cnt[k, v], 1)))
        row[f"skill_delta_{dd:02d}"] = (1 - d_norm / q_norm) if q_norm > 0 else float("nan")
        persist_row[f"persist_delta_{dd:02d}_overall_rmse_norm"] = q_norm
    # per-variable skill + overall skill (normalised, over all deltas)
    for v, name in enumerate(VARS):
        m_n = float(np.sqrt(n_sse[:, v].sum() / max(cnt[:, v].sum(), 1)))
        q_n = float(np.sqrt(q_sse[:, v].sum() / max(cnt[:, v].sum(), 1)))
        row[f"skill_{name}"] = (1 - m_n / q_n) if q_n > 0 else float("nan")
        persist_row[f"persist_{name}_rmse_norm"] = q_n
    q_overall = agg(q_sse, cnt)
    row["skill_overall"] = (1 - row["overall_rmse_norm"] / q_overall) if q_overall > 0 else float("nan")
    persist_row["persist_overall_rmse_norm"] = q_overall

    with open(os.path.join(run_dir, "test_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    with open(os.path.join(run_dir, "persistence_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, vv in persist_row.items():
            w.writerow([k, f"{vv:.6f}"])
    print(f"Saved test_metrics.csv + persistence_metrics.csv → {run_dir}")

    # ── Summary table + persistence-collapse check ────────────────────────
    print(f"\n{'lead':>7} {'LSTM RMSE(phys)':>16} {'persist(norm)':>14} {'skill':>8}")
    for k in range(K):
        dmin = GRID[k] * 10
        d_phys = float(np.sqrt(p_sse[k].sum() / max(cnt[k].sum(), 1)))
        skill  = row[f"skill_delta_{GRID[k]:02d}"]
        print(f"{dmin:>6}m {d_phys:>16.4f} "
              f"{persist_row[f'persist_delta_{GRID[k]:02d}_overall_rmse_norm']:>14.4f} {skill:>+8.3f}")

    fc = slice(1, K)
    m_fc = float(np.sqrt(n_sse[fc].sum() / max(cnt[fc].sum(), 1)))
    q_fc = float(np.sqrt(q_sse[fc].sum() / max(cnt[fc].sum(), 1)))
    gain = 1 - m_fc / q_fc if q_fc > 0 else float("nan")
    print(f"\nForecast horizons (Δ>0): LSTM {m_fc:.4f} vs persistence {q_fc:.4f} "
          f"(normalised) → skill {gain:+.3f}")
    if gain <= 0.02:
        print("⚠️  LSTM barely beats persistence (≤2%) — likely COLLAPSED to echoing the "
              "last input. Check lr / grad_clip / epochs.")
    else:
        print("✓ LSTM meaningfully beats persistence — not collapsed.")


if __name__ == "__main__":
    main()
