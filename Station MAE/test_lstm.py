"""
test_lstm.py

Evaluate the per-station LSTM baseline on the test split and save its predictions
in the SAME format as engine/evaluate.collect_predictions (test.py), so the LSTM
drops straight into the comparison notebooks alongside the transformer runs.

Because the LSTM has no masking, it is a pure forecaster — compare it against the
transformer's mr0.00 numbers.  Results are written to
  <save_dir>/<run>/best_mr0.00/predictions.pt   (+ lstm_metrics.csv)

It also computes the PERSISTENCE baseline on the same windows and prints a
side-by-side table, with an explicit check that the LSTM has not collapsed to
persistence (an RNN that just echoes the last input).

Usage
-----
    python test_lstm.py --data_root /path/to/PeakWeatherDataset \
        --checkpoint checkpoints/lstm_baseline/best.ckpt \
        --exclude_stations PFA --save_dir test_results --run_name lstm-v1
"""

import argparse
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
    p.add_argument("--run_name",  type=str, default="lstm-v1")
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
    K   = len(test_ds.delta_grid)
    Vt  = NUM_TARGET_VARIABLES

    # per-station physical std (kept stations), for metrics only
    _obs = train_ds.obs_stats
    _keep = getattr(test_ds, "_keep_indices", None)
    if _keep is not None and _obs["std"].dim() == 2:
        std_ps = _obs["std"][_keep][:, :Vt].to(device)          # (N, Vt)
    else:
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
    print(f"Loaded LSTM: {model.count_parameters():,} params")

    # ── Inference: collect predictions + persistence + running metrics ────
    sse = np.zeros((K, Vt)); sae = np.zeros((K, Vt)); cnt = np.zeros((K, Vt))       # LSTM
    sse_p = np.zeros((K, Vt)); cnt_p = np.zeros((K, Vt))                             # persistence
    keep = {k: [] for k in ("preds", "targets", "masks", "delta_steps",
                            "window_hours", "target_hours")}
    spatial_saved = None
    collected = 0

    with torch.no_grad():
        for batch in loader:
            x  = batch["x"].to(device)          # (B, W, N, V)
            xm = batch["x_mask"].to(device)
            y  = batch["y"].to(device)          # (B, K, N, V)
            ym = batch["y_mask"].to(device)     # (B, K, N, V)
            B, W, N, V = x.shape

            xf  = x.permute(0, 2, 1, 3).reshape(B * N, W, V)
            xmf = xm.permute(0, 2, 1, 3).reshape(B * N, W, V)
            pf  = model(xf, xmf)                                    # (B*N, K, Vt)
            preds = pf.view(B, N, K, Vt).permute(0, 2, 1, 3)       # (B, K, N, Vt)

            # persistence: repeat last observed input value across horizons
            persist = x[:, -1, :, :Vt].unsqueeze(1).expand(B, K, N, Vt)

            tt = y[:, :, :, :Vt]
            mm = ym[:, :, :, :Vt].bool()
            std = std_ps.view(1, 1, N, Vt) if std_ps.dim() == 2 else std_ps.view(1, 1, 1, Vt)
            err   = (preds   - tt) * std
            err_p = (persist - tt) * std
            for k in range(K):
                mk = mm[:, k]                                       # (B, N, Vt)
                e  = err[:, k][mk]; ep = err_p[:, k][mk]
                # accumulate per-variable via masked sums
                for v in range(Vt):
                    mv = mm[:, k, :, v]
                    if mv.any():
                        ev  = err[:, k, :, v][mv]; epv = err_p[:, k, :, v][mv]
                        sse[k, v] += float((ev ** 2).sum()); sae[k, v] += float(ev.abs().sum())
                        cnt[k, v] += int(mv.sum())
                        sse_p[k, v] += float((epv ** 2).sum()); cnt_p[k, v] += int(mv.sum())

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
    out_dir = os.path.join(args.save_dir, args.run_name, "best_mr0.00")
    os.makedirs(out_dir, exist_ok=True)
    result = {k: torch.cat(v, dim=0) for k, v in keep.items()}
    result["spatial"]   = spatial_saved
    result["var_names"] = list(TARGET_VARIABLE_NAMES)
    result["n_windows"] = collected
    pred_path = os.path.join(out_dir, "predictions.pt")
    torch.save(result, pred_path)
    print(f"Saved {collected} windows → {pred_path}")

    # ── Metrics table (physical) + persistence-collapse check ─────────────
    rmse   = np.sqrt(sse.sum(1) / cnt.sum(1))          # (K,) overall physical RMSE per horizon
    mae    = sae.sum(1) / cnt.sum(1)
    rmse_p = np.sqrt(sse_p.sum(1) / cnt_p.sum(1))
    import csv
    csv_path = os.path.join(args.save_dir, args.run_name, "lstm_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_min", "lstm_rmse", "lstm_mae", "persist_rmse", "skill_vs_persist"])
        for k in range(K):
            dmin = int(test_ds.delta_grid[k]) * 10
            skill = 1 - rmse[k] / rmse_p[k] if rmse_p[k] > 0 else float("nan")
            w.writerow([dmin, f"{rmse[k]:.4f}", f"{mae[k]:.4f}", f"{rmse_p[k]:.4f}", f"{skill:.4f}"])
    print(f"Saved metrics → {csv_path}\n")

    print(f"{'lead':>7} {'LSTM RMSE':>10} {'persist':>10} {'skill':>8}")
    for k in range(K):
        dmin = int(test_ds.delta_grid[k]) * 10
        skill = 1 - rmse[k] / rmse_p[k] if rmse_p[k] > 0 else float("nan")
        print(f"{dmin:>6}m {rmse[k]:>10.4f} {rmse_p[k]:>10.4f} {skill:>+8.3f}")

    # collapse check: forecast horizons (skip delta=0) — LSTM should beat persistence
    fc = slice(1, K)
    lstm_fc = np.sqrt(sse[fc].sum() / cnt[fc].sum())
    pers_fc = np.sqrt(sse_p[fc].sum() / cnt_p[fc].sum())
    gain = 1 - lstm_fc / pers_fc
    print(f"\nForecast-horizon (Δ>0) overall: LSTM {lstm_fc:.4f} vs persistence {pers_fc:.4f} "
          f"→ skill {gain:+.3f}")
    if gain <= 0.02:
        print("⚠️  LSTM barely beats persistence (skill ≤ 2%) — likely COLLAPSED to "
              "echoing the last input. Check training (lr, grad_clip, epochs).")
    else:
        print("✓ LSTM meaningfully beats persistence — not collapsed.")


if __name__ == "__main__":
    main()
