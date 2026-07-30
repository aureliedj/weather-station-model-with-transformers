"""
test_lstm.py

Dump the LSTM baseline's predictions for the WHOLE test split — nothing else.
All metrics and plots are produced in the Jupyter notebook from this file.

Saves (same schema as the transformer's collect_predictions, so it drops straight
into Test_Results_Exploration.ipynb):

    <save_dir>/<run>/best_mr0.00/predictions.pt
        preds        (M, K, N, V_target)   — normalised predictions
        targets      (M, K, N, V)          — normalised ground truth
        masks        (M, K, N, V)          — sensor availability
        delta_steps  (M, K)                — lead-times (10-min steps)
        window_hours (M,)                  — hours-since-epoch of window start
        target_hours (M, K)                — hours-since-epoch of each target
        spatial      (N, 15)               — static station features
        var_names, n_windows

The LSTM has no station masking → pure forecaster → results live under best_mr0.00
(compare against the transformer's mr0.00). Persistence is derivable in the notebook
from targets[:, 0] (the Δ=0 value = last observation), so it is NOT saved here.

A one-line persistence-collapse sanity is printed so a broken run is caught early.

Usage
-----
    python test_lstm.py --data_root /path/to/PeakWeatherDataset \
        --checkpoint checkpoints/lstm-baseline-v1/best.ckpt \
        --exclude_stations PFA --save_dir test_results --run_name lstm-baseline-v1
"""

import argparse
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Renku session containers give /dev/shm a small fixed quota (often 64 MB), which
# the default "file_descriptor" sharing strategy can exhaust when DataLoader workers
# pass batch tensors back to the main process. "file_system" routes that handoff
# through temp files instead of /dev/shm, avoiding "No space left on device" crashes.
mp.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM baseline — dump all test predictions")
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--cache_dir",  type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Windows per step. Effective sequences = batch_size × N (~155); "
                        "keep modest on small-VRAM GPUs (16 fits a 20 GB MIG).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--index_mode", type=str, default="sliding", choices=["blocks", "sliding"])
    p.add_argument("--stride", type=int, default=9,
                   help="Sliding forecast-origin spacing in 10-min steps (9 = 90 min = "
                        "1h30). Only used with --index_mode sliding; ignored for blocks.")
    p.add_argument("--max_windows", type=int, default=0,
                   help="Cap on saved windows (0 = ALL — the default). blocks over "
                        "2023-2024 is ~1,460 windows ≈ 200 MB.")
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

    from data.dataset import load_peakweather, StationMAEDataset
    from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES, TARGET_VARIABLE_NAMES
    Vt = NUM_TARGET_VARIABLES
    cache_dir = args.cache_dir or args.data_root
    window    = g("window", 72)
    max_delta = g("max_delta", 36)
    exclude   = g("exclude_stations", None) or args.exclude_stations

    print("Loading PeakWeather …")
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=window, delta_steps=max_delta, max_delta_steps=max_delta,
                  cache_dir=cache_dir, exclude_stations=exclude, delta_mode="fixed_grid",
                  delta_grid_stride=g("delta_grid_stride", 3))
    # train_ds only supplies the normalisation stats used for the test split
    train_ds = StationMAEDataset(ds, split="train", **common)
    test_ds  = StationMAEDataset(ds, split="test", obs_stats=train_ds.obs_stats,
                                 index_mode=args.index_mode, train_stride=args.stride, **common)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    K = len(test_ds.delta_grid)

    from model.lstm_baseline import StationLSTM
    model = StationLSTM(num_vars=NUM_VARIABLES, num_target_vars=Vt,
                        hidden=g("hidden", 256), num_layers=g("lstm_layers", 3),
                        dropout=0.0, num_horizons=g("num_horizons", K),
                        use_mask_feature=g("use_mask_feature", False))
    sd = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f"Loaded LSTM: {model.count_parameters():,} params  "
          f"(hidden={g('hidden',256)}, layers={g('lstm_layers',3)}, K={K})")

    limit = args.max_windows if args.max_windows > 0 else float("inf")
    keep = {k: [] for k in ("preds", "targets", "masks", "delta_steps",
                            "window_hours", "target_hours")}
    spatial_saved = None
    collected = 0

    print(f"Dumping predictions for {len(test_ds):,} test windows "
          f"(index_mode={args.index_mode}) …")
    try:
        from tqdm import tqdm
        _n_batches = (len(loader) if limit == float("inf")
                      else min(len(loader), -(-int(limit) // args.batch_size)))
        _iter = tqdm(loader, total=_n_batches, desc="predict", unit="batch")
    except ImportError:
        _iter = loader
    with torch.no_grad():
        for batch in _iter:
            if collected >= limit:
                break
            x  = batch["x"].to(device)
            xm = batch["x_mask"].to(device)
            y  = batch["y"]
            ym = batch["y_mask"]
            B, W, N, V = x.shape
            xf  = x.permute(0, 2, 1, 3).reshape(B * N, W, V)
            xmf = xm.permute(0, 2, 1, 3).reshape(B * N, W, V)
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _p = model(xf, xmf)
                _p = _p.float()
            else:
                _p = model(xf, xmf)
            preds = _p.view(B, N, K, Vt).permute(0, 2, 1, 3).cpu()        # (B,K,N,Vt)

            take = B if limit == float("inf") else min(B, int(limit - collected))
            # .clone() the CPU batch tensors before keeping them: with the
            # file_system sharing strategy each worker batch is backed by a
            # mapped temp file that stays open as long as any VIEW of it lives.
            # Keeping raw slices across all ~1.5k batches pins thousands of
            # mapped files → "Too many open files" in _new_shared mid-run.
            # clone() copies into process memory and releases the shared file.
            keep["preds"].append(preds[:take])                   # already a fresh tensor (.cpu() of GPU output)
            keep["targets"].append(y[:take].clone())
            keep["masks"].append(ym[:take].clone())
            keep["delta_steps"].append(batch["delta_steps"][:take].clone())
            keep["window_hours"].append(batch["x_hours"][:take, 0].clone())
            keep["target_hours"].append(batch["y_hours"][:take].clone())
            if spatial_saved is None:
                sp = batch["spatial"]
                spatial_saved = (sp[0] if sp.dim() == 3 else sp).cpu().clone()
            collected += take

    out_dir = os.path.join(args.save_dir, args.run_name, "best_mr0.00")
    os.makedirs(out_dir, exist_ok=True)
    result = {k: torch.cat(v, dim=0) for k, v in keep.items()}
    result["spatial"] = spatial_saved
    result["var_names"] = list(TARGET_VARIABLE_NAMES)
    result["n_windows"] = collected
    pred_path = os.path.join(out_dir, "predictions.pt")
    torch.save(result, pred_path)
    size_mb = os.path.getsize(pred_path) / 1e6
    print(f"Saved {collected:,} windows → {pred_path}  ({size_mb:.0f} MB)")

    # ── one-line persistence-collapse sanity (persistence = targets[:, 0]) ──
    P = result["preds"].numpy()
    T = result["targets"].numpy()[..., :Vt]
    M = result["masks"].numpy()[..., :Vt] > 0.5
    fc = slice(1, K)                                  # forecast horizons Δ>0
    persist = np.repeat(T[:, :1], K, axis=1)          # repeat Δ=0 value
    both = M & M[:, :1]                               # present at Δ=0 AND horizon
    e  = (P[:, fc] - T[:, fc])[both[:, fc]]
    ep = (persist[:, fc] - T[:, fc])[both[:, fc]]
    m  = float(np.sqrt(np.mean(e ** 2)))
    q  = float(np.sqrt(np.mean(ep ** 2)))
    skill = 1 - m / q if q > 0 else float("nan")
    print(f"\nSanity — forecast (Δ>0) normalised RMSE: LSTM {m:.4f} vs persistence {q:.4f} "
          f"→ skill {skill:+.3f}")
    print("✓ not collapsed" if skill > 0.02 else
          "⚠️ LSTM ≈ persistence — check training (lr / grad_clip / epochs).")
    print("\nAll metrics & plots: open Test_Results_Exploration.ipynb.")


if __name__ == "__main__":
    main()
