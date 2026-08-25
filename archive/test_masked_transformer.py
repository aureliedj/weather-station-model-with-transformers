"""
test_masked_transformer.py — dump masked-transformer predictions.

Two scenarios in the shared schema, so the run drops straight into
Test_Results_Exploration.ipynb next to v15 / simple-MAE / LSTM:

    best_mr0.00/predictions.pt   all 155 stations observed (pure forecasting)
    best_mr0.50/predictions.pt   50% of stations corrupted for the whole
                                 window; masked_idx records which ones, so the
                                 notebook can score gap-filling separately.

Schema (identical to test.py / test_lstm.py / test_simple_mae.py):
    preds (M,K,N,5) · targets (M,K,N,6) · masks (M,K,N,6) · masked_idx (M,n)
    delta_steps (M,K) · window_hours (M,) · target_hours (M,K) · spatial (N,15)

Δ=0 note: unlike the other models, k=0 here is a REAL trained output for
masked stations (it is the gap-filling target) and an untrained one for
visible stations (the loss skips them there — supervising a station's own
last observation would reward copying).
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Masked-transformer prediction dump")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--index_mode",  type=str, default="sliding",
                   choices=["sliding", "blocks"])
    p.add_argument("--stride",      type=int, default=9)
    p.add_argument("--max_windows", type=int, default=0, help="0 = ALL")
    p.add_argument("--mask_ratio",  type=float, default=0.5,
                   help="Corrupted-station fraction for the mr0.50 scenario")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--save_dir",    type=str, default="test_results")
    p.add_argument("--run_name",    type=str, default="masked-tf-v1")
    p.add_argument("--global_norm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]["cfg"]
    from model.masked_transformer import MaskedStationTransformer
    model = MaskedStationTransformer(
        window_steps=cfg["window"], temporal_patch=cfg["temporal_patch"],
        horizon_steps=cfg.get("delta_grid"),
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        mlp_ratio=cfg["mlp_ratio"], dropout=0.0, mask_ratio=cfg["mask_ratio"],
        huber_delta=cfg.get("huber_delta", 1.0), pool=cfg.get("pool", "last"),
    )
    sd = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)      # simple model → strict
    model = model.to(device).eval()
    print(f"Loaded MaskedStationTransformer: {model.count_parameters():,} params "
          f"(epoch {ckpt.get('epoch', '?')}) · K={model.K} leads {model.horizon_steps}")

    from data.dataset import load_peakweather, StationMAEDataset
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=cfg["window"], delta_steps=cfg["max_delta"],
                  max_delta_steps=cfg["max_delta"],
                  cache_dir=args.cache_dir or args.data_root, shared_memory=False,
                  fast_cache_dir=args.local_cache_dir.strip() or None,
                  exclude_stations=args.exclude_stations, delta_mode="fixed_grid",
                  delta_grid_stride=cfg["delta_grid_stride"],
                  global_norm=args.global_norm)
    train_ds = StationMAEDataset(ds, split="train", **common)        # stats only
    test_ds  = StationMAEDataset(ds, split="test", obs_stats=train_ds.obs_stats,
                                 index_mode=args.index_mode,
                                 train_stride=args.stride, **common)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"Test windows: {len(test_ds):,} ({args.index_mode}, stride={args.stride})")

    limit = args.max_windows if args.max_windows > 0 else float("inf")
    gen   = torch.Generator().manual_seed(args.seed)

    def run(tag: str, ratio: float) -> None:
        keep = {k: [] for k in ("preds", "targets", "masks", "masked_idx",
                                "delta_steps", "window_hours", "target_hours")}
        spatial_saved, collected = None, 0
        try:
            from tqdm import tqdm
            it = tqdm(loader, desc=f"predict {tag}", unit="batch")
        except ImportError:
            it = loader
        with torch.no_grad():
            for batch in it:
                if collected >= limit:
                    break
                x  = batch["x"].to(device); xm = batch["x_mask"].to(device)
                sp = batch["spatial"]; sp = (sp[0] if sp.dim() == 3 else sp).to(device)
                xh = batch["x_hours"].to(device)
                B, _, N, _ = x.shape

                n_mask = int(N * ratio)
                smask = torch.zeros(B, N, dtype=torch.bool, device=device)
                if n_mask:
                    order = torch.rand(B, N, generator=gen).argsort(dim=1)
                    midx  = order[:, :n_mask].to(device)
                    smask.scatter_(1, midx, True)
                else:
                    midx = torch.empty(B, 0, dtype=torch.long, device=device)

                amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if device.type == "cuda" else torch.autocast("cpu", enabled=False))
                with amp:
                    preds, _ = model(x, xm, sp, xh, station_mask=smask)
                preds = preds.float()

                take = B if limit == float("inf") else min(B, int(limit - collected))
                keep["preds"].append(preds[:take].cpu().clone())
                keep["targets"].append(batch["y"][:take].clone())
                keep["masks"].append(batch["y_mask"][:take].clone())
                keep["masked_idx"].append(midx[:take].cpu().clone())
                keep["delta_steps"].append(batch["delta_steps"][:take].clone())
                keep["window_hours"].append(batch["x_hours"][:take, 0].clone())
                keep["target_hours"].append(batch["y_hours"][:take].clone())
                if spatial_saved is None:
                    spatial_saved = sp.cpu().clone()
                collected += take

        res = {k: torch.cat(v, dim=0) for k, v in keep.items()}
        res["spatial"]   = spatial_saved
        res["var_names"] = ["temperature", "pressure", "humidity", "wind_u", "wind_v"]
        res["n_windows"] = collected
        out = os.path.join(args.save_dir, args.run_name, tag)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "predictions.pt")
        torch.save(res, path)
        print(f"[{tag}] saved {collected:,} windows → {path} "
              f"({os.path.getsize(path)/1e6:.0f} MB)")

        P = res["preds"].numpy(); T = res["targets"].numpy()[..., :5]
        M = res["masks"].numpy()[..., :5] > 0.5
        K = P.shape[1]
        per = np.repeat(T[:, :1], K, axis=1); both = M & M[:, :1]
        e  = (P[:, 1:] - T[:, 1:])[both[:, 1:]]
        ep = (per[:, 1:] - T[:, 1:])[both[:, 1:]]
        rm, rp = float(np.sqrt((e**2).mean())), float(np.sqrt((ep**2).mean()))
        disp = min(float(P[:, 1:, :, v][M[:, 1:, :, v]].std()
                         / max(T[:, 1:, :, v][M[:, 1:, :, v]].std(), 1e-6))
                   for v in range(5))
        skill = 1 - rm / rp if rp > 0 else float("nan")
        print(f"[{tag}] sanity — Δ>0 RMSE {rm:.4f} vs persistence {rp:.4f} "
              f"→ skill {skill:+.3f} | dispersion_min {disp:.2f}")
        print("[{}] {}".format(tag, "✓ looks sane" if skill > -0.5 and disp > 0.3
                               else "⚠ check the model before trusting this dump"))

    run("best_mr0.00", 0.0)
    run("best_mr0.50", args.mask_ratio)
    print("\nDone. Add the run to CMP_RUNS in Test_Results_Exploration.ipynb.")


if __name__ == "__main__":
    main()
