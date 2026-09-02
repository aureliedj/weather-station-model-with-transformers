"""
test_lstm.py

Run the LSTM baseline over the test split and write its predictions in the
same predictions.pt format as src/test.py, under

    <save_dir>/<run_name>/best_mr0.00/predictions.pt

The LSTM has no station masking, so only the mask-ratio-0 dump exists.

    python src/test_lstm.py --data_root /path/to/PeakWeatherDataset \\
        --checkpoint checkpoints/lstm-baseline-v1/best.ckpt \\
        --save_dir test_results --run_name lstm-baseline-v1
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM baseline test-set prediction dump")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None,
                   help="Default: the checkpoint's own exclusion list")
    p.add_argument("--index_mode",  type=str, default="sliding", choices=["blocks", "sliding"])
    p.add_argument("--stride",      type=int, default=9,
                   help="Window stride in 10-min steps for --index_mode sliding (9 = 90 min)")
    p.add_argument("--batch_size",  type=int, default=8, help="Windows; x N stations folded in")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_windows", type=int, default=0, help="0 = all windows")
    p.add_argument("--save_dir",    type=str, default="test_results")
    p.add_argument("--run_name",    type=str, default="lstm-baseline-v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[test_lstm.py] device={device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("hyper_parameters", {}).get("cfg", {})

    from data.dataset import load_peakweather, StationMAEDataset
    from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES, TARGET_VARIABLE_NAMES
    from model.lstm_baseline import StationLSTM

    Vt      = NUM_TARGET_VARIABLES
    exclude = args.exclude_stations or cfg.get("exclude_stations") or None

    print("Loading PeakWeather dataset ...")
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=int(cfg.get("window", 72)),
                  max_delta_steps=int(cfg.get("max_delta", 36)),
                  delta_grid_stride=int(cfg.get("delta_grid_stride", 3)),
                  cache_dir=args.cache_dir or args.data_root, exclude_stations=exclude)
    train_ds = StationMAEDataset(ds, split="train", **common)
    test_ds  = StationMAEDataset(ds, split="test", obs_stats=train_ds.obs_stats,
                                 index_mode=args.index_mode, train_stride=args.stride, **common)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    K = len(test_ds.delta_grid)
    print(f"  {len(test_ds):,} test windows, K={K} leads, N={test_ds.spatial.shape[0]} stations")

    model = StationLSTM(num_vars=NUM_VARIABLES, num_target_vars=Vt,
                        hidden=cfg.get("hidden", 256), num_layers=cfg.get("lstm_layers", 3),
                        dropout=0.0, horizon_steps=cfg.get("horizon_steps") or test_ds.delta_grid,
                        use_mask_feature=cfg.get("use_mask_feature", False))
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    if "horizon_steps" not in sd:                     # older checkpoint without the buffer
        sd["horizon_steps"] = model.horizon_steps.clone()
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f"StationLSTM: {model.count_parameters():,} parameters, "
          f"epoch {ckpt.get('epoch', '?')}, leads {model.horizon_steps.tolist()} steps")
    assert model.horizon_steps.tolist() == test_ds.delta_grid, \
        "checkpoint lead grid differs from the test dataset grid"

    limit = args.max_windows if args.max_windows > 0 else len(test_ds)
    keep = {k: [] for k in ("preds", "targets", "masks", "delta_steps", "window_hours", "target_hours")}
    spatial_saved, collected = None, 0

    try:
        from tqdm import tqdm
        it = tqdm(loader, total=min(len(loader), -(-limit // args.batch_size)),
                  desc="predict", unit="batch")
    except ImportError:
        it = loader

    with torch.no_grad():
        for batch in it:
            if collected >= limit:
                break
            x, xm = batch["x"].to(device), batch["x_mask"].to(device)
            B, W, N, V = x.shape
            xf  = x.permute(0, 2, 1, 3).reshape(B * N, W, V)
            xmf = xm.permute(0, 2, 1, 3).reshape(B * N, W, V)
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(xf, xmf)
            else:
                p = model(xf, xmf)
            preds = p.float().view(B, N, K, Vt).permute(0, 2, 1, 3).cpu()   # (B, K, N, Vt)

            take = min(B, limit - collected)
            keep["preds"].append(preds[:take])
            keep["targets"].append(batch["y"][:take].clone())
            keep["masks"].append(batch["y_mask"][:take].clone())
            keep["delta_steps"].append(batch["delta_steps"][:take].clone())
            keep["window_hours"].append(batch["x_hours"][:take, 0].clone())
            keep["target_hours"].append(batch["y_hours"][:take].clone())
            if spatial_saved is None:
                sp = batch["spatial"]
                spatial_saved = (sp[0] if sp.dim() == 3 else sp).cpu().clone()
            collected += take

    result = {k: torch.cat(v, dim=0) for k, v in keep.items()}
    result["masked_idx"] = torch.empty(collected, 0, dtype=torch.long)
    result["spatial"]    = spatial_saved
    result["var_names"]  = list(TARGET_VARIABLE_NAMES)
    result["n_windows"]  = collected

    out_dir = os.path.join(args.save_dir, args.run_name, "best_mr0.00")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.pt")
    torch.save(result, path)
    print(f"Saved {collected:,} windows to {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
