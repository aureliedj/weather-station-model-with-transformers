"""
test.py

Run a trained Station-MAE checkpoint over the test split (2023-2024) and
write the raw predictions, once per evaluation mask ratio, to

    <save_dir>/<ckpt-stem>_mr<R>/predictions.pt

No metric is computed here; everything is derived from predictions.pt in the
analysis notebooks (notebooks/analysis/).

    python src/test.py --data_root /path/to/PeakWeatherDataset \\
        --checkpoint checkpoints/full_run_cloud_v27/best.ckpt \\
        --test_mask_ratios 0.0 0.5 --save_dir test_results/v27

predictions.pt keys
-------------------
    preds        (M, K, N, 5)   normalised predictions
    targets      (M, K, N, 6)   normalised targets
    masks        (M, K, N, 6)   sensor availability
    masked_idx   (M, n_masked)  stations hidden from the encoder (width 0 at MR 0)
    delta_steps  (M, K)         lead times in 10-min steps
    window_hours (M,)           window start, hours since epoch
    target_hours (M, K)         target time per lead
    spatial      (N, 15)        static station descriptors
    log_var      (M, K, N, 5)   log sigma^2, only for the Gaussian-NLL model
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station-MAE test-set prediction dump")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--checkpoint",  type=str, nargs="+", required=True,
                   help="One or more Lightning .ckpt files; the architecture is read from "
                        "the saved cfg")
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None,
                   help="Default: the checkpoint's own exclusion list")
    p.add_argument("--test_mask_ratios", type=float, nargs="+", default=None,
                   help="Encoder mask ratios to evaluate (default: the trained ratio)")
    p.add_argument("--index_mode",  type=str, default="sliding", choices=["sliding", "blocks"])
    p.add_argument("--stride",      type=int, default=1,
                   help="Window stride in 10-min steps for --index_mode sliding (9 = 90 min)")
    p.add_argument("--fixed_eval_mask", action="store_true",
                   help="Draw one station mask from --seed and use it for every window "
                        "(default: a new mask per sample, seeded once per mask ratio)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--save_predictions", type=int, default=0,
                   help="Cap on the number of windows written (0 = all)")
    p.add_argument("--save_dir",    type=str, default="test_results")
    return p.parse_args()


def _read_cfg(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("hyper_parameters", {}).get("cfg", {})


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"[test.py] device={device}")

    missing = [p for p in args.checkpoint if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] checkpoint(s) not found:\n" + "\n".join(f"  {p}" for p in missing))
    os.makedirs(args.save_dir, exist_ok=True)

    # Data settings come from the first checkpoint; all reported runs share them.
    cfg0 = _read_cfg(args.checkpoint[0])
    if not cfg0:
        raise SystemExit("[ERROR] no hyper_parameters['cfg'] in the checkpoint")
    window    = int(cfg0.get("window", 72))
    max_delta = int(cfg0.get("max_delta", 36))
    stride_k  = int(cfg0.get("delta_grid_stride", 3))
    exclude   = args.exclude_stations or cfg0.get("exclude_stations") or None

    # ── Data ────────────────────────────────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset
    from model.mae import StationMAE
    from engine.evaluate import collect_predictions

    print("Loading PeakWeather dataset ...")
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=window, max_delta_steps=max_delta, delta_grid_stride=stride_k,
                  cache_dir=args.cache_dir or args.data_root, exclude_stations=exclude)
    print("Building train dataset (normalisation statistics only) ...")
    train_ds = StationMAEDataset(ds, split="train", **common)
    print("Building test dataset ...")
    test_ds = StationMAEDataset(ds, split="test", obs_stats=train_ds.obs_stats,
                                index_mode=args.index_mode, train_stride=args.stride, **common)
    print(f"  {len(test_ds):,} test windows (index_mode={args.index_mode}, stride={args.stride}), "
          f"K={len(test_ds.delta_grid)} leads, N={test_ds.spatial.shape[0]} stations")

    persist = args.num_workers > 0
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                             persistent_workers=persist, prefetch_factor=(4 if persist else None))

    # ── Per-checkpoint evaluation ───────────────────────────────────────
    for ckpt_path in args.checkpoint:
        print(f"\n{'=' * 60}\nCheckpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("hyper_parameters", {}).get("cfg", {})

        # Strip the Lightning "model." prefix and torch.compile's "_orig_mod.".
        # decoder.anchor_norm.* is an unused LayerNorm that the training code
        # still instantiated; it is not part of the model here and is dropped.
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("model."):
                k = k[len("model."):]
                if k.startswith("_orig_mod."):
                    k = k[len("_orig_mod."):]
                if k.startswith("decoder.anchor_norm."):
                    continue
                state_dict[k] = v

        # The log-variance head is detected from the weights.
        use_nll = any(k.endswith("decoder.log_var_head.weight") for k in state_dict)
        model = StationMAE.from_cfg(cfg, dropout=0.0, use_nll_loss=use_nll).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"  epoch {ckpt.get('epoch', '?')}  |  {model.count_parameters():,} parameters  |  "
              f"d_model={cfg.get('d_model')} enc_layers={cfg.get('enc_layers')} "
              f"dec_layers={cfg.get('dec_layers')} temporal_patch={cfg.get('temporal_patch')}  |  "
              f"spatial_attn={cfg.get('encoder_spatial_attn')} "
              f"station_local_decoder={bool(cfg.get('station_local_decoder'))}  |  "
              f"trained mask_ratio={cfg.get('mask_ratio')}  |  "
              f"{'Gaussian NLL (log_var saved)' if use_nll else 'Huber'}")

        mask_ratios = args.test_mask_ratios or [model.mask_ratio]
        if model.station_local_decoder:
            # Every station must contribute encoder tokens.
            dropped = [mr for mr in mask_ratios if mr > 0.0]
            mask_ratios = [mr for mr in mask_ratios if mr == 0.0] or [0.0]
            if dropped:
                print(f"  station_local_decoder: skipping mask ratios {dropped}")

        base_label = os.path.splitext(os.path.basename(ckpt_path))[0]
        N = test_ds.spatial.shape[0]

        for mr in mask_ratios:
            model.encoder.mask_ratio = mr
            num_masked = int(N * mr)

            # Seeding once per mask ratio makes the masks reproducible across
            # models evaluated with the same seed, batch size and stride.
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            if args.fixed_eval_mask and num_masked > 0:
                order = torch.argsort(torch.rand(N))
                model.encoder.set_fixed_eval_mask(order[:num_masked].sort().values,
                                                  order[num_masked:].sort().values)
                mask_desc = f"one fixed mask of {num_masked} stations (seed {args.seed})"
            else:
                model.encoder.set_fixed_eval_mask(None, None)
                mask_desc = (f"{num_masked} stations masked per window (seed {args.seed})"
                             if num_masked > 0 else "all stations visible")

            out_dir = os.path.join(args.save_dir, f"{base_label}_mr{mr:.2f}")
            os.makedirs(out_dir, exist_ok=True)
            n = args.save_predictions if args.save_predictions > 0 else len(test_ds)
            print(f"\n  mask_ratio={mr:.2f}: {mask_desc}\n  writing {n:,} windows to {out_dir}")

            t0 = time.time()
            res = collect_predictions(model, test_loader, device, n_windows=n,
                                      save_path=os.path.join(out_dir, "predictions.pt"))
            print(f"  done in {time.time() - t0:.0f}s"
                  + (f"  (log_var saved, shape {tuple(res['log_var'].shape)})"
                     if "log_var" in res else ""))
            del res

    print(f"\nDone. predictions.pt written under {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
