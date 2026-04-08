"""
main.py

Entry point for Station-MAE training.

Usage
-----
    # Full training run (default config)
    python main.py --data_root /path/to/peakweather

    # Custom hyperparameters
    python main.py --data_root /path/to/peakweather \\
                   --d_model 256 --enc_layers 6 --dec_layers 2 \\
                   --epochs 100 --lr 1e-4 --save_dir checkpoints/run1

    # Multi-delta training (K=6 lead-times per sample, up to 3 h ahead)
    python main.py --data_root /path/to/peakweather --num_delta 6 --max_delta 18

    # Quick pipeline check on 2 years of data (fast end-to-end sanity run)
    python main.py --data_root /path/to/peakweather --subset --epochs 5

    # Custom subset: specific training years
    python main.py --data_root /path/to/peakweather --train_years 2019 2020 2021

    # Resume from checkpoint
    python main.py --data_root /path/to/peakweather --resume checkpoints/run1/best.pt

    # Exploration / quick sanity check (much faster)
    python explore_pipeline.py --data_root /path/to/peakweather

Arguments
---------
  Data
    --data_root   STR   Path to PeakWeather data directory (required)
    --cache_dir   STR   Directory for the pre-built tensor cache (default: same as data_root)
    --window      INT   Input window size in 10-min steps (default 288 = 48 h)
    --num_workers INT   DataLoader worker processes (default 4)
    --batch_size  INT   Training batch size (default 16)

  Model
    --d_model     INT   Model dimension (default 128)
    --enc_heads   INT   Encoder attention heads (default 4)
    --enc_layers  INT   Encoder transformer depth (default 4)
    --dec_heads   INT   Decoder attention heads (default 4)
    --dec_layers  INT   Decoder transformer depth (default 2)
    --mlp_ratio   FLT   FFN hidden-dim ratio (default 4.0)
    --dropout     FLT   Dropout rate (default 0.1)
    --mask_ratio  FLT   Fraction of stations masked per sample (default 0.5)
    --max_delta   INT   Max forecast horizon in 10-min steps (default 18 = 3 h)

  Training
    --num_delta   INT   Lead-times sampled per training sample (default 1; max=max_delta)
    --epochs      INT   Number of training epochs (default 100)
    --lr          FLT   Peak learning rate (default 1e-4)
    --weight_decay FLT  AdamW weight decay (default 0.05)
    --warmup_epochs INT Warmup epochs (default 5)
    --grad_clip   FLT   Gradient clipping max norm (default 1.0)
    --amp               Enable automatic mixed precision (flag, CUDA/MPS)
    --device      STR   'cpu', 'cuda', or 'mps' (default: auto)
    --seed        INT   Random seed (default 42)

  Early stopping
    --patience    INT   Early-stop patience in epochs; 0 = disabled (default 10)
    --min_delta   FLT   Min val_loss improvement to reset ES counter (default 1e-4)

  Checkpointing
    --save_dir    STR   Directory for checkpoints and logs (default 'checkpoints')
    --save_every  INT   Save a numbered snapshot every N epochs (default 5)
    --resume      STR   Path to checkpoint .pt to resume from (optional)
    --log_interval INT  Steps between training log lines (default 50)
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station-MAE training")

    # Data
    p.add_argument("--data_root",    type=str,   required=True)
    p.add_argument("--cache_dir",    type=str,   default=None,
                   help="Directory for tensor cache; defaults to data_root")
    p.add_argument("--window",       type=int,   default=288,
                   help="Input window in 10-min steps (default 288 = 48 h)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--batch_size",   type=int,   default=16)

    # Subset / quick-check mode
    p.add_argument("--subset",       action="store_true",
                   help="Quick-check mode: train on 2 years only (2020–2021). "
                        "Overridden by --train_years if both are given.")
    p.add_argument("--train_years",  type=int,   nargs="+", default=None,
                   metavar="YEAR",
                   help="Custom list of training years, e.g. --train_years 2020 2021. "
                        "Defaults to 2017–2021. Val/test years are always fixed.")

    # Model
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--enc_heads",    type=int,   default=4)
    p.add_argument("--enc_layers",   type=int,   default=4)
    p.add_argument("--dec_heads",    type=int,   default=4)
    p.add_argument("--dec_layers",   type=int,   default=2)
    p.add_argument("--mlp_ratio",    type=float, default=4.0)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--mask_ratio",   type=float, default=0.5)
    p.add_argument("--max_delta",    type=int,   default=18,
                   help="Max forecast horizon in 10-min steps (default 18 = 3 h)")

    # Training
    p.add_argument("--num_delta",    type=int,   default=1,
                   help="Lead-times sampled per sample (1=single-delta, >1=multi-delta)")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=42)

    # Early stopping
    p.add_argument("--patience",     type=int,   default=10,
                   help="Early-stop patience in epochs; 0 = disabled")
    p.add_argument("--min_delta",    type=float, default=1e-4,
                   help="Min val_loss improvement to reset ES counter")

    # Checkpointing
    p.add_argument("--save_dir",     type=str,   default="checkpoints")
    p.add_argument("--save_every",   type=int,   default=5,
                   help="Save a numbered checkpoint every N epochs")
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--log_interval", type=int,   default=50)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Device & seed -------------------------------------------------
    if args.device is None:
        device = torch.device(
            "mps"  if torch.backends.mps.is_available() and torch.backends.mps.is_built() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)

    set_seed(args.seed)
    print(f"[Station-MAE]  device={device}  seed={args.seed}")

    # ---- Data ----------------------------------------------------------
    from data.dataset import load_peakweather, StationMAEDataset, TRAIN_YEARS

    cache_dir = args.cache_dir if args.cache_dir is not None else args.data_root

    # Resolve training years (--subset, --train_years, or default)
    if args.train_years is not None:
        train_years = sorted(args.train_years)
    elif args.subset:
        train_years = [2020, 2021]   # last 2 years of the default training window
    else:
        train_years = TRAIN_YEARS    # full 2017–2021

    if train_years != TRAIN_YEARS:
        print(f"[Subset mode]  Training years: {train_years}  "
              f"(full set would be {TRAIN_YEARS})")

    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    print("Building train dataset …")
    train_ds = StationMAEDataset(
        ds,
        window_size=args.window,
        delta_steps=args.max_delta,
        split="train",
        num_delta_per_sample=args.num_delta,
        max_delta_steps=args.max_delta,
        cache_dir=cache_dir,
        train_years=train_years,
    )

    print("Building val dataset …")
    val_ds = StationMAEDataset(
        ds,
        window_size=args.window,
        delta_steps=args.max_delta,
        split="val",
        obs_stats=train_ds.obs_stats,       # normalise with training-split stats
        num_delta_per_sample=1,             # single-delta for consistent val metrics
        max_delta_steps=args.max_delta,
        cache_dir=cache_dir,
        train_years=train_years,            # keeps stats consistent
    )

    print(f"  train: {len(train_ds):,} samples  "
          f"(years {train_years[0]}–{train_years[-1]})  |  val: {len(val_ds):,} samples")
    print(f"  num_delta (K) = {args.num_delta}  |  max_delta = {args.max_delta} steps "
          f"({args.max_delta * 10 // 60} h {args.max_delta * 10 % 60} min lead-time)")

    _use_persistent = (args.num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=_use_persistent,
        prefetch_factor=(4 if _use_persistent else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_use_persistent,
        prefetch_factor=(4 if _use_persistent else None),
    )

    # ---- Model ---------------------------------------------------------
    from model.mae import StationMAE

    model = StationMAE(
        d_model=args.d_model,
        enc_heads=args.enc_heads,
        enc_layers=args.enc_layers,
        dec_heads=args.dec_heads,
        dec_layers=args.dec_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        # Note: max_delta_steps is NOT passed to StationMAE — DeltaTimeEmbedding
        # uses continuous Fourier encoding over hours so no upper bound is needed.
    ).to(device)

    print(f"Model: {model.count_parameters():,} trainable parameters")

    # ---- Resume --------------------------------------------------------
    start_epoch = 1
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}  (epoch {ckpt['epoch']})")

    # ---- Training config dict ------------------------------------------
    cfg = {
        # Optimiser
        "lr":            args.lr,
        "weight_decay":  args.weight_decay,
        # Schedule
        "epochs":        args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip":     args.grad_clip,
        # Early stopping
        "patience":      args.patience,
        "min_delta":     args.min_delta,
        # Checkpointing / logging
        "save_dir":      args.save_dir,
        "save_every":    args.save_every,
        "log_interval":  args.log_interval,
        # Hardware
        "amp":           args.amp,
        # Informational (used in logs)
        "num_delta":     args.num_delta,
    }

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Run training --------------------------------------------------
    from engine.train import train

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
