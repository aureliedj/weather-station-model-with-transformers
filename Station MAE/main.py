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

    # Resume from checkpoint
    python main.py --data_root /path/to/peakweather --resume checkpoints/run1/best.pt

    # Exploration / quick sanity check (much faster)
    python explore_pipeline.py --data_root /path/to/peakweather

Arguments
---------
  Data
    --data_root   STR   Path to PeakWeather data directory (required)
    --window      INT   Input window size in 10-min steps (default 12 = 2 h)
    --delta       INT   Forecast lead-time in 10-min steps (default 6 = 1 h)
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
    --max_delta   INT   Max forecast horizon in 10-min steps for DeltaTimeEmbedding (default 36)

  Training
    --epochs      INT   Number of training epochs (default 100)
    --lr          FLT   Peak learning rate (default 1e-4)
    --weight_decay FLT  AdamW weight decay (default 0.05)
    --warmup_epochs INT Warmup epochs (default 5)
    --grad_clip   FLT   Gradient clipping max norm (default 1.0)
    --amp               Enable automatic mixed precision (flag, CUDA only)
    --device      STR   'cpu' or 'cuda' (default: auto)
    --seed        INT   Random seed (default 42)

  Checkpointing
    --save_dir    STR   Directory for checkpoints (default 'checkpoints')
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
    p.add_argument("--window",       type=int,   default=12)
    p.add_argument("--delta",        type=int,   default=6)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--batch_size",   type=int,   default=16)

    # Model
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--enc_heads",    type=int,   default=4)
    p.add_argument("--enc_layers",   type=int,   default=4)
    p.add_argument("--dec_heads",    type=int,   default=4)
    p.add_argument("--dec_layers",   type=int,   default=2)
    p.add_argument("--mlp_ratio",    type=float, default=4.0)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--mask_ratio",   type=float, default=0.5)
    p.add_argument("--max_delta",    type=int,   default=36)

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=42)

    # Checkpointing
    p.add_argument("--save_dir",     type=str,   default="checkpoints")
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)
    print(f"[Station-MAE]  device={device}  seed={args.seed}")

    # ---- Data ----------------------------------------------------------
    from data.dataset import load_peakweather, StationMAEDataset

    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    print("Building train dataset …")
    train_ds = StationMAEDataset(
        ds,
        window_size=args.window,
        delta_steps=args.delta,
        split="train",
    )

    print("Building val dataset …")
    val_ds = StationMAEDataset(
        ds,
        window_size=args.window,
        delta_steps=args.delta,
        split="val",
        obs_stats=train_ds.obs_stats,   # always normalise with training stats
    )

    print(f"  train: {len(train_ds):,} samples  |  val: {len(val_ds):,} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
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
        max_delta_steps=args.max_delta,
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
        "lr":            args.lr,
        "weight_decay":  args.weight_decay,
        "epochs":        args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip":     args.grad_clip,
        "log_interval":  args.log_interval,
        "save_dir":      args.save_dir,
        "amp":           args.amp,
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
