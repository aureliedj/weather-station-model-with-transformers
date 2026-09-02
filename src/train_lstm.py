"""
train_lstm.py

Train the per-station LSTM baseline on the same data, splits, normalisation
and lead-time grid as the Transformer (see model/lstm_baseline.py).

    python src/train_lstm.py --data_root /path/to/PeakWeatherDataset [options]

The configuration of the reported run is in scripts/train_lstm.sh.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-station LSTM baseline training")

    # Data (same options as main.py)
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache")
    p.add_argument("--window",      type=int, default=72)
    p.add_argument("--max_delta",   type=int, default=36)
    p.add_argument("--delta_grid_stride", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=16, help="Windows; x N stations folded in")
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--train_years", type=int, nargs="+", default=None)
    p.add_argument("--subset",      action="store_true")
    p.add_argument("--index_mode",  type=str, default="sliding",
                   choices=["sliding", "blocks", "random"])
    p.add_argument("--train_stride", type=int, default=1)
    p.add_argument("--random_epoch_size", type=int, default=None)

    # Model
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--lstm_layers",  type=int,   default=3)
    p.add_argument("--lstm_dropout", type=float, default=0.1)
    p.add_argument("--use_mask_feature", action="store_true",
                   help="Concatenate the sensor mask to the input")

    # Loss
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--var_weights", type=float, nargs=5, default=[1.0] * 5,
                   metavar=("TEMP", "PRES", "HUM", "WIND_U", "WIND_V"))

    # Optimisation
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--min_lr",        type=float, default=1e-6)
    p.add_argument("--weight_decay",  type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int,   default=3)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--bf16",          action="store_true")
    p.add_argument("--seed",          type=int,   default=42)

    # Early stopping / checkpoints / logging
    p.add_argument("--monitor",   type=str,   default="val/overall_mae")
    p.add_argument("--patience",  type=int,   default=15)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--overfit_stop", action="store_true")
    p.add_argument("--overfit_patience", type=int, default=5)
    p.add_argument("--save_dir",  type=str, default="checkpoints/lstm_baseline")
    p.add_argument("--resume",    type=str, default=None)
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_offline",  action="store_true")
    p.add_argument("--log_every_n_steps", type=int, default=50)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from data.dataset import load_peakweather, StationMAEDataset, TRAIN_YEARS
    from model.lstm_baseline import StationLSTM, StationLSTMLightning
    from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES
    from main import OverfitEarlyStop

    if args.train_years is not None:
        train_years = sorted(args.train_years)
    elif args.subset:
        train_years = [2020, 2021]
    else:
        train_years = TRAIN_YEARS

    print("Loading PeakWeather dataset ...")
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=args.window, max_delta_steps=args.max_delta,
                  delta_grid_stride=args.delta_grid_stride,
                  cache_dir=args.cache_dir or args.data_root,
                  fast_cache_dir=args.local_cache_dir.strip() or None,
                  train_years=train_years, exclude_stations=args.exclude_stations)
    train_ds = StationMAEDataset(ds, split="train", index_mode=args.index_mode,
                                 train_stride=args.train_stride,
                                 random_epoch_size=args.random_epoch_size, **common)
    val_ds   = StationMAEDataset(ds, split="val", obs_stats=train_ds.obs_stats,
                                 index_mode="blocks", **common)
    K = len(train_ds.delta_grid)
    print(f"  train {len(train_ds):,} windows | val {len(val_ds):,} | K={K} leads "
          f"{train_ds.delta_grid} steps")

    use_cuda = torch.cuda.is_available()
    persist  = args.num_workers > 0
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=use_cuda and persist, persistent_workers=persist,
                     prefetch_factor=(4 if persist else None))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = StationLSTM(num_vars=NUM_VARIABLES, num_target_vars=NUM_TARGET_VARIABLES,
                        hidden=args.hidden, num_layers=args.lstm_layers,
                        dropout=args.lstm_dropout, horizon_steps=train_ds.delta_grid,
                        use_mask_feature=args.use_mask_feature)
    print(f"StationLSTM: {model.count_parameters():,} parameters "
          f"(hidden={args.hidden}, layers={args.lstm_layers}, K={K})")

    obs_stats = train_ds.obs_stats
    keep = train_ds._keep_indices
    std  = obs_stats["std"][keep] if keep is not None else obs_stats["std"]

    cfg = dict(lr=args.lr, min_lr=args.min_lr, weight_decay=args.weight_decay,
               warmup_epochs=args.warmup_epochs, epochs=args.epochs,
               huber_delta=args.huber_delta, var_weights=list(args.var_weights),
               obs_stats_std=std.tolist(), hidden=args.hidden, lstm_layers=args.lstm_layers,
               lstm_dropout=args.lstm_dropout, window=args.window, max_delta=args.max_delta,
               delta_grid_stride=args.delta_grid_stride, horizon_steps=list(train_ds.delta_grid),
               num_horizons=K, use_mask_feature=args.use_mask_feature,
               index_mode=args.index_mode, train_stride=args.train_stride,
               exclude_stations=args.exclude_stations or [])
    lit = StationLSTMLightning(model=model, cfg=cfg)

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

    os.makedirs(args.save_dir, exist_ok=True)
    logger = CSVLogger(save_dir=args.save_dir, name="logs")
    if args.wandb_project:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name,
                                 config=vars(args), save_dir=args.save_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] WandB init failed ({e}); using the CSV logger.")

    callbacks = [
        ModelCheckpoint(dirpath=args.save_dir, filename="best", monitor=args.monitor,
                        mode="min", save_top_k=1, save_last=True, verbose=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor=args.monitor, patience=args.patience,
                                       min_delta=args.min_delta, mode="min", verbose=True))
    if args.overfit_stop:
        callbacks.append(OverfitEarlyStop(monitor=args.monitor, patience=args.overfit_patience,
                                          warmup_epochs=args.warmup_epochs))

    mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if args.amp and use_cuda:
        precision = "bf16-mixed" if args.bf16 else "16-mixed"
    elif args.amp and mps:
        precision = "16-mixed"
    else:
        precision = "32-true"

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto",
                         precision=precision, gradient_clip_val=args.grad_clip,
                         callbacks=callbacks, logger=logger,
                         log_every_n_steps=args.log_every_n_steps, enable_progress_bar=True)
    print(f"\n[LSTM baseline] precision={precision} monitor={args.monitor}\n"
          f"Saving checkpoints to: {args.save_dir}\n")
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume)
    print(f"\nTraining complete. Checkpoints saved to: {args.save_dir}")
    if args.wandb_project:
        try:
            import wandb
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
