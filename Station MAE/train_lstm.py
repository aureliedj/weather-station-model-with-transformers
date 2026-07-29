"""
train_lstm.py

Train the per-station LSTM baseline (model/lstm_baseline.py) on PeakWeather,
reusing StationMAEDataset so the data, normalisation, splits and forecast-horizon
grid are IDENTICAL to the transformer — an apples-to-apples baseline.

The station dimension is folded into the batch inside StationLSTMLightning, so no
masking / spatial / temporal-embedding machinery is involved.  Compare the result
against the transformer's mr0.00 (all-stations-visible) metrics.

Usage
-----
    python train_lstm.py --data_root /path/to/PeakWeatherDataset \
        --hidden 256 --lstm_layers 3 --epochs 50 --lr 1e-3 --grad_clip 1.0 \
        --wandb_project station-mae --wandb_run_name lstm-baseline-v1

See run_lstm_cloud.sh for the full cloud configuration.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-station LSTM baseline training")

    # Data (mirror the transformer so the comparison is fair)
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache")
    p.add_argument("--window",      type=int, default=72)      # 12 h @ 10 min
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=16)      # windows; ×N stations folded in
    p.add_argument("--max_delta",   type=int, default=36)      # 6 h
    p.add_argument("--delta_grid_stride", type=int, default=3)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--index_mode",  type=str, default="random",
                   choices=["sliding", "blocks", "random"])
    p.add_argument("--random_epoch_size", type=int, default=10000)
    p.add_argument("--train_years", type=int, nargs="+", default=None)
    p.add_argument("--subset",      action="store_true")

    # Model
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--lstm_layers",  type=int,   default=3)
    p.add_argument("--lstm_dropout", type=float, default=0.1)
    p.add_argument("--use_mask_feature", action="store_true",
                   help="Concatenate the sensor mask to the input (default off = vanilla).")

    # Loss (same family as the transformer)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--var_weights", type=float, nargs=5, default=[1.0, 1.0, 1.0, 1.0, 1.0],
                   metavar=("T", "P", "RH", "U", "V"),
                   help="Per-variable Huber weights [temp, pressure, humidity, wind_u, wind_v]. "
                        "Default uniform for a neutral baseline.")

    # Training
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--min_lr",       type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_epochs",type=int,   default=3)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--optimizer",    type=str,   default="adamw", choices=["adamw", "rmsprop"])
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--seed",         type=int,   default=42)

    # Early stopping
    p.add_argument("--monitor",   type=str,   default="val/overall_rmse")
    p.add_argument("--patience",  type=int,   default=15)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--overfit_stop", action="store_true")
    p.add_argument("--overfit_patience", type=int, default=5)

    # Checkpointing / logging
    p.add_argument("--save_dir",       type=str, default="checkpoints/lstm_baseline")
    p.add_argument("--resume",         type=str, default=None)
    p.add_argument("--limit_val_batches", type=int, default=0)
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_offline",  action="store_true")
    p.add_argument("--log_every_n_steps", type=int, default=50)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from data.dataset import load_peakweather, StationMAEDataset, TRAIN_YEARS
    cache_dir = args.cache_dir or args.data_root
    if args.train_years is not None:
        train_years = sorted(args.train_years)
    elif args.subset:
        train_years = [2020, 2021]
    else:
        train_years = TRAIN_YEARS

    print("Loading PeakWeather …")
    ds = load_peakweather(root=args.data_root)
    fast_cache = args.local_cache_dir.strip() or None

    common = dict(window_size=args.window, delta_steps=args.max_delta,
                  max_delta_steps=args.max_delta, cache_dir=cache_dir,
                  train_years=train_years, shared_memory=False,
                  fast_cache_dir=fast_cache, exclude_stations=args.exclude_stations,
                  delta_mode="fixed_grid", delta_grid_stride=args.delta_grid_stride)

    train_ds = StationMAEDataset(ds, split="train", index_mode=args.index_mode,
                                 random_epoch_size=args.random_epoch_size, **common)
    val_ds   = StationMAEDataset(ds, split="val", obs_stats=train_ds.obs_stats, **common)

    K = len(train_ds.delta_grid)
    print(f"  train {len(train_ds):,} windows | val {len(val_ds):,} | K={K} horizons")

    _use_cuda = torch.cuda.is_available()
    _persist  = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=_use_cuda and _persist, persistent_workers=_persist,
                              prefetch_factor=(4 if _persist else None))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=_use_cuda and _persist, persistent_workers=_persist,
                              prefetch_factor=(4 if _persist else None))

    # ── Model ────────────────────────────────────────────────────────────
    from model.lstm_baseline import StationLSTM, StationLSTMLightning
    from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES

    model = StationLSTM(num_vars=NUM_VARIABLES, num_target_vars=NUM_TARGET_VARIABLES,
                        hidden=args.hidden, num_layers=args.lstm_layers,
                        dropout=args.lstm_dropout, num_horizons=K,
                        use_mask_feature=args.use_mask_feature)
    print(f"StationLSTM: {model.count_parameters():,} params  "
          f"(hidden={args.hidden}, layers={args.lstm_layers}, K={K})")

    # per-station std for physical val metrics (sliced to kept stations, like main.py)
    _obs = train_ds.obs_stats
    _keep = getattr(train_ds, "_keep_indices", None)
    if _keep is not None and _obs["std"].dim() == 2:
        _std_list = _obs["std"][_keep].tolist()
    else:
        _std_list = _obs["std"].tolist()

    cfg = dict(lr=args.lr, min_lr=args.min_lr, weight_decay=args.weight_decay,
               warmup_epochs=args.warmup_epochs, epochs=args.epochs,
               optimizer=args.optimizer, huber_delta=args.huber_delta,
               var_weights=list(args.var_weights), obs_stats_std=_std_list,
               hidden=args.hidden, lstm_layers=args.lstm_layers,
               lstm_dropout=args.lstm_dropout, window=args.window,
               max_delta=args.max_delta, delta_grid_stride=args.delta_grid_stride,
               num_horizons=K, use_mask_feature=args.use_mask_feature,
               exclude_stations=args.exclude_stations or [])
    lit = StationLSTMLightning(model=model, cfg=cfg)

    # ── Trainer ──────────────────────────────────────────────────────────
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from main import OverfitEarlyStop   # reuse the overfit-aware stopper

    if args.wandb_project:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name,
                             config=vars(args), save_dir=args.save_dir)
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=args.save_dir, name="logs")

    os.makedirs(args.save_dir, exist_ok=True)
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
                                          min_delta=0.0, mode="min", train_metric="train/loss"))

    _mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if args.amp and _use_cuda:
        precision = "bf16-mixed" if args.bf16 else "16-mixed"
    elif args.amp and _mps:
        precision = "16-mixed"
    else:
        precision = "32-true"

    _tk = {}
    if args.limit_val_batches > 0:
        _tk["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto",
                         precision=precision, gradient_clip_val=args.grad_clip,
                         callbacks=callbacks, logger=logger,
                         log_every_n_steps=args.log_every_n_steps,
                         enable_progress_bar=True, **_tk)
    print(f"\n[LSTM-baseline]  precision={precision}  optimizer={args.optimizer}  "
          f"monitor={args.monitor}\nSaving to: {args.save_dir}\n")
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume)
    print(f"\nDone. Checkpoints in {args.save_dir}")
    if args.wandb_project:
        import wandb; wandb.finish()


if __name__ == "__main__":
    main()
