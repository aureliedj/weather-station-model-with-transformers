"""
main.py

Train the Station-MAE Transformer with PyTorch Lightning.

    python src/main.py --data_root /path/to/PeakWeatherDataset [options]

The configurations of the reported runs are in scripts/train_transformer.sh.
Architecture and data settings are saved in the checkpoint under
``hyper_parameters["cfg"]`` and read back by src/test.py.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# File-based tensor sharing between DataLoader workers: avoids exhausting
# /dev/shm in containers with a small shared-memory quota.
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station-MAE training")

    # Data
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None,
                   help="Raw-tensor cache directory (default: data_root)")
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache",
                   help="Memory-mapped per-split cache; '' disables it")
    p.add_argument("--window",      type=int, default=72, help="Input steps (72 = 12 h)")
    p.add_argument("--max_delta",   type=int, default=36, help="Longest lead in steps (36 = 6 h)")
    p.add_argument("--delta_grid_stride", type=int, default=3, help="Lead spacing in steps")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None, metavar="ABBR")
    p.add_argument("--train_years", type=int, nargs="+", default=None, metavar="YEAR")
    p.add_argument("--subset",      action="store_true", help="Train on 2020-2021 only")
    p.add_argument("--index_mode",  type=str, default="sliding",
                   choices=["sliding", "blocks", "random"])
    p.add_argument("--train_stride", type=int, default=1)
    p.add_argument("--random_epoch_size", type=int, default=None)

    # Model
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--enc_heads",    type=int,   default=4)
    p.add_argument("--enc_layers",   type=int,   default=4)
    p.add_argument("--dec_heads",    type=int,   default=4)
    p.add_argument("--dec_layers",   type=int,   default=2)
    p.add_argument("--mlp_ratio",    type=float, default=4.0)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--temporal_patch", type=int, default=1,
                   help="Consecutive steps merged into one encoder token")
    p.add_argument("--mask_ratio",   type=float, default=0.5,
                   help="Fraction of stations hidden from the encoder during training")
    p.add_argument("--val_mask_ratio", type=float, default=0.0,
                   help="Mask ratio used for the validation metrics")
    p.add_argument("--no_spatial_attn", action="store_true",
                   help="Remove cross-station attention from the encoder")
    p.add_argument("--station_local_decoder", action="store_true",
                   help="Decoder attends within one station only (needs --mask_ratio 0)")
    p.add_argument("--residual_head", action="store_true",
                   help="Add the last observation of visible stations to the output")
    p.add_argument("--var_weights",  type=float, nargs=5, default=None,
                   metavar=("TEMP", "PRES", "HUM", "WIND_U", "WIND_V"))
    p.add_argument("--nll_loss",     action="store_true",
                   help="Gaussian NLL objective with a log-variance head (default: Huber)")
    p.add_argument("--grad_checkpoint", action="store_true")

    # Optimisation
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--min_lr",        type=float, default=1e-6)
    p.add_argument("--weight_decay",  type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--amp",           action="store_true", help="Mixed precision")
    p.add_argument("--bf16",          action="store_true", help="bfloat16 instead of float16")
    p.add_argument("--compile",       action="store_true", help="torch.compile the model")
    p.add_argument("--seed",          type=int,   default=42)

    # Early stopping / checkpoints
    p.add_argument("--monitor",   type=str,   default="val/overall_mae")
    p.add_argument("--patience",  type=int,   default=50, help="0 disables early stopping")
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--overfit_stop", action="store_true",
                   help="Also stop after --overfit_patience validations without improvement "
                        "(not counted during warm-up)")
    p.add_argument("--overfit_patience", type=int, default=5)
    p.add_argument("--save_dir",  type=str,   default="checkpoints")
    p.add_argument("--resume",    type=str,   default=None, help="Path to last.ckpt")

    # Logging
    p.add_argument("--wandb_project",  type=str, default=None, help="Omit for CSV logging only")
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


class OverfitEarlyStop:
    """
    Stop when the monitored validation metric has not improved for ``patience``
    consecutive validations. Validations during the LR warm-up are not counted.
    Implemented as a plain object so that unused Lightning hooks are no-ops.
    """

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def __init__(self, monitor: str, patience: int = 5, min_delta: float = 0.0,
                 warmup_epochs: int = 0):
        self.monitor       = monitor
        self.patience      = patience
        self.min_delta     = min_delta
        self.warmup_epochs = int(warmup_epochs)
        self.best          = float("inf")
        self.best_epoch    = -1
        self.wait          = 0

    def on_validation_end(self, trainer, pl_module) -> None:
        m = trainer.callback_metrics.get(self.monitor)
        if m is None:
            return
        value = float(m)
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch = value, int(trainer.current_epoch)
        if trainer.current_epoch < self.warmup_epochs:
            return
        if improved:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True
                print(f"[OverfitStop] {self.monitor} has not improved for {self.patience} "
                      f"validations (best {self.best:.5f} at epoch {self.best_epoch}); stopping.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Data ────────────────────────────────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset, TRAIN_YEARS

    if args.train_years is not None:
        train_years = sorted(args.train_years)
    elif args.subset:
        train_years = [2020, 2021]
    else:
        train_years = TRAIN_YEARS

    print("Loading PeakWeather dataset ...")
    ds = load_peakweather(root=args.data_root)

    common = dict(
        window_size=args.window, max_delta_steps=args.max_delta,
        delta_grid_stride=args.delta_grid_stride,
        cache_dir=args.cache_dir or args.data_root,
        fast_cache_dir=args.local_cache_dir.strip() or None,
        train_years=train_years, exclude_stations=args.exclude_stations,
    )
    print("Building train dataset ...")
    train_ds = StationMAEDataset(ds, split="train", index_mode=args.index_mode,
                                 train_stride=args.train_stride,
                                 random_epoch_size=args.random_epoch_size, **common)
    print("Building val dataset ...")
    val_ds = StationMAEDataset(ds, split="val", obs_stats=train_ds.obs_stats,
                               index_mode="blocks", **common)
    K = len(train_ds.delta_grid)
    print(f"  train: {len(train_ds):,} samples (years {train_years[0]}-{train_years[-1]}, "
          f"index_mode={args.index_mode})  |  val: {len(val_ds):,} samples  |  "
          f"K={K} leads: {train_ds.delta_grid} steps")

    use_cuda = torch.cuda.is_available()
    persist  = args.num_workers > 0
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=use_cuda and persist, persistent_workers=persist,
                     prefetch_factor=(4 if persist else None))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ── Model ───────────────────────────────────────────────────────────
    from model.mae import StationMAE
    from model.lightning_module import StationMAELightning

    model = StationMAE(
        d_model=args.d_model, enc_heads=args.enc_heads, enc_layers=args.enc_layers,
        dec_heads=args.dec_heads, dec_layers=args.dec_layers, mlp_ratio=args.mlp_ratio,
        dropout=args.dropout, mask_ratio=args.mask_ratio, use_checkpoint=args.grad_checkpoint,
        encoder_spatial_attn=not args.no_spatial_attn,
        station_local_decoder=args.station_local_decoder,
        num_horizons=K, temporal_patch=args.temporal_patch,
        drop_path_rate=args.drop_path_rate, residual_head=args.residual_head,
        var_weights=args.var_weights, use_nll_loss=args.nll_loss, window_size=args.window,
    )
    print(f"StationMAE: {model.count_parameters():,} trainable parameters")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="default")

    # Per-station normalisation statistics of the kept stations, saved for
    # physical-unit validation metrics.
    obs_stats = train_ds.obs_stats
    keep = train_ds._keep_indices
    std  = obs_stats["std"][keep]  if keep is not None else obs_stats["std"]
    mean = obs_stats["mean"][keep] if keep is not None else obs_stats["mean"]

    cfg = {
        # optimisation
        "lr": args.lr, "min_lr": args.min_lr, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "log_every_n_steps": args.log_every_n_steps,
        # normalisation
        "obs_stats_std": std.tolist(), "obs_stats_mean": mean.tolist(),
        "num_stations": int(std.shape[0]),
        # architecture (read by StationMAE.from_cfg)
        "d_model": args.d_model, "enc_heads": args.enc_heads, "enc_layers": args.enc_layers,
        "dec_heads": args.dec_heads, "dec_layers": args.dec_layers, "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout, "drop_path_rate": args.drop_path_rate,
        "mask_ratio": args.mask_ratio, "val_mask_ratio": args.val_mask_ratio,
        "temporal_patch": args.temporal_patch,
        "encoder_spatial_attn": not args.no_spatial_attn,
        "station_local_decoder": args.station_local_decoder,
        "residual_head": args.residual_head, "var_weights": args.var_weights,
        "use_nll_loss": args.nll_loss, "grad_checkpoint": args.grad_checkpoint,
        # data
        "window": args.window, "max_delta": args.max_delta,
        "delta_grid_stride": args.delta_grid_stride,
        "index_mode": args.index_mode, "train_stride": args.train_stride,
        "exclude_stations": args.exclude_stations or [],
    }
    lit_model = StationMAELightning(model=model, cfg=cfg)

    # ── Logger and callbacks ────────────────────────────────────────────
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

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices="auto", precision=precision,
        gradient_clip_val=args.grad_clip, callbacks=callbacks, logger=logger,
        log_every_n_steps=args.log_every_n_steps, enable_progress_bar=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    print(f"\n[Station-MAE] seed={args.seed} precision={precision} "
          f"monitor={args.monitor}\nSaving checkpoints to: {args.save_dir}\n")

    trainer.fit(model=lit_model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=args.resume)

    print(f"\nTraining complete. Checkpoints saved to: {args.save_dir}")
    if args.wandb_project:
        try:
            import wandb
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
