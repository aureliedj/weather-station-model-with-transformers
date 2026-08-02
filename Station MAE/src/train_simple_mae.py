"""
train_simple_mae.py — train the canonical MAE (model/simple_mae.py).

Reuses StationMAEDataset so data, normalisation and splits are identical to
every other model in the project. The dataset is used in pure-window mode
(max_delta = 0): the model reconstructs masked parts of the input window
itself; no separate targets, no delta grid.

Validation runs the FUTURE mask (last 6 h hidden) — the forecasting scenario —
and logs per-variable RMSE over the masked block plus a persistence skill
computed from the last visible step, so the val metric is directly meaningful
even though training mixes mask strategies.

Usage
-----
    python train_simple_mae.py --data_root /path/to/PeakWeatherDataset
See run_simple_mae_cloud.sh for the full cloud configuration.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple MAE training")

    # Data (same conventions as train_lstm.py / main.py)
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache")
    p.add_argument("--window",      type=int, default=108,
                   help="Window in 10-min steps: 108 = 12 h context + 6 h future")
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--index_mode",  type=str, default="random",
                   choices=["sliding", "blocks", "random"])
    p.add_argument("--random_epoch_size", type=int, default=10000)
    p.add_argument("--train_years", type=int, nargs="+", default=None)
    p.add_argument("--subset",      action="store_true")
    p.add_argument("--global_norm", action="store_true")

    # Model
    p.add_argument("--patch_steps", type=int,   default=6)
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--enc_layers",  type=int,   default=8)
    p.add_argument("--enc_heads",   type=int,   default=8)
    p.add_argument("--dec_dim",     type=int,   default=128)
    p.add_argument("--dec_layers",  type=int,   default=2)
    p.add_argument("--dec_heads",   type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--mask_ratio",  type=float, default=0.75)
    p.add_argument("--station_ratio", type=float, default=0.5)
    p.add_argument("--future_slots",  type=int,   default=6)
    p.add_argument("--strategy_probs", type=float, nargs=3, default=[0.5, 0.25, 0.25],
                   metavar=("P_RANDOM", "P_STATION", "P_FUTURE"))

    # Optimisation
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_epochs",type=int,   default=8)
    p.add_argument("--min_lr",       type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--patience",     type=int,   default=25)
    p.add_argument("--monitor",      type=str,   default="val/overall_rmse")

    # Logging / checkpoints
    p.add_argument("--save_dir",        type=str, default="checkpoints/simple-mae-v1")
    p.add_argument("--wandb_project",   type=str, default=None)
    p.add_argument("--wandb_run_name",  type=str, default="simple-mae-v1")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------------

import pytorch_lightning as pl                                    # noqa: E402
from model.simple_mae import SimpleMAE                            # noqa: E402
from model.embeddings import TARGET_VARIABLE_NAMES                # noqa: E402


class SimpleMAELightning(pl.LightningModule):
    def __init__(self, cfg: dict, obs_std):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
        self.model = SimpleMAE(
            window_steps=cfg["window"],
            patch_steps=cfg["patch_steps"],
            d_model=cfg["d_model"],
            enc_layers=cfg["enc_layers"],
            enc_heads=cfg["enc_heads"],
            dec_dim=cfg["dec_dim"],
            dec_layers=cfg["dec_layers"],
            dec_heads=cfg["dec_heads"],
            dropout=cfg["dropout"],
            mask_ratio=cfg["mask_ratio"],
            station_ratio=cfg["station_ratio"],
            future_slots=cfg["future_slots"],
            strategy_probs=tuple(cfg["strategy_probs"]),
        )
        # (V,) mean-over-stations std for phys-unit logging (monitoring only)
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32))

    def _unpack(self, batch):
        return (batch["x"], batch["x_mask"], batch["spatial"], batch["x_hours"])

    def training_step(self, batch, batch_idx):
        x, xm, sp, xh = self._unpack(batch)
        sp = sp[0] if sp.dim() == 3 else sp
        loss, _, _, strategy = self.model(x, xm, sp, xh)          # sampled strategy
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"train/loss_{strategy}", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Forecast-scenario validation (future block hidden), logged at TWO
        granularities so wandb overlays are honest:

        val/*            — the +30 MIN LEAD, all stations: SAME keys and SAME
                           semantics as the v14 / LSTM panels (their val is the
                           30-min forecast), so cross-run charts line up.
                           The checkpoint monitor (val/overall_rmse) therefore
                           also matches the other models' monitor semantics.
        val/block6h_*    — averaged over the WHOLE masked 6 h block: the
                           stricter MAE-native metric (was val/* in the subset
                           smoke run — renamed to stop the key collision).
        val/persist_skill — block-level skill vs last-observation persistence.
        sanity/dispersion_min — pred/target std ratio (mean-collapse detector).
        """
        x, xm, sp, xh = self._unpack(batch)
        sp = sp[0] if sp.dim() == 3 else sp
        # Forecast scenario: last future_slots hours hidden.
        loss, preds, token_mask, _ = self.model(x, xm, sp, xh, strategy="future")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        Vt = preds.shape[-1]
        W  = x.shape[1]
        f0 = W - self.model.future_slots * self.model.P           # first hidden step
        t  = x[:, f0:, :, :Vt]
        p  = preds[:, f0:]
        m  = xm[:, f0:, :, :Vt] > 0.5

        # ── +30 min lead (comparable with v14 / LSTM val panels) ─────────
        # future starts at +10 min → +30 min is the 3rd hidden step (index 2).
        s30 = 2
        t30, p30, m30 = t[:, s30], p[:, s30], m[:, s30]           # (B, N, Vt)
        sq30 = 0.0; n30 = 0
        for v, name in enumerate(TARGET_VARIABLE_NAMES):
            mv = m30[..., v]
            if not mv.any():
                continue
            e = p30[..., v][mv] - t30[..., v][mv]
            rmse, mae = e.pow(2).mean().sqrt(), e.abs().mean()
            self.log(f"val/{name}_rmse",      rmse, on_epoch=True, sync_dist=True)
            self.log(f"val/{name}_mae",       mae,  on_epoch=True, sync_dist=True)
            self.log(f"val/{name}_rmse_phys", rmse * self.obs_std[v],
                     on_epoch=True, sync_dist=True)
            self.log(f"val/{name}_mae_phys",  mae * self.obs_std[v],
                     on_epoch=True, sync_dist=True)
            sq30 = sq30 + e.pow(2).sum()
            n30  = n30 + mv.sum()
        if not isinstance(n30, int):
            self.log("val/overall_rmse", (sq30 / n30.clamp(min=1)).sqrt(),
                     on_epoch=True, prog_bar=True, sync_dist=True)

        # ── Whole 6 h block (MAE-native, stricter) ───────────────────────
        sq_sum = 0.0; n_sum = 0
        for v, name in enumerate(TARGET_VARIABLE_NAMES):
            mv = m[..., v]
            if not mv.any():
                continue
            e = p[..., v][mv] - t[..., v][mv]
            self.log(f"val/block6h_{name}_rmse_phys",
                     e.pow(2).mean().sqrt() * self.obs_std[v],
                     on_epoch=True, sync_dist=True)
            sq_sum = sq_sum + e.pow(2).sum()
            n_sum  = n_sum + mv.sum()
        if not isinstance(n_sum, int):
            self.log("val/block6h_overall_rmse",
                     (sq_sum / n_sum.clamp(min=1)).sqrt(),
                     on_epoch=True, sync_dist=True)

        # ── Persistence skill over the block + dispersion sanity ─────────
        persist = x[:, f0 - 1:f0, :, :Vt].expand_as(t)
        mboth   = m & (xm[:, f0 - 1:f0, :, :Vt] > 0.5).expand_as(m)
        if mboth.any():
            rm = (p[mboth] - t[mboth]).pow(2).mean().sqrt()
            rp = (persist[mboth] - t[mboth]).pow(2).mean().sqrt().clamp(min=1e-6)
            self.log("val/persist_skill", 1.0 - rm / rp, on_epoch=True,
                     sync_dist=True)
        disp = [
            (p[..., v][m[..., v]].std() / t[..., v][m[..., v]].std().clamp(min=1e-6))
            for v in range(Vt) if m[..., v].any()
        ]
        if disp:
            self.log("sanity/dispersion_min", torch.stack(disp).min(),
                     on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg["weight_decay"])
        warm, total = self.cfg["warmup_epochs"], self.cfg["epochs"]
        min_frac = self.cfg["min_lr"] / self.cfg["lr"]

        def lr_lambda(epoch):
            if epoch < warm:
                return (epoch + 1) / max(warm, 1)
            import math
            prog = (epoch - warm) / max(total - warm, 1)
            return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


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

    # Pure-window mode: max_delta = 0 → the sample IS the window; the y/delta
    # fields exist but are ignored (the MAE reconstructs x itself).
    common = dict(window_size=args.window, delta_steps=0, max_delta_steps=0,
                  cache_dir=cache_dir, train_years=train_years,
                  shared_memory=False, fast_cache_dir=fast_cache,
                  exclude_stations=args.exclude_stations,
                  delta_mode="fixed_grid", delta_grid_stride=1,
                  global_norm=args.global_norm)

    train_ds = StationMAEDataset(ds, split="train", index_mode=args.index_mode,
                                 random_epoch_size=args.random_epoch_size, **common)
    val_ds   = StationMAEDataset(ds, split="val", obs_stats=train_ds.obs_stats,
                                 index_mode="blocks", **common)
    print(f"  train {len(train_ds):,} windows | val {len(val_ds):,} "
          f"| window {args.window} steps = {args.window // 6} h")

    _use_cuda = torch.cuda.is_available()
    _persist  = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=_use_cuda and _persist,
                              persistent_workers=_persist,
                              prefetch_factor=(4 if _persist else None))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=_use_cuda and _persist,
                            persistent_workers=_persist,
                            prefetch_factor=(4 if _persist else None))

    # (V,) std for phys-unit monitoring: mean over stations if per-station
    _std = train_ds.obs_stats["std"]
    _std = _std.mean(dim=0) if _std.dim() == 2 else _std
    cfg = vars(args).copy()
    module = SimpleMAELightning(cfg, _std[:5])
    print(f"SimpleMAE: {module.model.count_parameters():,} params "
          f"(S={module.model.S} slots × P={module.model.P} steps)")

    from pytorch_lightning.callbacks import (ModelCheckpoint, EarlyStopping,
                                             LearningRateMonitor)
    logger = None
    if args.wandb_project:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name,
                             save_dir=args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=args.save_dir, filename="best",
                        monitor=args.monitor, mode="min", save_top_k=1,
                        save_last=True, verbose=True),
        EarlyStopping(monitor=args.monitor, mode="min", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices="auto",
        precision=("bf16-mixed" if args.bf16 and _use_cuda else "32-true"),
        gradient_clip_val=args.grad_clip, callbacks=callbacks, logger=logger,
        log_every_n_steps=50, default_root_dir=args.save_dir,
    )
    trainer.fit(module, train_loader, val_loader)
    print("Done. Best checkpoint:", callbacks[0].best_model_path)


if __name__ == "__main__":
    main()
