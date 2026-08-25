"""
train_masked_transformer.py — train the encoder-only masked transformer.

Reuses StationMAEDataset so data, normalisation, splits and the 13-lead grid
are identical to main.py (v15), train_lstm.py and train_simple_mae.py.

Validation mirrors the other models exactly so the wandb panels overlay:
    val/*            — +30 min lead, ALL stations visible (mask 0), MAE only
    val/overall_mae  — the checkpoint monitor
    val/horizon_sensitivity — |pred(6 h) − pred(30 min)|; collapse detector
    sanity/*         — dispersion, persistence skill, ctx_ratio at the TRAINED
                       mask ratio (diagnostics need masked stations)

Usage
-----
    python train_masked_transformer.py --data_root /path/to/PeakWeatherDataset
See run_masked_transformer_cloud.sh for the cloud configuration.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Masked station transformer training")

    # Data — same conventions as every other trainer here
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache")
    p.add_argument("--window",      type=int, default=72)
    p.add_argument("--max_delta",   type=int, default=36)
    p.add_argument("--delta_grid_stride", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--index_mode",  type=str, default="random",
                   choices=["sliding", "blocks", "random"])
    p.add_argument("--random_epoch_size", type=int, default=10000)
    p.add_argument("--train_years", type=int, nargs="+", default=None)
    p.add_argument("--subset",      action="store_true")
    p.add_argument("--global_norm", action="store_true")

    # Model
    p.add_argument("--temporal_patch", type=int, default=6)
    p.add_argument("--d_model",     type=int,   default=384)
    p.add_argument("--n_layers",    type=int,   default=8)
    p.add_argument("--n_heads",     type=int,   default=8)
    p.add_argument("--mlp_ratio",   type=float, default=4.0)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--mask_ratio",  type=float, default=0.5)
    p.add_argument("--val_mask_ratio", type=float, default=0.0,
                   help="0.0 = validate with ALL stations visible, matching "
                        "the LSTM / simple-MAE / v15 panels.")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--pool",        type=str, default="last", choices=["last", "mean"])

    # Optimisation
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_epochs",type=int,   default=10)
    p.add_argument("--min_lr",       type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--accumulate_grad_batches", type=int, default=2)
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--patience",     type=int,   default=40)
    p.add_argument("--monitor",      type=str,   default="val/overall_mae")

    # Logging / checkpoints
    p.add_argument("--save_dir",       type=str, default="checkpoints/masked-tf-v1")
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="masked-tf-v1")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import pytorch_lightning as pl                                    # noqa: E402
from model.masked_transformer import MaskedStationTransformer     # noqa: E402
from model.embeddings import TARGET_VARIABLE_NAMES                # noqa: E402


class MaskedTFLightning(pl.LightningModule):
    def __init__(self, cfg: dict, obs_std, horizon_steps):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
        self.model = MaskedStationTransformer(
            window_steps=cfg["window"], temporal_patch=cfg["temporal_patch"],
            horizon_steps=horizon_steps,
            d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], mlp_ratio=cfg["mlp_ratio"],
            dropout=cfg["dropout"], mask_ratio=cfg["mask_ratio"],
            huber_delta=cfg["huber_delta"], pool=cfg["pool"],
        )
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32))
        self._val_mask_ratio = float(cfg.get("val_mask_ratio", 0.0))

    def _unpack(self, b):
        sp = b["spatial"]
        sp = sp[0] if sp.dim() == 3 else sp
        return b["x"], b["x_mask"], sp, b["x_hours"], b["y"], b["y_mask"]

    # ── train ────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x, xm, sp, xh, y, ym = self._unpack(batch)
        preds, smask = self.model(x, xm, sp, xh)
        loss = self.model.loss(preds, y, ym, smask)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"],
                 on_step=True, on_epoch=False)
        with torch.no_grad():
            m = ym[:, 1, :, :self.model.Vt] > 0.5
            e = (preds[:, 1] - y[:, 1, :, :self.model.Vt]).abs()
            if m.any():
                self.log("train/overall_mae", e[m].mean(),
                         on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ── validation: all stations visible, MAE only ───────────────────────
    def validation_step(self, batch, batch_idx):
        x, xm, sp, xh, y, ym = self._unpack(batch)
        B, N = x.shape[0], x.shape[2]
        Vt = self.model.Vt

        # val_mask_ratio 0 → nothing corrupted → same task as the other models
        n_mask = int(N * self._val_mask_ratio)
        vmask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        if n_mask:
            order = torch.rand(B, N, device=x.device).argsort(dim=1)
            vmask.scatter_(1, order[:, :n_mask], True)
        preds, _ = self.model(x, xm, sp, xh, station_mask=vmask)
        self.log("val/loss", self.model.loss(preds, y, ym, vmask),
                 on_epoch=True, prog_bar=True, sync_dist=True)

        k30 = 1 if preds.shape[1] > 1 else 0        # +30 min, as everywhere else
        t, m = y[:, k30, :, :Vt], ym[:, k30, :, :Vt] > 0.5
        p = preds[:, k30]
        abs_sum, n_tot = None, 0
        for v, name in enumerate(TARGET_VARIABLE_NAMES):
            mv = m[..., v]
            if not mv.any():
                continue
            e_v = (p[..., v][mv] - t[..., v][mv]).abs()
            mae = e_v.mean()
            self.log(f"val/{name}_mae", mae, on_epoch=True, sync_dist=True)
            self.log(f"val/{name}_mae_phys", mae * self.obs_std[v],
                     on_epoch=True, sync_dist=True)
            abs_sum = e_v.sum() if abs_sum is None else abs_sum + e_v.sum()
            n_tot += int(mv.sum())
        if n_tot > 0:
            self.log("val/overall_mae", abs_sum / n_tot,
                     on_epoch=True, prog_bar=True, sync_dist=True)

        # collapse detector — the metric that caught the v15 raw failure
        if preds.shape[1] > 2:
            self.log("val/horizon_sensitivity",
                     (preds[:, -1] - preds[:, 1]).abs().mean(),
                     on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self._sanity(x, xm, sp, xh, y, ym)

    @torch.no_grad()
    def _sanity(self, x, xm, sp, xh, y, ym):
        """Diagnostics at the TRAINED mask ratio (they need masked stations)."""
        Vt = self.model.Vt
        preds, smask = self.model(x, xm, sp, xh)          # trained mask_ratio
        t, m = y[..., :Vt], ym[..., :Vt] > 0.5
        e = (preds - t).abs()
        disp = [(preds[..., v][m[..., v]].std()
                 / t[..., v][m[..., v]].std().clamp(min=1e-6)).item()
                for v in range(Vt) if m[..., v].any()]
        K = preds.shape[1]
        per = t[:, :1].expand_as(t); both = m & m[:, :1]
        em = (preds[:, 1:] - t[:, 1:])[both[:, 1:]]
        ep = (per[:, 1:] - t[:, 1:])[both[:, 1:]]
        skill = (1 - em.pow(2).mean().sqrt()
                 / ep.pow(2).mean().sqrt().clamp(min=1e-6)).item() if em.numel() else float("nan")
        sel = smask.unsqueeze(-1)
        mm, vv = m[:, 0] & sel, m[:, 0] & ~sel
        ctx = float("nan")
        if mm.any() and vv.any():
            ctx = (e[:, 0][vv].mean() / e[:, 0][mm].mean().clamp(min=1e-6)).item()
        for k, v in [("sanity/dispersion_min", min(disp) if disp else float("nan")),
                     ("sanity/persist_skill", skill), ("sanity/ctx_ratio", ctx)]:
            if v == v:
                self.log(k, v, on_epoch=True, sync_dist=True)
        print(f"  [sanity e{self.current_epoch}] disp_min {min(disp) if disp else float('nan'):.2f} "
              f"| persist_skill {skill:+.3f} | ctx_ratio {ctx:.2f}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg["weight_decay"])
        warm, total = self.cfg["warmup_epochs"], self.cfg["epochs"]
        frac = self.cfg["min_lr"] / self.cfg["lr"]

        def f(ep):
            if ep < warm:
                return (ep + 1) / max(warm, 1)
            prog = (ep - warm) / max(total - warm, 1)
            return frac + (1 - frac) * 0.5 * (1 + math.cos(math.pi * prog))

        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, f),
                                 "interval": "epoch"}}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from data.dataset import load_peakweather, StationMAEDataset, TRAIN_YEARS
    cache_dir = args.cache_dir or args.data_root
    years = (sorted(args.train_years) if args.train_years
             else [2020, 2021] if args.subset else TRAIN_YEARS)

    print("Loading PeakWeather …")
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=args.window, delta_steps=args.max_delta,
                  max_delta_steps=args.max_delta, cache_dir=cache_dir,
                  train_years=years, shared_memory=False,
                  fast_cache_dir=args.local_cache_dir.strip() or None,
                  exclude_stations=args.exclude_stations,
                  delta_mode="fixed_grid", delta_grid_stride=args.delta_grid_stride,
                  global_norm=args.global_norm)
    train_ds = StationMAEDataset(ds, split="train", index_mode=args.index_mode,
                                 random_epoch_size=args.random_epoch_size, **common)
    val_ds   = StationMAEDataset(ds, split="val", obs_stats=train_ds.obs_stats,
                                 index_mode="blocks", **common)
    grid = list(train_ds.delta_grid)
    print(f"  train {len(train_ds):,} | val {len(val_ds):,} | K={len(grid)} leads {grid}")

    _persist = args.num_workers > 0
    _cuda = torch.cuda.is_available()
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh,
                                  num_workers=args.num_workers, drop_last=sh,
                                  pin_memory=_cuda and _persist,
                                  persistent_workers=_persist,
                                  prefetch_factor=(4 if _persist else None))

    std = train_ds.obs_stats["std"]
    std = std.mean(dim=0) if std.dim() == 2 else std
    cfg = vars(args).copy(); cfg["delta_grid"] = grid
    module = MaskedTFLightning(cfg, std[:5], grid)
    T = args.window // args.temporal_patch
    print(f"MaskedStationTransformer: {module.model.count_parameters():,} params "
          f"· {T} temporal positions × 155 stations (sequence does NOT shrink "
          f"with masking)")

    from pytorch_lightning.callbacks import (ModelCheckpoint, EarlyStopping,
                                             LearningRateMonitor)
    logger = None
    if args.wandb_project:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name,
                             save_dir=args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    cbs = [ModelCheckpoint(dirpath=args.save_dir, filename="best",
                           monitor=args.monitor, mode="min", save_top_k=1,
                           save_last=True, verbose=True),
           EarlyStopping(monitor=args.monitor, mode="min", patience=args.patience),
           LearningRateMonitor(logging_interval="epoch")]
    pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto",
               precision=("bf16-mixed" if args.bf16 and _cuda else "32-true"),
               gradient_clip_val=args.grad_clip,
               accumulate_grad_batches=args.accumulate_grad_batches,
               callbacks=cbs, logger=logger, log_every_n_steps=50,
               default_root_dir=args.save_dir).fit(module, dl(train_ds, True),
                                                   dl(val_ds, False))
    print("Done. Best checkpoint:", cbs[0].best_model_path)


if __name__ == "__main__":
    main()
