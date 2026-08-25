"""
main.py

Entry point for Station-MAE training with PyTorch Lightning + WandB.

Usage
-----
    # Full training run
    python main.py --data_root /path/to/peakweather

    # Quick subset check (2 years, fast)
    python main.py --data_root /path/to/peakweather --subset --epochs 20

    # Custom training years
    python main.py --data_root /path/to/peakweather --train_years 2019 2020 2021

    # Resume from Lightning checkpoint
    python main.py --data_root /path/to/peakweather --resume checkpoints/run1/last.ckpt

    # Enable WandB logging
    python main.py --data_root /path/to/peakweather --wandb_project station-mae

Arguments
---------
  Data
    --data_root      STR   Path to PeakWeather data directory (required)
    --cache_dir      STR   Directory for tensor cache (defaults to data_root)
    --window         INT   Input window in 10-min steps (default 288 = 48 h)
    --num_workers    INT   DataLoader worker processes (default 4)
    --batch_size     INT   Training batch size (default 16)

  Subset / quick-check
    --subset               Train on 2 years only (2020–2021)
    --train_years  YEAR…   Custom list of training years (e.g. 2019 2020 2021)

  Model
    --d_model        INT   Model dimension (default 128)
    --enc_heads      INT   Encoder attention heads (default 4)
    --enc_layers     INT   Encoder transformer depth (default 4)
    --dec_heads      INT   Decoder attention heads (default 4)
    --dec_layers     INT   Decoder transformer depth (default 2)
    --mlp_ratio      FLT   FFN hidden-dim ratio (default 4.0)
    --dropout        FLT   Dropout rate (default 0.1)
    --mask_ratio     FLT   Station masking fraction (default 0.5)
    --max_delta      INT   Max forecast horizon in 10-min steps (default 18 = 3 h)
    --factorised_encoder   Axial (temporal + spatial) attention in encoder
    --cross_attn_decoder   Cross-attention decoder (queries attend to encoder context)
    --grad_checkpoint      Gradient checkpointing (~66% less VRAM, ~33% slower)
    --drop_path_rate FLT   Max stochastic depth probability (0 = off; try 0.1)

  Regularisation
    --train_stride   INT   Step between training window starts (1=default, 6=hourly, 12=2-h)
    --drop_path_rate FLT   Stochastic depth max prob (0=off, try 0.05–0.20)

  Training
    --num_delta      INT   Lead-times per sample (1 = single-delta; default 1)
    --epochs         INT   Training epochs (default 100)
    --lr             FLT   Peak learning rate (default 1e-4)
    --weight_decay   FLT   AdamW weight decay (default 0.05)
    --warmup_epochs  INT   Linear LR warmup epochs (default 5)
    --grad_clip      FLT   Gradient clipping max-norm (default 1.0)
    --amp                  Mixed-precision training (fp16; CUDA/MPS only)
    --seed           INT   Random seed (default 42)

  Early stopping
    --patience       INT   Checks without val/loss improvement before stopping; 0 = off (default 10)
                           Counts "validation checks" not epochs — with --val_check_interval N,
                           patience=3 means 3×N steps without improvement before stopping.
    --min_delta      FLT   Minimum improvement to reset patience counter (default 1e-4)

  Checkpointing
    --save_dir       STR   Directory for checkpoints (default "checkpoints")
    --resume         STR   Path to Lightning .ckpt file to resume from

  Logging / WandB
    --wandb_project  STR   WandB project name; omit to disable WandB (CSV log only)
    --wandb_run_name STR   WandB run name (optional; auto-generated if omitted)
    --log_every_n_steps INT  Steps between training log lines (default 50)
"""

import argparse
import os
import random
import shutil
import threading

import numpy as np
import torch
from torch.utils.data import DataLoader

# Use file-based tensor sharing between DataLoader workers instead of the
# default shared-memory (/dev/shm) strategy.  This avoids the "bus error /
# insufficient shm" crash in containerised environments (Renku, Kubernetes,
# Docker with default shm-size=64 MB) while still allowing num_workers > 0.
torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station-MAE training")

    # Data
    p.add_argument("--data_root",    type=str, required=True)
    p.add_argument("--cache_dir",    type=str, default=None)
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache",
                   help="Fast local directory for split-specific numpy memmap files "
                        "(default /tmp/station_mae_cache — a tmpfs RAM disk on Linux). "
                        "Workers mmap .npy files directly from the OS page cache, "
                        "eliminating IPC overhead for source data. "
                        "Set to '' to disable and use the standard in-memory path.")
    p.add_argument("--window",       type=int, default=72)
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--batch_size",   type=int, default=16)

    # Subset / quick-check
    p.add_argument("--subset",       action="store_true")
    p.add_argument("--train_years",  type=int, nargs="+", default=None, metavar="YEAR")
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None, metavar="NAME",
                   help="Station name(s) to drop from ALL splits before training. "
                        "Matched case-insensitively against the stations_table index, "
                        "'name', and 'abbr' columns. "
                        "Example: --exclude_stations 110  or  --exclude_stations KLO BAS")

    # Model
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--enc_heads",    type=int,   default=4)
    p.add_argument("--enc_layers",   type=int,   default=4)
    p.add_argument("--dec_heads",    type=int,   default=4)
    p.add_argument("--dec_layers",   type=int,   default=2)
    p.add_argument("--mlp_ratio",    type=float, default=4.0)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--mask_ratio",   type=float, default=0.5)
    p.add_argument("--val_mask_ratio", type=float, default=0.0,
                   help="Station mask ratio used for the VALIDATION metrics "
                        "(default 0.0 = all stations visible). Training is "
                        "unaffected. 0.0 makes val/* the same task the LSTM "
                        "and simple-MAE runs report — a 30-min forecast from "
                        "the full network — so the wandb panels are directly "
                        "comparable. Set to --mask_ratio to validate under the "
                        "trained masking instead. The per-epoch sanity check "
                        "always runs at the TRAINED ratio (it needs masked "
                        "stations for ctx_ratio).")
    p.add_argument("--max_delta",    type=int,   default=18)
    p.add_argument("--factorised_encoder",  action="store_true",
                   help="Axial attention in encoder (~100× cheaper at W=288)")
    p.add_argument("--temporal_patch",      type=int, default=1,
                   help="Group P consecutive timesteps into ONE encoder token "
                        "(ViT/PatchTST-style patch merging). P must divide --window. "
                        "P=6 turns a 72-step window into 12 tokens, so FULL attention "
                        "over 12*N_vis costs LESS than the windowed setup it replaces "
                        "while giving every token the whole window from layer 1. "
                        "Use with --temporal_window 0. Default 1 = no patching.")
    p.add_argument("--residual_head", action="store_true",
                   help="v15: predict the DEVIATION from the last observation "
                        "(ŷ = y(t0) + f). Visible stations start at persistence; "
                        "masked stations keep a zero base (no leakage), so the "
                        "gap-filling regime is unchanged.")
    p.add_argument("--temporal_window",     type=int, default=0,
                   help="Local temporal window size in timesteps (0 = full attention). "
                        "W must be exactly divisible by this value. Odd encoder layers "
                        "use a Swin-style half-window shift so tokens communicate across "
                        "chunk boundaries after two layers.\n"
                        "  Flat encoder (default): full cross-station attention within "
                        "each tw×N_vis chunk. W=72, tw=6, N_vis=65 → 390 tokens/chunk, "
                        "144× cheaper than full flat. Supports d_model=512+.\n"
                        "  Factorised encoder: windowed temporal-only attention within "
                        "each station's tw-step chunk. Spatial sub-layer unchanged.\n"
                        "Example: --window 72 --temporal_window 6")
    p.add_argument("--cross_attn_decoder",  action="store_true",
                   help="Cross-attention decoder (query tokens attend to encoder context)")
    p.add_argument("--grad_checkpoint",     action="store_true",
                   help="Gradient checkpointing (~33%% extra compute, ~66%% less VRAM)")
    p.add_argument("--drop_path_rate",  type=float, default=0.0,
                   help="Maximum stochastic-depth (DropPath) drop probability "
                        "(linearly scheduled from 0 at layer 0 to this value at the "
                        "deepest layer, in both encoder and decoder).  Default 0.0 = "
                        "disabled.  Recommended range 0.05–0.20 for overfitting runs.")
    p.add_argument("--value_embedding",     type=str, default="linear",
                   choices=["linear", "mlp", "fourier"],
                   help="v18 observation encoder. 'linear' (default) is v17: "
                        "e_v = x_v*w_v + b_v, rank 1, no nonlinearity. "
                        "'mlp' (recommended) is 1 -> 32 -> GELU -> d: a "
                        "piecewise basis over the value axis, best on kinks "
                        "and thresholds. 'fourier' is PLR (Gorishniy et al. "
                        "2022) and wins only on angular quantities.")
    p.add_argument("--station_local_decoder", action="store_true",
                   help="Make the DECODER station-independent: each station's "
                        "Delta-queries attend only to one another and cross-attend "
                        "only to that station's own encoder tokens. Combined with "
                        "--no_spatial_attn this gives a fully station-blind model "
                        "that still uses the Delta-query decoder. Requires "
                        "--mask_ratio 0, since a masked station has no encoder "
                        "tokens to attend to.")
    p.add_argument("--no_spatial_attn",     action="store_true",
                   help="Remove the spatial sub-layer from every factorised encoder "
                        "block: each station is encoded from its own temporal window "
                        "with no cross-station mixing. This is the CONTROLLED STUDY "
                        "against the LSTM — if the error curves coincide, neighbouring "
                        "stations are contributing nothing and the spatial machinery is "
                        "not earning its cost.")
    p.add_argument("--readout",             type=str, default="last",
                   choices=["last", "mean"],
                   help="Readout mode. 'last' takes the most recent "
                        "temporal slot (the attention analogue of an LSTM's "
                        "final hidden state, and the shortest path from a "
                        "station's last observation to its own prediction); "
                        "'mean' averages the slots, so every temporal token "
                        "gets direct gradient rather than only the last.")
    p.add_argument("--static_in_token",     action="store_true",
                   help="v21 (Aurora-style): put the 15 static station features "
                        "INSIDE VariableProjection as extra slots, and drop the "
                        "separate pos_emb / station_emb branches. Terrain is then "
                        "scaled by the same mechanism as the weather instead of "
                        "by an independent MLP. Grouped as position (cols 0:2) + "
                        "topography (cols 2:15); one slot per feature would give "
                        "the weather only 6/21 of the block and measures WORSE "
                        "than the separate-branch default.")

    p.add_argument("--var_weights", type=float, nargs=5, default=None,
                   metavar=("TEMP","PRES","HUM","WIND_U","WIND_V"),
                   help="Per-variable loss weights. Default None keeps the values "
                        "hardcoded in mae.py: [1.0, 1.0, 0.7, 0.5, 0.5]. "
                        "NOTE the LSTM baseline trains with 1.0 across the board "
                        "(run_lstm_cloud.sh), so the default makes the per-variable "
                        "comparison on humidity and wind unfair to the transformer — "
                        "it gets 0.7x and 0.5x the gradient on exactly the variables "
                        "where it loses. Pass 1.0 1.0 1.0 1.0 1.0 for a fair match.")
    p.add_argument("--nll_loss", action="store_true",
                   help="Replace MSE/Huber with heteroscedastic Gaussian NLL (CRPS). "
                        "Adds a log-variance head to the decoder: the model predicts both "
                        "a mean and a per-variable uncertainty σ² at every station. "
                        "Loss = 0.5 × (err² / σ² + log σ²). "
                        "Initialised so σ²=1 at training start (= same scale as MSE). "
                        "The model learns to widen uncertainty for hard samples (long "
                        "horizons, isolated masked stations). "
                        "Val RMSE is still computed from the mean prediction — comparable "
                        "across runs regardless of loss mode. "
                        "(Priority 2 — REVIEW.md §5.2; Gneiting & Raftery 2007; "
                        "Andrychowicz et al. 2023 MetNet-3)")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Gradient accumulation steps (default 1 = no accumulation). "
                        "Effective batch_size = batch_size × accumulate_grad_batches. "
                        "Use 4 with batch_size=4 for effective batch=16 without extra VRAM.")

    # Data augmentation / regularisation
    p.add_argument("--index_mode", type=str, default="sliding",
                   choices=["sliding", "blocks", "random"],
                   help="Training window selection strategy (val/test always use sliding/1).\n"
                        "  sliding (default, Strategy C — GraphDOP / most baselines):\n"
                        "    Every contiguity-valid start, thinned by --train_stride.\n"
                        "    DataLoader shuffle gives random-without-replacement epochs.\n"
                        "  blocks  (Strategy B — PatchTST / iTransformer):\n"
                        "    Greedy non-overlapping windows; no two windows share any\n"
                        "    input timestep.  Smallest dataset, cleanest gradients.\n"
                        "  random  (Strategy A — Aurora / W-MAE / VideoMAE):\n"
                        "    Full pool stored; __getitem__ samples uniformly at random\n"
                        "    regardless of DataLoader idx — true per-item replacement\n"
                        "    sampling, different windows every epoch.")
    p.add_argument("--train_stride", type=int, default=1,
                   help="Thinning step for --index_mode sliding on the train split. "
                        "stride=1 (default): every contiguity-valid start. "
                        "stride=6 → 60-min spacing (W=72: 92%% overlap, ~6× fewer samples). "
                        "stride=12 → 2-h spacing. Ignored for 'blocks' and 'random' modes. "
                        "Val/test always use stride=1.")

    # Delta / lead-time configuration
    p.add_argument("--delta_mode", type=str, default="fixed_grid",
                   choices=["fixed_grid", "random"],
                   help="Lead-time selection mode.\n"
                        "  fixed_grid (default): every sample returns K targets at\n"
                        "    0, delta_grid_stride, …, max_delta steps (every 30 min\n"
                        "    up to 6 h with max_delta=36, stride=3 → K=13).\n"
                        "    Encoder runs once; decoder runs K times per sample.\n"
                        "  random: num_delta distinct lead-times drawn uniformly\n"
                        "    from [1, max_delta] per sample (legacy behaviour).")
    p.add_argument("--delta_grid_stride", type=int, default=3,
                   help="Spacing between fixed-grid horizons in 10-min steps "
                        "(default 3 = 30 min).  Only used with --delta_mode fixed_grid.")
    p.add_argument("--random_epoch_size", type=int, default=None,
                   help="Epoch length when --index_mode random is used (default: "
                        "len(pool) // window_size ≈ number of non-overlapping blocks). "
                        "Increase to see more samples per epoch; decrease to cut epoch time.")
    # Training
    p.add_argument("--num_delta",    type=int,   default=6)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--min_lr",        type=float, default=1e-6,
                   help="Minimum LR floor for cosine decay schedule (default 1e-6). "
                        "Prevents the LR from decaying all the way to zero near end of "
                        "training. Expressed as an absolute LR value, not a ratio. "
                        "Set to 0.0 to disable (original behaviour: decay to zero).")
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--bf16",         action="store_true",
                   help="Use bfloat16 mixed precision instead of float16. "
                        "Requires --amp. Preferred on A100/H100/3090+ GPUs: same "
                        "tensor-core speed as fp16 but wider dynamic range, no "
                        "loss scaling needed. Falls back to fp16 on MPS.")
    p.add_argument("--compile",      action="store_true",
                   help="Wrap model with torch.compile (reduce-overhead mode). "
                        "Fuses kernels and eliminates Python overhead — typically "
                        "2-3× faster on A100/H100 after a one-time warm-up of "
                        "~2-3 min on the first epoch. Requires PyTorch >= 2.0.")
    p.add_argument("--seed",         type=int,   default=42)

    # Early stopping
    p.add_argument("--patience",     type=int,   default=50)
    p.add_argument("--min_delta",    type=float, default=1e-4)
    p.add_argument("--monitor",      type=str,   default="val/temperature_mae",
                   help="Metric monitored by ModelCheckpoint and EarlyStopping. "
                        "Must be logged by lightning_module.py. "
                        "Common choices: val/loss, val/temperature_mae, val/overall_mae. "
                        "(default: val/temperature_mae)")

    # Overfitting-aware early stop
    p.add_argument("--overfit_stop", action="store_true",
                   help="Enable overfit-aware early stopping. Stops training when the "
                        "monitored VALIDATION metric stops improving for --overfit_patience "
                        "consecutive validation checks — the signature of overfitting — and "
                        "also logs the val−train generalisation gap after every check. "
                        "This is a tighter, overfit-focused complement to --patience (which "
                        "is often left large to allow long plateaus). best.ckpt always holds "
                        "the pre-overfit (best-validation) model.")
    p.add_argument("--overfit_patience", type=int, default=5,
                   help="Consecutive validation checks without improvement before the "
                        "overfit stopper fires (default 5).")
    p.add_argument("--overfit_min_delta", type=float, default=0.0,
                   help="Minimum improvement in the monitored metric to reset the overfit "
                        "counter (default 0.0 = any non-improvement counts).")

    # Checkpointing
    p.add_argument("--save_dir",        type=str, default="checkpoints")
    p.add_argument("--resume",          type=str, default=None)

    # Sub-epoch validation / checkpointing
    p.add_argument("--val_check_interval", type=int, default=0,
                   help="Run validation every N training steps instead of every epoch "
                        "(0 = once per epoch, the default). EarlyStopping patience "
                        "then counts in validation-checks, not epochs. Example for a "
                        "50-min epoch on A100 (~6500 steps): use 4000 for ~30-min "
                        "checks. Checkpointing is unaffected: best.ckpt and last.ckpt are "
                        "written by ModelCheckpoint at every validation point.")

    # Logging / WandB
    p.add_argument("--wandb_project",   type=str, default=None,
                   help="WandB project name; omit to use CSV logger only")
    p.add_argument("--wandb_run_name",  type=str, default=None)
    p.add_argument("--wandb_offline",   action="store_true",
                   help="Run WandB in offline mode (sets WANDB_MODE=offline). "
                        "Logs are saved locally and can be synced later with "
                        "'wandb sync <run-dir>'.  Useful on Renku / HPC where "
                        "outbound connections to wandb.com may be blocked.")
    p.add_argument("--log_every_n_steps", type=int, default=50)

    # Profiling
    p.add_argument("--profiler", type=str, default="none",
                   choices=["none", "simple", "pytorch"],
                   help="Lightning profiler: 'none' (default) = off; "
                        "'simple' = wall-clock summary per op; "
                        "'pytorch' = full CUDA kernel trace (export_to_chrome=True). "
                        "Use with --limit_train_batches for a mini profiling run.")
    p.add_argument("--limit_train_batches", type=int, default=0,
                   help="Cap training to N batches per epoch (0 = unlimited, default). "
                        "Useful with --profiler to run a short representative trace "
                        "without waiting for a full epoch.")
    p.add_argument("--limit_val_batches", type=int, default=0,
                   help="Cap validation to N batches per check (0 = full val set). "
                        "Useful when the val set is much larger than the train epoch "
                        "(e.g. random mode with small epoch size). "
                        "Example: --limit_val_batches 200 evaluates 6,400 samples.")

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

class OverfitEarlyStop(object):
    """
    Overfit-aware early stopping callback.

    After every validation run it compares the monitored validation metric to its
    best value so far.  When the metric fails to improve (by ``min_delta``) for
    ``patience`` consecutive checks — the tell-tale sign that the model has stopped
    generalising and is starting to overfit — it sets ``trainer.should_stop`` and
    training ends cleanly.  ``best.ckpt`` (saved by ModelCheckpoint on the same
    metric) therefore always holds the pre-overfit model.

    It additionally logs the val−train **generalisation gap** after every check, so
    a widening gap (val worsening while train keeps falling) is visible in the logs
    and in WandB as ``diag/gen_gap``.

    Implemented as a plain object with a no-op ``__getattr__`` so every Lightning
    hook we don't define is silently ignored; this avoids importing
    ``pytorch_lightning`` at module import time.
    """

    def __getattr__(self, name):
        """Return a no-op for any Lightning hook not explicitly defined."""
        return lambda *args, **kwargs: None

    def __init__(self, monitor: str = "val/loss", patience: int = 5,
                 min_delta: float = 0.0, mode: str = "min",
                 train_metric: str = "train/loss", verbose: bool = True,
                 warmup_epochs: int = 0):
        self.monitor      = monitor
        self.patience     = patience
        self.min_delta    = min_delta
        self.mode         = mode
        self.train_metric = train_metric
        self.verbose      = verbose
        self.best         = float("inf") if mode == "min" else float("-inf")
        self.wait         = 0
        self.best_epoch   = -1
        # Do not count non-improving epochs while the LR is still warming up.
        # Without this, patience=5 against warmup_epochs=15 stops training
        # before the learning rate has even reached its peak — which is what
        # invalidated v12.
        self.warmup_epochs = int(warmup_epochs)

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_validation_end(self, trainer, pl_module) -> None:   # noqa: N802
        metrics = trainer.callback_metrics
        m = metrics.get(self.monitor)
        if m is None:
            return                          # metric not logged yet (e.g. sanity check)
        value = float(m)

        # ── Warm-up guard ────────────────────────────────────────────────
        # During warmup the LR is still ramping, so flat or worsening val is
        # expected and must not consume patience.
        if trainer.current_epoch < self.warmup_epochs:
            if self.verbose:
                print(f"[OverfitStop] warmup {trainer.current_epoch + 1}"
                      f"/{self.warmup_epochs} — not counting "
                      f"({self.monitor}={value:.5f})")
            if self._is_improvement(value):
                self.best, self.best_epoch = value, int(trainer.current_epoch)
            return

        # Generalisation gap (val − train), logged for visibility.
        tr  = metrics.get(self.train_metric)
        gap = (value - float(tr)) if tr is not None else float("nan")
        try:
            pl_module.log("diag/gen_gap", gap, on_epoch=True, prog_bar=False)
        except Exception:
            pass

        if self._is_improvement(value):
            self.best       = value
            self.wait       = 0
            self.best_epoch = int(trainer.current_epoch)
            if self.verbose:
                print(f"[OverfitStop] new best {self.monitor}={value:.5f} "
                      f"(epoch {self.best_epoch})  gap(val-train)={gap:+.4f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"[OverfitStop] no improvement {self.wait}/{self.patience}  "
                      f"({self.monitor}={value:.5f}, best={self.best:.5f} "
                      f"@ep{self.best_epoch}, gap(val-train)={gap:+.4f})")
            if self.wait >= self.patience:
                trainer.should_stop = True
                print(f"[OverfitStop] ⛔ STOP — {self.monitor} has not improved for "
                      f"{self.patience} checks (best {self.best:.5f} at epoch "
                      f"{self.best_epoch}). best.ckpt holds the pre-overfit model.")






def main() -> None:
    args = parse_args()

    # ── Seed ────────────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── CUDA optimisations (Lightning handles device placement, but we set
    #    backend flags before the Trainer initialises) ──────────────────────
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cuda.enable_flash_sdp(True)

    # ── Data ────────────────────────────────────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset, TRAIN_YEARS

    cache_dir   = args.cache_dir or args.data_root

    if args.train_years is not None:
        train_years = sorted(args.train_years)
    elif args.subset:
        train_years = [2020, 2021]
    else:
        train_years = TRAIN_YEARS

    if train_years != TRAIN_YEARS:
        print(f"[Subset mode]  Training years: {train_years}  "
              f"(full set would be {TRAIN_YEARS})")

    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    # Resolve fast cache dir — empty string disables it
    fast_cache_dir = args.local_cache_dir.strip() or None
    if fast_cache_dir:
        print(f"Fast cache dir           : {fast_cache_dir}  "
              f"(numpy memmap, workers bypass IPC queue for source data)")
    else:
        print("Fast cache dir           : disabled  (using standard in-memory path)")

    if args.exclude_stations:
        print(f"Excluding stations       : {args.exclude_stations}  "
              f"(matched against stations_table index / name / abbr)")

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
        shared_memory=False,
        fast_cache_dir=fast_cache_dir,
        exclude_stations=args.exclude_stations,
        train_stride=args.train_stride,
        index_mode=args.index_mode,
        delta_mode=args.delta_mode,
        delta_grid_stride=args.delta_grid_stride,
        random_epoch_size=args.random_epoch_size,
    )

    print("Building val dataset …")
    val_ds = StationMAEDataset(
        ds,
        window_size=args.window,
        delta_steps=args.max_delta,
        split="val",
        obs_stats=train_ds.obs_stats,       # always normalise with train-split stats
        max_delta_steps=args.max_delta,
        cache_dir=cache_dir,
        train_years=train_years,
        shared_memory=False,
        fast_cache_dir=fast_cache_dir,
        exclude_stations=args.exclude_stations,
        delta_mode=args.delta_mode,
        delta_grid_stride=args.delta_grid_stride,
        index_mode="blocks",                # non-overlapping windows for val
    )

    # K = number of lead-times per sample
    _K = len(train_ds.delta_grid) if args.delta_mode == "fixed_grid" else args.num_delta
    print(f"  train: {len(train_ds):,} samples  "
          f"(years {train_years[0]}–{train_years[-1]})  |  "
          f"val: {len(val_ds):,} samples")
    if args.delta_mode == "fixed_grid":
        _grid_h = [f"{dt*10//60}h{dt*10%60:02d}" if dt > 0 else "0" for dt in train_ds.delta_grid]
        print(f"  delta_mode = fixed_grid  |  K={_K}  |  "
              f"horizons: {', '.join(_grid_h)}")
    else:
        print(f"  num_delta (K) = {args.num_delta}  |  max_delta = {args.max_delta} steps "
              f"({args.max_delta * 10 // 60} h {args.max_delta * 10 % 60} min lead-time)")

    _use_cuda     = torch.cuda.is_available()
    _persistent   = (args.num_workers > 0)
    # pin_memory only helps when DataLoader workers are running (async transfer).
    # With num_workers=0 it spawns pin_memory threads that consume /dev/shm,
    # which is capped in containerised environments (Renku, Kubernetes, Docker).
    _pin_memory   = _use_cuda and _persistent

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=_pin_memory,
        drop_last=True,
        persistent_workers=_persistent,
        prefetch_factor=(4 if _persistent else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_persistent,
        prefetch_factor=(4 if _persistent else None),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    from model.mae import StationMAE
    from model.lightning_module import StationMAELightning

    model = StationMAE(
        d_model=args.d_model,
        enc_heads=args.enc_heads,
        enc_layers=args.enc_layers,
        dec_heads=args.dec_heads,
        dec_layers=args.dec_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        use_checkpoint=args.grad_checkpoint,
        factorised_encoder=args.factorised_encoder,
        temporal_window=args.temporal_window,
        value_embedding=args.value_embedding,
        encoder_spatial_attn=not args.no_spatial_attn,
        station_local_decoder=args.station_local_decoder,
        static_in_token=args.static_in_token,
        readout=args.readout,
        num_horizons=(args.max_delta // args.delta_grid_stride + 1),
        temporal_patch=args.temporal_patch,
        cross_attention_decoder=args.cross_attn_decoder,
        drop_path_rate=args.drop_path_rate,
        residual_head=args.residual_head,
        var_weights=args.var_weights,
        use_nll_loss=args.nll_loss,
        window_size=args.window,
    )

    _mode_labels = {
        "sliding": "C — full sliding window, thinned by train_stride (GraphDOP / default)",
        "blocks":  "B — greedy non-overlapping windows (PatchTST / iTransformer)",
        "random":  "A — per-item random sampling with replacement (Aurora / W-MAE)",
    }
    print(f"Window strategy          : {args.index_mode}  "
          f"({_mode_labels[args.index_mode]})")
    print(f"  → {len(train_ds.indices):,} training windows  "
          f"(pool={len(train_ds.indices):,} for random; effective for others)")
    if args.train_stride > 1:
        _n_before = len(train_ds.indices) * args.train_stride  # rough estimate
        print(f"Train stride             : {args.train_stride}  "
              f"({len(train_ds):,} samples  → ~{args.train_stride}× less window overlap)")
    if args.drop_path_rate > 0:
        print(f"Stochastic depth         : drop_path_rate={args.drop_path_rate}  "
              f"(linearly 0 → {args.drop_path_rate} across encoder + decoder layers)")
    if args.nll_loss:
        print("Loss mode                : Gaussian NLL / CRPS  "
              "(heteroscedastic — decoder predicts mean + log σ² per variable; "
              "NLL = 0.5×(err²/σ² + log σ²))")
    if args.grad_checkpoint:
        print("Gradient checkpointing   : ON  (~33% extra compute, ~66% less VRAM)")
    if args.factorised_encoder:
        _tw_str = (f"  |  temporal_window={args.temporal_window}"
                   if args.temporal_window > 0 else "  |  full temporal attention")
        print(f"Factorised encoder       : ON  (temporal + spatial{_tw_str})")
    if args.cross_attn_decoder:
        print("Cross-attention decoder  : ON")
    if args.value_embedding != "linear":
        print(f"Observation encoder      : {args.value_embedding}  (v18)")
    if args.static_in_token:
        print("Static features          : INSIDE the variable block (v21) — "
              "pos_emb / station_emb removed from the token")
    print(f"Model: {model.count_parameters():,} trainable parameters")

    # ── Token balance (initialisation invariant, not a training metric) ─────
    # The encoder token is the sum of five branches and only one of them sees
    # an observation. Nothing downstream — not the loss, not val/*, not
    # sanity/* — reports it if that branch arrives too small to matter, so it
    # is checked here, once, before any gradient is taken. Costs ~1 s.
    try:
        from model.token_balance import token_balance, synthetic_batch, format_report
        _tb = token_balance(model.encoder, **synthetic_batch(seed=0))
        print(format_report(_tb))
        if not _tb["passes"]:
            print("  ^ the encoder is about to train on tokens that barely encode "
                  "the weather; see tests/test_token_balance.py")
    except Exception as _e:                                       # noqa: BLE001
        print(f"[token-balance] skipped ({type(_e).__name__}: {_e})")


    # ── torch.compile ────────────────────────────────────────────────────────
    # Compile BEFORE wrapping in Lightning so that the compiled forward()
    # is what Lightning calls.  reduce-overhead mode eliminates most Python
    # dispatch cost; safe for all standard PyTorch ops used here.
    # The first 2-3 batches are slow (kernel tracing/compilation), then fast.
    if args.compile:
        if not hasattr(torch, "compile"):
            print("torch.compile            : SKIPPED  (requires PyTorch >= 2.0)")
        else:
            # Use "default" mode rather than "reduce-overhead":
            # reduce-overhead relies on CUDA graph capture which is restricted
            # inside MIG partitions and will silently fall back or error.
            # "default" still fuses ops and removes Python overhead (~1.3-1.5×
            # speedup) without needing CUDA graphs.
            model = torch.compile(model, mode="default")
            print("torch.compile            : ON  (default mode, MIG-safe) "
                  "— first epoch warm-up ~1-2 min, then ~1.3-1.5× faster")

    # Pass per-station normalisation statistics into cfg so that
    # lightning_module.py can log physically-unnormalized val metrics.
    # obs_stats["mean"] / ["std"] shape: (N_full, V) per-station-per-variable.
    # After station exclusion, obs_stats stays at N_full for cache compatibility;
    # we slice to the kept stations here using _keep_indices set by exclusion.
    _obs_stats = train_ds.obs_stats
    _keep = getattr(train_ds, "_keep_indices", None)
    if _keep is not None and _obs_stats["std"].dim() == 2:
        _obs_stats_kept = {
            "mean": _obs_stats["mean"][_keep],
            "std":  _obs_stats["std"][_keep],
        }
    else:
        _obs_stats_kept = _obs_stats
    cfg = {
        "lr":                  args.lr,
        "min_lr":              args.min_lr,
        "weight_decay":        args.weight_decay,
        "epochs":              args.epochs,
        "warmup_epochs":       args.warmup_epochs,
        "log_every_n_steps":   args.log_every_n_steps,
        # Normalisation stats for physically-unnormalized WandB metrics.
        # Now (N_keep, V) per-station-per-variable; lightning_module averages
        # over stations for a single monitoring std per variable.
        "obs_stats_std":       _obs_stats_kept["std"].tolist(),   # (N_keep, V) nested list
        "obs_stats_mean":      _obs_stats_kept["mean"].tolist(),  # (N_keep, V) nested list
        "num_stations":        _obs_stats_kept["std"].shape[0],   # N_keep
        # Informational — stored in checkpoint hyper_parameters for test.py
        "factorised_encoder":  args.factorised_encoder,
        "temporal_window":     args.temporal_window,
        "value_embedding":     args.value_embedding,
        "encoder_spatial_attn": not args.no_spatial_attn,
        "static_in_token":     args.static_in_token,
        "readout":             args.readout,
        "temporal_patch":      args.temporal_patch,
        # Structural (v15): MUST be recorded or evaluation rebuilds the
        # wrong tokenization / head parameterisation.
        "residual_head":       args.residual_head,
        "var_weights":         args.var_weights,
        "cross_attn_decoder":  args.cross_attn_decoder,
        "grad_checkpoint":     args.grad_checkpoint,
        "window":              args.window,
        "max_delta":           args.max_delta,
        "num_delta":           args.num_delta,
        "d_model":             args.d_model,
        "enc_heads":           args.enc_heads,
        "enc_layers":          args.enc_layers,
        "dec_heads":           args.dec_heads,
        "dec_layers":          args.dec_layers,
        "mlp_ratio":           args.mlp_ratio,
        "mask_ratio":          args.mask_ratio,
        "val_mask_ratio":     args.val_mask_ratio,
        "dropout":             args.dropout,
        "drop_path_rate":      args.drop_path_rate,
        "use_nll_loss":        args.nll_loss,
        "station_local_decoder": args.station_local_decoder,
        "index_mode":          args.index_mode,
        "train_stride":        args.train_stride,
        "delta_mode":          args.delta_mode,
        "delta_grid_stride":   args.delta_grid_stride,
        "exclude_stations":    args.exclude_stations or [],
    }

    lit_model = StationMAELightning(model=model, cfg=cfg)

    # ── Logger ──────────────────────────────────────────────────────────────
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    if args.wandb_project:
        # Set offline mode BEFORE importing wandb so the env-var is picked up.
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        # Suppress WandB's own network-error tracebacks — we handle them below.
        os.environ.setdefault("WANDB_SILENT", "true")
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),          # logs all CLI flags to WandB
                save_dir=args.save_dir,
            )
            _mode = "offline" if args.wandb_offline else "online"
            print(f"WandB logger             : project='{args.wandb_project}'  [{_mode}]")
        except Exception as _wandb_err:
            print(f"[WARNING] WandB init failed: {_wandb_err}")
            print("          Falling back to CSV logger — training will continue.")
            print("          Tip: re-run with --wandb_offline to log without internet,")
            print("               or run 'wandb login' in the terminal first.")
            logger = CSVLogger(save_dir=args.save_dir, name="logs")
    else:
        logger = CSVLogger(save_dir=args.save_dir, name="logs")
        print("Logger                   : CSV only  (use --wandb_project to enable WandB)")

    # ── Callbacks ───────────────────────────────────────────────────────────
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Sub-epoch validation / checkpointing ──────────────────────────────
    # When --val_check_interval N is set, Lightning runs validation every N
    # training steps instead of once per epoch.  EarlyStopping patience then
    # counts "validation checks" (i.e. every N steps), not full epochs.
    # Example: 50-min epoch ≈ 6500 steps, val_check_interval=4000 → val every ~30 min.
    _step_based = args.val_check_interval > 0
    if _step_based:
        print(f"Val check interval       : every {args.val_check_interval} steps "
              f"(~{args.val_check_interval * args.batch_size / 1000:.0f}k samples between checks)")

    callbacks = [
        # Saves two files only:
        #   best.ckpt  — lowest monitored metric seen so far (overwritten in-place)
        #   last.ckpt  — most recent epoch (overwritten in-place, used for resuming)
        # save_top_k=1 + save_last=True is all that's needed — no periodic snapshots.
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="best",
            monitor=args.monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        # Learning rate monitor — logs train/lr to WandB automatically
        LearningRateMonitor(logging_interval="step"),
    ]


    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=args.monitor,
                patience=args.patience,
                min_delta=args.min_delta,
                mode="min",
                verbose=True,
            )
        )
        _patience_unit = "steps" if _step_based else "epochs"
        _patience_val  = (args.patience * args.val_check_interval
                          if _step_based else args.patience)
        print(f"EarlyStopping            : monitor={args.monitor}  "
              f"patience={args.patience} checks  "
              f"(= {_patience_val} {_patience_unit})  min_delta={args.min_delta}")

    # ── Overfit-aware early stopping ──────────────────────────────────────
    # Stops as soon as the validation metric stops improving for a short window
    # (overfit_patience), regardless of the more permissive --patience above.
    # Logs the val−train gap after every check.  best.ckpt = pre-overfit model.
    if args.overfit_stop:
        callbacks.append(
            OverfitEarlyStop(
                monitor      = args.monitor,
                patience     = args.overfit_patience,
                min_delta    = args.overfit_min_delta,
                mode         = "min",
                train_metric = "train/loss",
                verbose      = True,
                # Never consume patience while the LR is still ramping.
                warmup_epochs = args.warmup_epochs,
            )
        )
        _ofp_unit = "steps" if _step_based else "epochs"
        _ofp_val  = (args.overfit_patience * args.val_check_interval
                     if _step_based else args.overfit_patience)
        print(f"OverfitStop              : ON  monitor={args.monitor}  "
              f"patience={args.overfit_patience} checks "
              f"(= {_ofp_val} {_ofp_unit})  min_delta={args.overfit_min_delta}  "
              f"(logs diag/gen_gap)")

    # ── Precision (AMP) ─────────────────────────────────────────────────────
    # "bf16-mixed" — preferred on A100/H100/RTX 3090+: same tensor-core speed
    #                as fp16 but wider dynamic range, no loss-scaling needed.
    # "16-mixed"   — fp16 AMP for older CUDA GPUs and MPS (Apple Silicon).
    # "32-true"    — full float32 (CPU or no --amp flag).
    _mps_available = (torch.backends.mps.is_available()
                      and torch.backends.mps.is_built())
    if args.amp and torch.cuda.is_available():
        precision = "bf16-mixed" if args.bf16 else "16-mixed"
    elif args.amp and _mps_available:
        precision = "16-mixed"   # MPS does not support bf16 AMP
    else:
        precision = "32-true"

    # ── LR floor reporting ───────────────────────────────────────────────────
    if args.min_lr > 0:
        print(f"LR schedule              : warmup {args.warmup_epochs} ep → "
              f"cosine → floor {args.min_lr:.2e}  "
              f"(ratio {args.min_lr / args.lr:.4f})")

    # ── Profiler ─────────────────────────────────────────────────────────────
    # "none"    → no profiling overhead
    # "simple"  → wall-clock table per Lightning hook (useful baseline)
    # "pytorch" → full PyTorch / CUDA kernel trace via torch.profiler.
    #             Exports a Chrome trace JSON to profile/ subdirectory.
    #             Open at https://ui.perfetto.dev — then "Open trace file".
    #             Use --limit_train_batches 400 to keep trace size manageable.
    profiler = None
    if args.profiler == "simple":
        from pytorch_lightning.profilers import SimpleProfiler
        profiler = SimpleProfiler(dirpath=args.save_dir, filename="simple_profile")
        print("Profiler                 : SimpleProfiler  "
              f"(output → {args.save_dir}/simple_profile.txt)")
    elif args.profiler == "pytorch":
        from pytorch_lightning.profilers import PyTorchProfiler
        _profile_dir = os.path.join(args.save_dir, "profile")
        os.makedirs(_profile_dir, exist_ok=True)
        profiler = PyTorchProfiler(
            dirpath=_profile_dir,
            filename="pytorch_trace",
            export_to_chrome=True,
            # Profile after a short warm-up so compile/JIT doesn't dominate
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,           # stack traces add overhead; enable if needed
        )
        print(f"Profiler                 : PyTorchProfiler  "
              f"(Chrome trace → {_profile_dir}/pytorch_trace*.json)\n"
              f"                           Open at https://ui.perfetto.dev")

    # ── Trainer ─────────────────────────────────────────────────────────────
    _trainer_kwargs = {}
    if args.val_check_interval > 0:
        _trainer_kwargs["val_check_interval"] = args.val_check_interval
    if args.limit_train_batches > 0:
        _trainer_kwargs["limit_train_batches"] = args.limit_train_batches
        print(f"limit_train_batches      : {args.limit_train_batches}  "
              f"(mini run — full epoch would be ~{len(train_loader)} batches)")
    if args.limit_val_batches > 0:
        _trainer_kwargs["limit_val_batches"] = args.limit_val_batches
        print(f"limit_val_batches        : {args.limit_val_batches}  "
              f"({args.limit_val_batches * args.batch_size:,} val samples per check)")
    if args.accumulate_grad_batches > 1:
        _trainer_kwargs["accumulate_grad_batches"] = args.accumulate_grad_batches
        print(f"Gradient accumulation    : {args.accumulate_grad_batches}×  "
              f"(effective batch={args.batch_size * args.accumulate_grad_batches})")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision=precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
        deterministic=False,
        profiler=profiler,
        **_trainer_kwargs,
    )

    print(f"\n[Station-MAE]  seed={args.seed}  precision={precision}")
    print(f"Saving checkpoints to: {args.save_dir}\n")

    # ── CPU keepalive (cloud/Renku idle-timeout guard) ───────────────────────
    # On Renku / Kubernetes, sessions are paused when the CPU appears idle for
    # too long.  With torch.compile + fast mmap cache, the CPU is nearly idle
    # during GPU-bound training: DataLoader workers fill the prefetch queue in
    # milliseconds, then block; the main thread dispatches CUDA kernels with
    # negligible Python overhead.  The platform idle detector can mis-classify
    # this as an inactive session and pause the job mid-epoch.
    #
    # The keepalive thread wakes every 20 seconds and performs a tiny CPU
    # computation (a dot-product on a small float32 vector) — enough to register
    # meaningful CPU utilisation without meaningfully affecting training speed.
    # The thread is daemonised so it exits automatically when the main process
    # finishes or is killed.
    _keepalive_stop = threading.Event()

    def _keepalive_fn(stop_event: threading.Event, interval: int = 20) -> None:
        _dummy = [float(i) for i in range(512)]
        while not stop_event.wait(timeout=interval):
            # Tiny CPU work: sum a small list — invisible overhead, visible activity
            _ = sum(_dummy)

    _keepalive_thread = threading.Thread(
        target=_keepalive_fn,
        args=(_keepalive_stop,),
        name="cpu-keepalive",
        daemon=True,
    )
    _keepalive_thread.start()
    print("CPU keepalive            : ON  (wakes every 20 s — guards against idle timeout)")

    # ── Fit ─────────────────────────────────────────────────────────────────
    # Pass ckpt_path to resume a Lightning checkpoint (full training state).
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,          # None = start fresh
    )

    _keepalive_stop.set()   # signal the keepalive thread to exit cleanly

    print(f"\nTraining complete.  Checkpoints saved to: {args.save_dir}")
    if args.wandb_project:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass   # WandB may not have initialised if network failed at startup


if __name__ == "__main__":
    main()
