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
    --save_every     INT   Save a numbered snapshot every N epochs (default 5)
                           Ignored when --save_every_steps > 0.
    --save_every_steps INT Save a numbered snapshot every N training steps (default 0 = off)
                           When set, replaces --save_every.  Useful for long epochs (>30 min):
                           set equal to --val_check_interval so checkpoints align with
                           validation runs.  Example: --val_check_interval 4000
                           --save_every_steps 4000  → check + save every ~30 min on A100.
    --resume         STR   Path to Lightning .ckpt file to resume from

  Logging / WandB
    --wandb_project  STR   WandB project name; omit to disable WandB (CSV log only)
    --wandb_run_name STR   WandB run name (optional; auto-generated if omitted)
    --log_every_n_steps INT  Steps between training log lines (default 50)
"""

import argparse
import os
import random
import threading
import time

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
    p.add_argument("--window",       type=int, default=288)
    p.add_argument("--num_workers",  type=int, default=4)
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
    p.add_argument("--max_delta",    type=int,   default=18)
    p.add_argument("--factorised_encoder",  action="store_true",
                   help="Axial attention in encoder (~100× cheaper at W=288)")
    p.add_argument("--temporal_window",     type=int, default=0,
                   help="Local temporal attention window size in timesteps (0 = full). "
                        "W must be divisible by this value. Odd encoder layers use a "
                        "Swin-style half-window shift so tokens can communicate across "
                        "chunk boundaries after two layers. Only applies with "
                        "--factorised_encoder. Example: --window 72 --temporal_window 6 "
                        "gives 12 one-hour chunks, 12x cheaper temporal attention.")
    p.add_argument("--no_spatial_attn",     action="store_true",
                   help="Disable spatial attention sub-layer in factorised encoder blocks. "
                        "Each station is encoded independently from its own temporal window; "
                        "cross-station reasoning is delegated entirely to the decoder. "
                        "Only applies when --factorised_encoder is set. "
                        "Reduces encoder cost from O(N·W²+W·N²) to O(N·W²).")
    p.add_argument("--cross_attn_decoder",  action="store_true",
                   help="Cross-attention decoder (query tokens attend to encoder context)")
    p.add_argument("--grad_checkpoint",     action="store_true",
                   help="Gradient checkpointing (~33%% extra compute, ~66%% less VRAM)")

    # Training
    p.add_argument("--num_delta",    type=int,   default=1)
    p.add_argument("--epochs",       type=int,   default=100)
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
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--min_delta",    type=float, default=1e-4)

    # Checkpointing
    p.add_argument("--save_dir",        type=str, default="checkpoints")
    p.add_argument("--save_every",      type=int, default=5)
    p.add_argument("--save_every_steps", type=int, default=0,
                   help="Save a periodic checkpoint every N training steps (0 = off, "
                        "use --save_every epochs instead). Useful when one epoch is "
                        ">30 min: set to the same value as --val_check_interval so "
                        "checkpoints always align with a validation run.")
    p.add_argument("--resume",          type=str, default=None)

    # Sub-epoch validation / checkpointing
    p.add_argument("--val_check_interval", type=int, default=0,
                   help="Run validation every N training steps instead of every epoch "
                        "(0 = once per epoch, the default). EarlyStopping patience "
                        "then counts in validation-checks, not epochs. Example for a "
                        "50-min epoch on A100 (~6500 steps): use 4000 for ~30-min "
                        "checks. Set --save_every_steps to the same value to also save "
                        "checkpoints at each validation point.")

    # Logging / WandB
    p.add_argument("--wandb_project",   type=str, default=None,
                   help="WandB project name; omit to use CSV logger only")
    p.add_argument("--wandb_run_name",  type=str, default=None)
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
    )

    print("Building val dataset …")
    val_ds = StationMAEDataset(
        ds,
        window_size=args.window,
        delta_steps=args.max_delta,
        split="val",
        obs_stats=train_ds.obs_stats,       # always normalise with train-split stats
        num_delta_per_sample=1,             # single-delta for consistent val metrics
        max_delta_steps=args.max_delta,
        cache_dir=cache_dir,
        train_years=train_years,
        shared_memory=False,
        fast_cache_dir=fast_cache_dir,
        exclude_stations=args.exclude_stations,
    )

    print(f"  train: {len(train_ds):,} samples  "
          f"(years {train_years[0]}–{train_years[-1]})  |  "
          f"val: {len(val_ds):,} samples")
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
        encoder_spatial_attn=not args.no_spatial_attn,
        temporal_window=args.temporal_window,
        cross_attention_decoder=args.cross_attn_decoder,
    )

    if args.grad_checkpoint:
        print("Gradient checkpointing   : ON  (~33% extra compute, ~66% less VRAM)")
    if args.factorised_encoder:
        _spatial_str = "temporal-only (--no_spatial_attn)" if args.no_spatial_attn \
                       else "temporal + spatial"
        _tw_str = f"  |  temporal_window={args.temporal_window}" \
                  if args.temporal_window > 0 else ""
        print(f"Factorised encoder       : ON  ({_spatial_str}{_tw_str})")
    if args.cross_attn_decoder:
        print("Cross-attention decoder  : ON")
    print(f"Model: {model.count_parameters():,} trainable parameters")

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

    # Pass per-variable normalisation statistics into cfg so that
    # lightning_module.py can log physically-unnormalized val metrics.
    # obs_stats["std"] shape: (num_variables,) — index matches TARGET_VARIABLE_NAMES.
    _obs_stats = train_ds.obs_stats
    cfg = {
        "lr":                  args.lr,
        "min_lr":              args.min_lr,
        "weight_decay":        args.weight_decay,
        "epochs":              args.epochs,
        "warmup_epochs":       args.warmup_epochs,
        "log_every_n_steps":   args.log_every_n_steps,
        # Normalisation stats for unnormalized WandB metrics
        "obs_stats_std":       _obs_stats["std"].tolist(),   # list[float], len=num_vars
        "obs_stats_mean":      _obs_stats["mean"].tolist(),  # stored for test.py use
        # Informational — stored in checkpoint hyper_parameters for test.py
        "factorised_encoder":  args.factorised_encoder,
        "no_spatial_attn":     args.no_spatial_attn,
        "temporal_window":     args.temporal_window,
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
        "dropout":             args.dropout,
        "exclude_stations":    args.exclude_stations or [],
    }

    lit_model = StationMAELightning(model=model, cfg=cfg)

    # ── Logger ──────────────────────────────────────────────────────────────
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    if args.wandb_project:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),          # logs all CLI flags to WandB
            save_dir=args.save_dir,
        )
        print(f"WandB logger             : project='{args.wandb_project}'")
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
        # Best checkpoint by val/loss
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,             # also writes last.ckpt after every epoch
            verbose=True,
        ),
        # Periodic numbered snapshots — epoch-based or step-based
        *(
            [ModelCheckpoint(
                dirpath=args.save_dir,
                filename="step_{step:07d}",
                every_n_train_steps=args.save_every_steps,
                save_top_k=-1,
            )]
            if args.save_every_steps > 0 else
            [ModelCheckpoint(
                dirpath=args.save_dir,
                filename="epoch_{epoch:03d}",
                every_n_epochs=args.save_every,
                save_top_k=-1,
            )]
        ),
        # Learning rate monitor — logs train/lr to WandB automatically
        LearningRateMonitor(logging_interval="step"),
    ]

    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=args.patience,
                min_delta=args.min_delta,
                mode="min",
                verbose=True,
            )
        )
        _patience_unit = "steps" if _step_based else "epochs"
        _patience_val  = (args.patience * args.val_check_interval
                          if _step_based else args.patience)
        print(f"EarlyStopping            : patience={args.patience} checks  "
              f"(= {_patience_val} {_patience_unit})  min_delta={args.min_delta}")

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

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",             # auto-selects CUDA → MPS → CPU
        devices="auto",
        precision=precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
        deterministic=False,            # True slows training; seed already set
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
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
