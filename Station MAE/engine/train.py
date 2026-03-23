"""
engine/train.py

Training loop for Station-MAE.

Public API
----------
    train_one_epoch(model, loader, optimizer, scheduler, device, scaler)
        → avg_loss: float

    build_optimizer(model, lr, weight_decay)
        → torch.optim.AdamW  (separate param-groups for decay / no-decay)

    build_scheduler(optimizer, num_warmup_steps, num_training_steps)
        → LambdaLR  (linear warmup + cosine annealing)

    train(model, train_loader, val_loader, cfg, device)
        → None  (saves checkpoints to cfg["save_dir"])

Config keys (passed as dict `cfg`)
------------------------------------
    lr              float   (default 1e-4)
    weight_decay    float   (default 0.05)
    epochs          int     (default 100)
    warmup_epochs   int     (default 5)
    save_dir        str     (default "checkpoints")
    log_interval    int     (default 50, steps between log lines)
    amp             bool    (default True, use torch.cuda.amp)
    grad_clip       float   (default 1.0)
"""

import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(
    model:        nn.Module,
    lr:           float = 1e-4,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """
    AdamW with decoupled weight decay.

    Biases, LayerNorm parameters, embedding parameters and the learnable
    mask token are excluded from weight decay (common best practice for
    transformer pre-training).

    Args:
        model:        nn.Module to optimise.
        lr:           Base learning rate.
        weight_decay: Weight decay for non-bias/norm params.

    Returns:
        AdamW optimiser with two param-groups.
    """
    decay_params   = []
    nodecay_params = []

    no_decay_keywords = ("bias", "norm", "embedding", "mask_token", "lambdas")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),   # following MAE paper
        eps=1e-8,
    )
    return optimizer


def build_scheduler(
    optimizer:           torch.optim.Optimizer,
    num_warmup_steps:    int,
    num_training_steps:  int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup followed by cosine annealing to zero.

    Args:
        optimizer:            Optimiser to schedule.
        num_warmup_steps:     Steps over which LR ramps linearly from 0 → base_lr.
        num_training_steps:   Total training steps.

    Returns:
        LambdaLR scheduler (call scheduler.step() after each optimiser step).
    """
    def _lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(num_warmup_steps, 1)
        progress = float(step - num_warmup_steps) / max(
            num_training_steps - num_warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device:    torch.device,
    scaler:    torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
    log_interval: int = 50,
    epoch:     int   = 0,
) -> float:
    """
    Run one training epoch.

    Args:
        model:         StationMAE (or any nn.Module with the same forward signature).
        loader:        DataLoader yielding batches from StationMAEDataset.
        optimizer:     AdamW (or other).
        scheduler:     Per-step LR scheduler (or None).
        device:        torch.device.
        scaler:        GradScaler for AMP (or None for fp32).
        grad_clip:     Max gradient norm (default 1.0).
        log_interval:  Print loss every N steps.
        epoch:         Current epoch index (for logging).

    Returns:
        avg_loss: float — mean batch loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    for step, batch in enumerate(loader):
        # ---- Move to device ----------------------------------------
        x           = batch["x"].to(device)            # (B, W, N, V)
        x_mask      = batch["x_mask"].to(device)       # (B, W, N, V)
        spatial     = batch["spatial"].to(device)      # (N, 14) or (B, N, 14)
        x_hours     = batch["x_hours"].to(device)      # (B, W)
        y           = batch["y"].to(device)             # (B, N, V)
        y_mask      = batch["y_mask"].to(device)        # (B, N, V)
        y_hours     = batch["y_hours"].to(device)       # (B,)
        delta_steps = batch["delta_steps"].to(device)   # (B,)

        # spatial may be (N, 14) when collated from samples sharing the same
        # station set — squeeze the batch dim if accidentally added
        if spatial.dim() == 3 and spatial.size(0) == x.size(0):
            spatial = spatial[0]  # (N, 14)

        # ---- Forward + loss ----------------------------------------
        optimizer.zero_grad()

        if scaler is not None:
            # CUDA path: GradScaler + autocast
            with torch.autocast(device_type=device.type):
                loss, _, _ = model(
                    x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        elif device.type == "mps":
            # MPS path: autocast without GradScaler
            with torch.autocast(device_type="mps"):
                loss, _, _ = model(
                    x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            loss, _, _ = model(
                x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}  step {step + 1:5d}/{len(loader)}  "
                f"loss {loss.item():.5f}  lr {lr_now:.2e}  "
                f"({elapsed:.1f}s elapsed)"
            )

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader | None,
    cfg:          dict,
    device:       torch.device,
) -> None:
    """
    Full training loop with validation, checkpointing and learning-rate
    scheduling.

    Expected cfg keys (with defaults):
        lr              (float,  1e-4)
        weight_decay    (float,  0.05)
        epochs          (int,    100)
        warmup_epochs   (int,    5)
        grad_clip       (float,  1.0)
        log_interval    (int,    50)
        save_dir        (str,    "checkpoints")
        amp             (bool,   True)

    Checkpoints are saved as:
        <save_dir>/epoch_{N:03d}.pt   — after every epoch
        <save_dir>/best.pt            — best validation loss

    The checkpoint dict contains:
        epoch, model_state_dict, optimizer_state_dict,
        scheduler_state_dict, train_loss, val_loss (or None)
    """
    lr            = cfg.get("lr",            1e-4)
    weight_decay  = cfg.get("weight_decay",  0.05)
    epochs        = cfg.get("epochs",        100)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    grad_clip     = cfg.get("grad_clip",     1.0)
    log_interval  = cfg.get("log_interval",  50)
    save_dir      = cfg.get("save_dir",      "checkpoints")
    use_amp       = cfg.get("amp",           True) and device.type in ("cuda", "mps")

    os.makedirs(save_dir, exist_ok=True)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)

    total_steps  = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)

    # GradScaler is CUDA-only; MPS uses autocast without a scaler
    scaler   = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ---- Training --------------------------------------------------
        t_epoch = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            scaler=scaler, grad_clip=grad_clip,
            log_interval=log_interval, epoch=epoch,
        )

        # ---- Validation ------------------------------------------------
        val_loss = None
        if val_loader is not None:
            from engine.evaluate import evaluate   # lazy import to avoid circulars
            metrics  = evaluate(model, val_loader, device)
            val_loss = metrics.get("overall_rmse", float("nan"))

        # ---- Logging ---------------------------------------------------
        elapsed = time.time() - t_epoch
        val_str = f"  val_rmse {val_loss:.5f}" if val_loss is not None else ""
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss {train_loss:.5f}{val_str}  "
            f"({elapsed:.1f}s)"
        )

        # ---- Checkpointing ---------------------------------------------
        ckpt = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss":          train_loss,
            "val_loss":            val_loss,
        }
        torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            print(f"  ✓ New best val_rmse: {best_val:.5f} — saved best.pt")
