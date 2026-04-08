"""
engine/train.py

Training loop for Station-MAE.

Public API
----------
    EarlyStopping                          — stops when val_loss plateaus
    train_one_epoch(...)  → avg_loss       — single epoch, single- or multi-delta
    build_optimizer(...)  → AdamW
    build_scheduler(...)  → LambdaLR
    train(...)            → None           — full loop with logging + checkpointing

Multi-delta support
-------------------
When the DataLoader yields batches where ``delta_steps`` has shape (B, K),
``train_one_epoch`` calls ``model.forward_multi_delta()``, which runs the
encoder once and the decoder K times.  The reported loss is the mean over K.

Checkpointing strategy
----------------------
    last.pt          — overwritten every epoch (safe resume point)
    best.pt          — overwritten whenever val_loss improves
    epoch_{N:03d}.pt — saved every ``save_every`` epochs (for history)

Loss log
--------
    <save_dir>/logs.csv  — epoch, train_loss, val_loss, val_rmse, epoch_secs
    Written incrementally so the file survives a crash mid-run.

Config keys (passed as dict ``cfg``)
--------------------------------------
    lr              float   (default 1e-4)
    weight_decay    float   (default 0.05)
    epochs          int     (default 100)
    warmup_epochs   int     (default 5)
    patience        int     early-stop patience, 0 = disabled (default 10)
    min_delta       float   minimum val_loss improvement to reset counter (1e-4)
    save_dir        str     (default "checkpoints")
    save_every      int     save numbered checkpoint every N epochs (default 5)
    log_interval    int     steps between log lines (default 50)
    amp             bool    use torch AMP (default True)
    grad_clip       float   (default 1.0)
    num_delta       int     K lead-times per sample, forwarded for logging (default 1)
"""

import csv
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when validation loss has not improved for ``patience`` epochs.

    A relative ``min_delta`` threshold prevents spurious stops from tiny
    fluctuations in loss.  The best seen loss is always tracked regardless of
    whether improvement was large enough to reset the counter.

    Usage::

        es = EarlyStopping(patience=10, min_delta=1e-4)
        for epoch in ...:
            val_loss = evaluate(...)
            if es.step(val_loss):
                print(f"Early stop — best {es.best:.6f}")
                break

    Args:
        patience:  Consecutive epochs without improvement before stopping
                   (0 = disabled, training always runs to completion).
        min_delta: Minimum absolute improvement counted as "new best".
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    @property
    def disabled(self) -> bool:
        return self.patience == 0

    def step(self, val_loss: float) -> bool:
        """
        Update internal state with the latest validation loss.

        Returns:
            True if training should stop, False otherwise.
            Always returns False when patience == 0 (disabled).
        """
        if self.disabled:
            return False

        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

def build_optimizer(
    model:        nn.Module,
    lr:           float = 1e-4,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """
    AdamW with decoupled weight decay.

    Biases, LayerNorm parameters, embeddings and the learnable mask token
    are excluded from weight decay (standard transformer pre-training practice).

    Returns:
        AdamW optimiser with two param-groups (decay / no-decay).
    """
    decay_params   = []
    nodecay_params = []
    no_decay_kw    = ("bias", "norm", "embedding", "mask_token", "lambdas")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_kw):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(
    optimizer:           torch.optim.Optimizer,
    num_warmup_steps:    int,
    num_training_steps:  int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup followed by cosine annealing to zero.

    Args:
        optimizer:           Optimiser to schedule.
        num_warmup_steps:    Steps for linear LR ramp from 0 → base_lr.
        num_training_steps:  Total training steps.

    Returns:
        LambdaLR scheduler; call scheduler.step() after each optimiser step.
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
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    scheduler:    "torch.optim.lr_scheduler._LRScheduler | None",
    device:       torch.device,
    scaler:       "torch.cuda.amp.GradScaler | None" = None,
    grad_clip:    float = 1.0,
    log_interval: int   = 50,
    epoch:        int   = 0,
) -> float:
    """
    Run one training epoch.

    Automatically detects single-delta vs multi-delta batches:
      - ``delta_steps`` shape (B,)   → calls ``model.forward()``
      - ``delta_steps`` shape (B, K) → calls ``model.forward_multi_delta()``

    Args:
        model:         StationMAE instance moved to ``device``.
        loader:        DataLoader yielding StationMAEDataset batches.
        optimizer:     AdamW (or other).
        scheduler:     Per-step LR scheduler, or None.
        device:        torch.device.
        scaler:        GradScaler for CUDA AMP, or None.
        grad_clip:     Max gradient norm.
        log_interval:  Print loss every N steps.
        epoch:         Current epoch index (for display).

    Returns:
        avg_loss: float — mean batch loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    for step, batch in enumerate(loader):
        # ── Move batch to device ────────────────────────────────────────
        x           = batch["x"].to(device)
        x_mask      = batch["x_mask"].to(device)
        spatial     = batch["spatial"].to(device)
        x_hours     = batch["x_hours"].to(device)
        y           = batch["y"].to(device)
        y_mask      = batch["y_mask"].to(device)
        y_hours     = batch["y_hours"].to(device)
        delta_steps = batch["delta_steps"].to(device)

        # spatial may be (B, N, 15) from the collate — use first item
        if spatial.dim() == 3 and spatial.size(0) == x.size(0):
            spatial = spatial[0]   # (N, 15)

        # ── Choose forward pass ─────────────────────────────────────────
        # Multi-delta: delta_steps is (B, K)
        # Single-delta: delta_steps is (B,)
        is_multi = (delta_steps.dim() == 2)
        forward_fn = model.forward_multi_delta if is_multi else model.forward

        optimizer.zero_grad()

        # ── Forward + backward ──────────────────────────────────────────
        if scaler is not None:
            with torch.autocast(device_type=device.type):
                loss, _, _ = forward_fn(
                    x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        elif device.type == "mps":
            with torch.autocast(device_type="mps"):
                loss, _, _ = forward_fn(
                    x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        else:
            loss, _, _ = forward_fn(
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
            mode    = f"K={delta_steps.shape[1]}" if is_multi else "K=1"
            print(
                f"  Epoch {epoch:3d}  step {step + 1:5d}/{len(loader)}  "
                f"loss {loss.item():.5f}  lr {lr_now:.2e}  "
                f"[{mode}]  ({elapsed:.1f}s)",
                flush=True,
            )

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   "DataLoader | None",
    cfg:          dict,
    device:       torch.device,
) -> None:
    """
    Full training loop with validation, early stopping, checkpointing and
    learning-rate scheduling.

    Expected cfg keys (with defaults):
        lr              (float,  1e-4)
        weight_decay    (float,  0.05)
        epochs          (int,    100)
        warmup_epochs   (int,    5)
        patience        (int,    10)   — early-stop patience; 0 = disabled
        min_delta       (float,  1e-4) — min val_loss drop to reset counter
        grad_clip       (float,  1.0)
        log_interval    (int,    50)
        save_dir        (str,    "checkpoints")
        save_every      (int,    5)    — save numbered checkpoint every N epochs
        amp             (bool,   True)

    Checkpoints
    -----------
        last.pt          — latest epoch (always; safe resume point)
        best.pt          — best val_loss (updated whenever val_loss improves)
        epoch_{N:03d}.pt — periodic numbered snapshots (every save_every epochs)

    Loss log
    --------
        <save_dir>/logs.csv  — columns: epoch,train_loss,val_loss,val_rmse,epoch_secs
    """
    lr            = cfg.get("lr",            1e-4)
    weight_decay  = cfg.get("weight_decay",  0.05)
    epochs        = cfg.get("epochs",        100)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    patience      = cfg.get("patience",      10)
    min_delta     = cfg.get("min_delta",     1e-4)
    grad_clip     = cfg.get("grad_clip",     1.0)
    log_interval  = cfg.get("log_interval",  50)
    save_dir      = cfg.get("save_dir",      "checkpoints")
    save_every    = cfg.get("save_every",    5)
    use_amp       = cfg.get("amp",           True) and device.type in ("cuda", "mps")

    os.makedirs(save_dir, exist_ok=True)

    # ── CSV loss log ─────────────────────────────────────────────────────
    log_path = os.path.join(save_dir, "logs.csv")
    write_header = not os.path.exists(log_path)
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "train_loss", "val_loss", "val_rmse",
                              "epoch_secs"])
        log_file.flush()

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer    = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    total_steps  = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # ── Early stopping ────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)
    if patience > 0:
        print(f"[EarlyStopping] patience={patience}  min_delta={min_delta}")

    best_val = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # ── Train ────────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            scaler=scaler, grad_clip=grad_clip,
            log_interval=log_interval, epoch=epoch,
        )

        # ── Validate ─────────────────────────────────────────────────────
        val_loss = None
        val_rmse = None
        if val_loader is not None:
            from evaluate import evaluate   # avoid circular import
            metrics  = evaluate(model, val_loader, device)
            val_loss = metrics.get("avg_loss",     float("nan"))
            val_rmse = metrics.get("overall_rmse", float("nan"))

        elapsed = time.time() - t_epoch

        # ── Console log ──────────────────────────────────────────────────
        val_str = (
            f"  val_loss {val_loss:.5f}  val_rmse {val_rmse:.5f}"
            if val_loss is not None else "  (no validation)"
        )
        es_str = (
            f"  [ES {early_stop.counter}/{patience}]"
            if not early_stop.disabled else ""
        )
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss {train_loss:.5f}{val_str}{es_str}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        # ── CSV log ───────────────────────────────────────────────────────
        log_writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}"  if val_loss is not None else "",
            f"{val_rmse:.6f}"  if val_rmse is not None else "",
            f"{elapsed:.1f}",
        ])
        log_file.flush()

        # ── Checkpointing ─────────────────────────────────────────────────
        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
            "val_rmse":             val_rmse,
            "cfg":                  cfg,
        }

        # Always save last.pt — safe resume point regardless of save_every
        torch.save(ckpt, os.path.join(save_dir, "last.pt"))

        # Periodic numbered snapshots
        if epoch % save_every == 0:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))

        # Best model by validation loss
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            print(
                f"  ✓ New best val_loss {best_val:.5f}  — saved best.pt",
                flush=True,
            )

        # ── Early stopping check ──────────────────────────────────────────
        if val_loss is not None and early_stop.step(val_loss):
            print(
                f"\n[EarlyStopping] Triggered at epoch {epoch}  "
                f"(no improvement for {patience} epochs)  "
                f"best val_loss = {early_stop.best:.6f}",
                flush=True,
            )
            break

    log_file.close()
    print(f"\nTraining complete.  Checkpoints saved to: {save_dir}", flush=True)
