"""
engine/train.py  (stub — training migrated to PyTorch Lightning)

The full training loop, optimiser, scheduler, early stopping, checkpointing
and AMP setup that used to live here are now handled by:

    model/lightning_module.py  — StationMAELightning (LightningModule)
    main.py                    — pl.Trainer setup, WandB logger, callbacks

This file is kept to avoid breaking any existing notebook imports.
"""
