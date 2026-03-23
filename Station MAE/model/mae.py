"""
model/mae.py

Full Station-MAE model — ties the encoder and decoder together.

Data flow
---------

   Input window (B, W, N, V)
        │
        │  +  x_mask  +  spatial  +  x_hours
        ▼
   StationMAEEncoder
        │
        │  encoded_vis  (B, W·N_vis, d_model)
        │  masked_indices  (B, N_masked)
        │  visible_indices (B, N_vis)
        ▼
   StationMAEDecoder
        │  + spatial  +  y_hours  +  delta_steps
        ▼
   preds  (B, N, V)
        │
        ▼
   _masked_loss  →  loss   (scalar, MSE on masked stations × present sensors)


Training objective
------------------
We mask a fraction of stations in the encoder (default 50 %).  The decoder
must predict all variables for EVERY station at the target time (t + Δt),
but the gradient signal flows only through the masked stations — visible
stations are used as context, not as supervision targets.

The loss is weighted by y_mask so that absent sensors (NaN originally,
replaced with 0) never contribute to the gradient.

Inference
---------
Call model.predict(…) for a no-grad forward pass that returns (B, N, V)
predictions for all stations without computing the loss.
"""

import torch
import torch.nn as nn

from .encoder import StationMAEEncoder
from .decoder import StationMAEDecoder
from .embeddings import (
    NUM_VARIABLES,
    SPATIAL_INPUT_DIM,
    TEMPORAL_FOURIER_DIM,
)


class StationMAE(nn.Module):
    """
    Station-MAE: Masked AutoEncoder for weather station data.

    Combines an asymmetric encoder–decoder pair following He et al. (2022)
    adapted for irregular station grids with topographic context and
    Aurora-inspired multi-scale temporal encodings.

    Key design choices
    ------------------
    * Station-level masking  — an entire station's window is hidden from the
      encoder, simulating gap-filling under missing-station scenarios.
    * Three embeddings per token  — variable (what), spatial (where),
      temporal Aurora Fourier (when).
    * Decoder query tokens use a fourth embedding: delta-time (lead).
    * Asymmetric depth  — deep encoder (default 4 layers), shallow decoder
      (default 2 layers) following the MAE recipe.

    Args:
        d_model:          Shared model dimension for encoder and decoder.
        enc_heads:        Encoder attention heads.
        enc_layers:       Encoder transformer depth.
        dec_heads:        Decoder attention heads.
        dec_layers:       Decoder transformer depth (lighter than encoder).
        mlp_ratio:        FFN hidden-dim = d_model × mlp_ratio.
        dropout:          Dropout applied in attention and FFN.
        mask_ratio:       Fraction of stations masked per sample (0–1).
        num_vars:         Number of meteorological variables.
        spatial_dim:      Static feature dimension (14).
        fourier_dim:      Fourier feature dimension for temporal embedding (32).
        max_delta_steps:  Maximum forecast lead-time in 10-min steps (36 = 6 h).
    """

    def __init__(
        self,
        d_model:          int   = 128,
        enc_heads:        int   = 4,
        enc_layers:       int   = 4,
        dec_heads:        int   = 4,
        dec_layers:       int   = 2,
        mlp_ratio:        float = 4.0,
        dropout:          float = 0.1,
        mask_ratio:       float = 0.5,
        num_vars:         int   = NUM_VARIABLES,
        spatial_dim:      int   = SPATIAL_INPUT_DIM,
        fourier_dim:      int   = TEMPORAL_FOURIER_DIM,
        max_delta_steps:  int   = 36,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.num_vars   = num_vars

        self.encoder = StationMAEEncoder(
            d_model=d_model,
            num_heads=enc_heads,
            num_layers=enc_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mask_ratio=mask_ratio,
            num_vars=num_vars,
            spatial_dim=spatial_dim,
            fourier_dim=fourier_dim,
        )

        self.decoder = StationMAEDecoder(
            d_model=d_model,
            num_heads=dec_heads,
            num_layers=dec_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_vars=num_vars,
            spatial_dim=spatial_dim,
            fourier_dim=fourier_dim,
            max_delta=max_delta_steps,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:           torch.Tensor,   # (B, W, N, V)   normalised input window
        x_mask:      torch.Tensor,   # (B, W, N, V)   sensor availability (1=present)
        spatial:     torch.Tensor,   # (N, 14) or (B, N, 14)
        x_hours:     torch.Tensor,   # (B, W)          hours-since-epoch per input step
        y:           torch.Tensor,   # (B, N, V)       normalised target snapshot
        y_mask:      torch.Tensor,   # (B, N, V)       target sensor availability
        y_hours:     torch.Tensor,   # (B,)             hours-since-epoch for target
        delta_steps: torch.Tensor,   # (B,)             lead-time in 10-min steps
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full training forward pass.

        Returns:
            loss:           scalar — MSE loss on masked stations (present sensors only)
            preds:          (B, N, V) — predictions for all stations at target time
            masked_indices: (B, N_masked) — which station indices were masked
        """
        # 1. Encode: process visible station tokens only
        encoded, masked_idx, visible_idx = self.encoder(
            x, x_mask, spatial, x_hours
        )
        # encoded:    (B, W*N_vis, d_model)
        # masked_idx: (B, N_masked)

        # 2. Decode: predict all N stations at the target time
        preds = self.decoder(encoded, spatial, y_hours, delta_steps)
        # preds: (B, N, V)

        # 3. Compute loss on masked stations only
        loss = self._masked_loss(preds, y, y_mask, masked_idx)

        return loss, preds, masked_idx

    # ------------------------------------------------------------------
    # Inference (no grad, no loss)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x:           torch.Tensor,
        x_mask:      torch.Tensor,
        spatial:     torch.Tensor,
        x_hours:     torch.Tensor,
        y_hours:     torch.Tensor,
        delta_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-only forward pass — no masking randomness (eval mode) and
        no loss computation.

        Args:
            (same shapes as forward, but y / y_mask not needed)

        Returns:
            preds: (B, N, V) predictions for all stations at target time.
        """
        self.eval()
        encoded, _, _ = self.encoder(x, x_mask, spatial, x_hours)
        preds         = self.decoder(encoded, spatial, y_hours, delta_steps)
        return preds

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_loss(
        preds:          torch.Tensor,   # (B, N, V)
        y:              torch.Tensor,   # (B, N, V)
        y_mask:         torch.Tensor,   # (B, N, V)  1.0 = sensor present
        masked_indices: torch.Tensor,   # (B, N_masked)
    ) -> torch.Tensor:
        """
        MSE loss restricted to:
          (a) stations that were masked by the encoder, AND
          (b) sensors that were present at the target timestep (y_mask == 1).

        Stations masked but with no active sensors at target time contribute
        zero to the loss (no NaN: denominator is clamped to ≥ 1).

        Returns:
            scalar loss tensor.
        """
        B, N, V = preds.shape

        # Binary mask over stations: True where station was masked
        # scatter_ places True at positions given by masked_indices along dim 1
        station_masked = torch.zeros(B, N, dtype=torch.bool, device=preds.device)
        station_masked.scatter_(1, masked_indices, True)              # (B, N)

        # Combine: station must be masked AND sensor must be present at target
        sensor_ok  = y_mask.bool()                                    # (B, N, V)
        full_mask  = station_masked.unsqueeze(-1) & sensor_ok         # (B, N, V)

        sq_err = (preds - y).pow(2)                                   # (B, N, V)
        denom  = full_mask.float().sum().clamp(min=1.0)
        loss   = (sq_err * full_mask.float()).sum() / denom

        return loss

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
