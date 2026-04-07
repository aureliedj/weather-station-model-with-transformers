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
    NUM_TARGET_VARIABLES,
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
        num_vars:         Input variables per station (6: temp, pres, hum, wind_u, wind_v, precip).
        num_target_vars:  Predicted variables per station (5: all except precipitation).
                          Precipitation is used as input context but excluded from the
                          loss — its zero-inflated distribution makes MSE unsuitable.
        fourier_dim:      Fourier feature dimension for temporal embedding (32).
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
        num_target_vars:  int   = NUM_TARGET_VARIABLES,
        fourier_dim:      int   = TEMPORAL_FOURIER_DIM,
    ):
        super().__init__()

        self.mask_ratio      = mask_ratio
        self.num_vars        = num_vars
        self.num_target_vars = num_target_vars

        self.encoder = StationMAEEncoder(
            d_model=d_model,
            num_heads=enc_heads,
            num_layers=enc_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mask_ratio=mask_ratio,
            num_vars=num_vars,
            fourier_dim=fourier_dim,
        )

        self.decoder = StationMAEDecoder(
            d_model=d_model,
            num_heads=dec_heads,
            num_layers=dec_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_vars=num_vars,
            num_target_vars=num_target_vars,
            fourier_dim=fourier_dim,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:           torch.Tensor,   # (B, W, N, V)   normalised input window
        x_mask:      torch.Tensor,   # (B, W, N, V)   sensor availability (1=present)
        spatial:     torch.Tensor,   # (N, 15) or (B, N, 15)
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
            preds:          (B, N, num_target_vars) — predictions for all stations
                            num_target_vars = 5 (excludes precipitation)
            masked_indices: (B, N_masked) — which station indices were masked
        """
        # 1. Encode: process visible station tokens only
        encoded, masked_idx, visible_idx = self.encoder(
            x, x_mask, spatial, x_hours
        )
        # encoded:    (B, W*N_vis, d_model)
        # masked_idx: (B, N_masked)

        # 2. Decode: predict target variables for all N stations at the target time
        preds = self.decoder(encoded, spatial, y_hours, delta_steps)
        # preds: (B, N, num_target_vars)

        # 3. Slice y and y_mask to target variables only (drop precipitation column)
        y_target      = y[:, :, :self.num_target_vars]       # (B, N, num_target_vars)
        y_mask_target = y_mask[:, :, :self.num_target_vars]  # (B, N, num_target_vars)

        # 4. Compute loss on masked stations only
        loss = self._masked_loss(preds, y_target, y_mask_target, masked_idx)

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
            preds: (B, N, num_target_vars) predictions for all stations at target time.
                   num_target_vars = 5 (temperature, pressure, humidity, wind_u, wind_v).
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
