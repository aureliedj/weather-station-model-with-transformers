"""
model/embeddings.py

Embedding modules shared by the encoder and the decoder.

An encoder token is the sum of five components:

    v   VariableProjection   per-variable observation value (or a learned
                             "sensor absent" vector)                 -> d_model
    p1  PositionalEmbedding  Fourier features of the LV95 coordinates -> d_model
    p2  StationEmbedding     MLP over 13 topographic descriptors      -> d_model
    t   TemporalEmbedding    Fourier features of absolute time        -> d_model
    s   StepIndexEmbedding   Fourier features of the step index       -> d_model

Decoder queries use p1, p2, t, s and additionally

    d   DeltaTimeEmbedding   Fourier features of the lead time        -> d_model

All Fourier embeddings use fixed wavelengths lambda_i and the features
[cos(2*pi*x/lambda_i), sin(2*pi*x/lambda_i)], followed by a two-layer MLP.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Input variables, in tensor order. Wind is given as (u, v) components.
VARIABLE_NAMES = [
    "temperature",
    "pressure",
    "humidity",
    "wind_u",
    "wind_v",
    "precipitation",
]
NUM_VARIABLES = len(VARIABLE_NAMES)   # 6

# Predicted variables. Precipitation is an input only (zero-inflated).
TARGET_VARIABLE_NAMES = [v for v in VARIABLE_NAMES if v != "precipitation"]
NUM_TARGET_VARIABLES = len(TARGET_VARIABLE_NAMES)   # 5

# Static station features, see data/dataset.py::build_spatial_features.
# Columns 0:2 are the LV95 coordinates (-> PositionalEmbedding),
# columns 2:15 the topographic descriptors (-> StationEmbedding):
#   2,3   sin/cos ASPECT_2000M        8   TPI_2000M
#   4,5   sin/cos ASPECT_10000M       9   SLOPE_2000M
#   6     station_height              10  SLOPE_10000M
#   7     dem                         11  SN_DERIVATIVE_2000M
#                                     12  SN_DERIVATIVE_10000M
#                                     13  WE_DERIVATIVE_2000M
#                                     14  WE_DERIVATIVE_10000M
SPATIAL_INPUT_DIM = 15
POSITION_DIM      = 2
STATION_CHAR_DIM  = 13

# Fourier features per coordinate for PositionalEmbedding (8 wavelengths).
# In normalised-coordinate units lambda 0.1 .. 10 spans roughly 8.5 km .. 850 km.
POSITION_FOURIER_DIM = 16

# Fourier features for TemporalEmbedding: 16 explicit wavelengths in hours.
TEMPORAL_FOURIER_DIM = 32
TEMPORAL_WAVELENGTHS_H = (
    # diurnal harmonics 24/k, k = 1..6
    24.0, 12.0, 8.0, 6.0, 4.8, 4.0,
    # synoptic to monthly (2 d, 3 d, 5 d, 8 d, 14 d, 30 d, 60 d)
    48.0, 72.0, 120.0, 192.0, 336.0, 720.0, 1440.0,
    # annual harmonics 8766/k, k = 3, 2, 1
    2922.0, 4383.0, 8766.0,
)
assert 2 * len(TEMPORAL_WAVELENGTHS_H) == TEMPORAL_FOURIER_DIM

# Fourier features for DeltaTimeEmbedding (8 wavelengths, 1.25 h .. 8 h).
DELTA_FOURIER_DIM = 16

# Value-embedding MLP (scalar -> H -> d_model), see VariableProjection.
VALUE_MLP_HIDDEN = 32
VALUE_MLP_THRESH = 3.0    # first-layer biases ~ U(-3, 3): thresholds over +-3 sigma
VALUE_MLP_GAIN   = 1.47   # second-layer init gain, divided by sqrt(H)
ABSENT_INIT_STD  = 0.5    # init scale of the "sensor absent" vectors


def encode_temporal(ts: pd.Timestamp) -> float:
    """Hours elapsed since 1970-01-01 00:00 UTC (input of TemporalEmbedding)."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return float((ts - epoch).total_seconds() / 3600.0)


def _mlp(in_dim: int, d_model: int, dropout: float) -> nn.Sequential:
    """Projection head shared by all Fourier embeddings."""
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.GELU(),
        nn.LayerNorm(d_model),
        nn.Dropout(dropout),
        nn.Linear(d_model, d_model),
    )


# ---------------------------------------------------------------------------
# Embedding modules
# ---------------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    """
    Fourier embedding of the normalised LV95 coordinates (easting, northing).

    Each coordinate is expanded independently with log-spaced wavelengths from
    ``lambda_min`` to ``lambda_max`` (normalised-coordinate units); the two
    feature sets are concatenated and projected to d_model.
    """

    def __init__(
        self,
        d_model:     int   = 128,
        fourier_dim: int   = POSITION_FOURIER_DIM,
        lambda_min:  float = 0.1,
        lambda_max:  float = 10.0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even"
        self.fourier_dim = fourier_dim
        n_wl = fourier_dim // 2
        lambdas = torch.exp(torch.linspace(math.log(lambda_min), math.log(lambda_max), n_wl))
        self.register_buffer("lambdas", lambdas)
        self.proj = _mlp(2 * fourier_dim, d_model, dropout)

    def _fourier_1d(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * x.unsqueeze(-1) / self.lambdas
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """pos: (..., 2) normalised [easting, northing] -> (..., d_model)."""
        feats = torch.cat([self._fourier_1d(pos[..., 0]),
                           self._fourier_1d(pos[..., 1])], dim=-1)
        return self.proj(feats)


class StationEmbedding(nn.Module):
    """Two-layer MLP over the 13 normalised topographic descriptors."""

    def __init__(self, d_model: int, input_dim: int = STATION_CHAR_DIM,
                 dropout: float = 0.0):
        super().__init__()
        self.proj = _mlp(input_dim, d_model, dropout)

    def forward(self, char_features: torch.Tensor) -> torch.Tensor:
        """char_features: (..., 13) -> (..., d_model)."""
        return self.proj(char_features)


class TemporalEmbedding(nn.Module):
    """
    Fourier embedding of absolute time (hours since the Unix epoch) over the
    explicit periods in TEMPORAL_WAVELENGTHS_H: diurnal harmonics, a synoptic
    band and annual harmonics.

    The features are computed in float64: the input is ~4.9e5 hours, so the
    phase 2*pi*x/lambda would lose several hundredths of a radian in float32.
    """

    def __init__(
        self,
        d_model:     int   = 128,
        fourier_dim: int   = TEMPORAL_FOURIER_DIM,
        wavelengths: "tuple | None" = None,
        dropout:     float = 0.0,
    ):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even"
        self.fourier_dim = fourier_dim
        wl = tuple(wavelengths) if wavelengths is not None else TEMPORAL_WAVELENGTHS_H
        assert 2 * len(wl) == fourier_dim, (
            f"fourier_dim={fourier_dim} needs {fourier_dim // 2} wavelengths, got {len(wl)}")
        self.register_buffer("lambdas", torch.tensor(wl, dtype=torch.float64))
        self.proj = _mlp(fourier_dim, d_model, dropout)

    def _fourier(self, hours: torch.Tensor) -> torch.Tensor:
        x      = hours.double().unsqueeze(-1)
        angles = 2.0 * math.pi * x / self.lambdas.double()
        feats  = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return feats.to(hours.dtype)

    def forward(self, hours: torch.Tensor) -> torch.Tensor:
        """hours: (B,) or (B, W) -> (B, d_model) or (B, W, d_model)."""
        return self.proj(self._fourier(hours))


class DeltaTimeEmbedding(nn.Module):
    """
    Fourier embedding of the lead time (decoder only).

    ``delta_steps`` (10-min steps) is converted to hours and expanded with
    log-spaced wavelengths from ``lambda_min`` to ``lambda_max``. lambda_min
    must stay strictly above twice the lead-time grid spacing (0.5 h), or the
    shortest sine channel is identically zero on the grid.
    """

    def __init__(
        self,
        d_model:      int   = 128,
        fourier_dim:  int   = DELTA_FOURIER_DIM,
        lambda_min:   float = 1.25,
        lambda_max:   float = 8.0,
        step_size_h:  float = 1.0 / 6.0,
        dropout:      float = 0.0,
    ):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even"
        self.fourier_dim = fourier_dim
        self.step_size_h = step_size_h
        n_wl = fourier_dim // 2
        lambdas = torch.exp(torch.linspace(math.log(lambda_min), math.log(lambda_max), n_wl))
        self.register_buffer("lambdas", lambdas)
        self.proj = _mlp(fourier_dim, d_model, dropout)

    def forward(self, delta_steps: torch.Tensor) -> torch.Tensor:
        """delta_steps: (...) integer lead times in 10-min steps -> (..., d_model)."""
        hours   = delta_steps.float() * self.step_size_h
        angles  = 2.0 * math.pi * hours.unsqueeze(-1) / self.lambdas
        fourier = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.proj(fourier)


class StepIndexEmbedding(nn.Module):
    """
    Fourier embedding of an integer step index on a timeline shared by the
    encoder (input steps 0 .. W-1) and the decoder (query at W-1+delta).

    Wavelengths are log-spaced from 2.5 steps (strictly above the Nyquist
    wavelength of the integer grid) to ``max_steps``.
    """

    def __init__(
        self,
        d_model:     int   = 128,
        fourier_dim: int   = 32,
        max_steps:   int   = 512,
        dropout:     float = 0.0,
    ):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even"
        n_wl = fourier_dim // 2
        lambdas = torch.exp(torch.linspace(math.log(2.5), math.log(float(max_steps)), n_wl))
        self.register_buffer("lambdas", lambdas)
        self.proj = _mlp(fourier_dim, d_model, dropout)

    def _fourier(self, steps: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * steps.float().unsqueeze(-1) / self.lambdas
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        """steps: (W,), (B,) or (B, K) integer indices -> (..., d_model)."""
        return self.proj(self._fourier(steps))


class VariableProjection(nn.Module):
    """
    Observation embedding: one small MLP per variable, plus a learned
    "sensor absent" vector per variable.

        present:  e_v = W2_v @ (GELU(a_v * x_v + b_v) - GELU(b_v)) + c_v
        absent:   e_v = var_absent_embedding[v]

    The V contributions are summed and divided by sqrt(V), so the scale of the
    branch does not depend on the number of variables. The GELU is centred
    (GELU(b) subtracted) so that a zero observation maps to a zero embedding
    and the branch carries no token-independent constant.

    Initialisation: |a_v| = 1 with random sign and b_v ~ U(-3, 3), so the H
    hidden units act as thresholds spread uniformly over +-3 standard
    deviations of the per-station normalised observations.
    """

    def __init__(self, num_vars: int = NUM_VARIABLES, d_model: int = 256):
        super().__init__()
        self.num_vars = num_vars
        self.d_model  = d_model
        H = VALUE_MLP_HIDDEN
        self.mlp_w1 = nn.Parameter(torch.empty(num_vars, H))
        self.mlp_b1 = nn.Parameter(torch.empty(num_vars, H))
        self.mlp_w2 = nn.Parameter(torch.empty(num_vars, H, d_model))
        self.mlp_b2 = nn.Parameter(torch.zeros(num_vars, d_model))
        with torch.no_grad():
            self.mlp_w1.copy_(torch.randint(0, 2, self.mlp_w1.shape).float() * 2 - 1)
            nn.init.uniform_(self.mlp_b1, -VALUE_MLP_THRESH, VALUE_MLP_THRESH)
            bound = VALUE_MLP_GAIN / math.sqrt(H)
            nn.init.uniform_(self.mlp_w2, -bound, bound)
        self.var_absent_embedding = nn.Parameter(torch.zeros(num_vars, d_model))
        nn.init.trunc_normal_(self.var_absent_embedding, std=ABSENT_INIT_STD)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, N, V) normalised values, 0.0 where the sensor is absent.
            mask: (B, N, V) 1.0 present / 0.0 absent.
        Returns:
            (B, N, d_model)
        """
        m = mask.unsqueeze(-1)                                              # (B, N, V, 1)
        h = F.gelu(x.unsqueeze(-1) * self.mlp_w1 + self.mlp_b1) - F.gelu(self.mlp_b1)
        e = torch.einsum("...vh,vhd->...vd", h, self.mlp_w2) + self.mlp_b2  # (B, N, V, d)
        e = e * m + self.var_absent_embedding * (1.0 - m)
        return e.sum(dim=-2) / math.sqrt(self.num_vars)
