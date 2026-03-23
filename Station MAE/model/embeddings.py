"""
model/embeddings.py

Embedding modules for Station-MAE.

  - SpatialEmbedding    : static station metadata (lat/lon/topo) → d_model
  - TemporalEmbedding   : multi-scale Fourier time encoding       → d_model
  - DeltaTimeEmbedding  : forecast lead-time steps                → d_model  [decoder only]
  - VariableProjection  : per-variable measurement values         → d_model

Temporal encoding design (inspired by Aurora, Price et al. 2024 Section B.4):
    Time is represented as hours since the Unix epoch (a single float).  The
    TemporalEmbedding module expands this scalar into log-spaced Fourier features
    spanning λ_min (10 min) to λ_max (1 year), then projects to d_model via a
    2-layer MLP.  This lets the model jointly discover sub-daily, weekly, monthly
    and seasonal patterns without hard-coding the relevant periods.

    Formula:  Emb(x) = [cos(2πx/λ_i), sin(2πx/λ_i)]  for i = 0 … D/2 − 1
              λ_i = exp(log λ_min + i·(log λ_max − log λ_min) / (D/2 − 1))
              x   = hours elapsed since 1970-01-01 00:00 UTC
"""

import math
import torch
import torch.nn as nn
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of meteorological variables used in this project.
# wind_speed / wind_direction replaced by wind_u / wind_v (compute_uv=True).
VARIABLE_NAMES = [
    "temperature",
    "pressure",
    "humidity",
    "wind_u",
    "wind_v",
    "precipitation",
]
NUM_VARIABLES = len(VARIABLE_NAMES)   # 6

# Names of the static station metadata fields (in the order used by encode_spatial_static).
# Aspects are stored as raw degrees; sin/cos encoding is applied inside encode_spatial_static.
SPATIAL_FEATURE_NAMES = [
    "swiss_easting",
    "swiss_northing",
    "ASPECT_2000M_SIGRATIO1",
    "ASPECT_10000M_SIGRATIO1",
    "station_height",
    "dem",
    "TPI_2000M",
    "SLOPE_2000M_SIGRATIO1",
    "SLOPE_10000M_SIGRATIO1",
    "SN_DERIVATIVE_2000M_SIGRATIO1",
    "SN_DERIVATIVE_10000M_SIGRATIO1",
    "WE_DERIVATIVE_2000M_SIGRATIO1",
]

# Output dimension of encode_spatial_static:
#   lat / lon        → 2 each (sin/cos)
#   aspect 2km/10km  → 2 each (sin/cos)
#   scalars          → 8 (station_height, dem, tpi, 2×slope, 2×sn_deriv, we_deriv)
SPATIAL_INPUT_DIM = 18

# Fourier feature dimension for TemporalEmbedding.
# Must be even: TEMPORAL_FOURIER_DIM // 2 wavelengths are used.
# 32 → 16 wavelengths spanning ~10 min to ~1 year.
TEMPORAL_FOURIER_DIM = 32


# ---------------------------------------------------------------------------
# Static encoding helpers  (preprocessing / CPU, not nn.Module)
# ---------------------------------------------------------------------------

def encode_spatial_static(station_info: dict) -> torch.Tensor:
    """
    Encode static station metadata into an 18-dimensional feature vector.

    Cyclic variables (latitude, longitude, aspects) are sin/cos encoded.
    Scalar topographic features are returned as-is and should be normalised
    (zero-mean, unit-variance across stations) before being passed to
    SpatialEmbedding.

    Args:
        station_info: dict-like with keys matching SPATIAL_FEATURE_NAMES
                      (e.g. a row from ds.stations_table).

    Returns:
        torch.Tensor of shape (18,), dtype float32.
    """
    def _sincos_deg(deg: float):
        rad = math.radians(deg)
        return math.sin(rad), math.cos(rad)

    def _sincos_rad(rad: float):
        return math.sin(rad), math.cos(rad)

    sin_lat, cos_lat   = _sincos_rad(math.radians(station_info["swiss_easting"]))
    sin_lon, cos_lon   = _sincos_rad(math.radians(station_info["swiss_northing"]))
    sin_asp2k,  cos_asp2k  = _sincos_deg(station_info["ASPECT_2000M_SIGRATIO1"])
    sin_asp10k, cos_asp10k = _sincos_deg(station_info["ASPECT_10000M_SIGRATIO1"])

    features = [
        sin_lat,    cos_lat,                                        # 2 — geographic position
        sin_lon,    cos_lon,                                        # 2
        sin_asp2k,  cos_asp2k,                                      # 2 — local aspect
        sin_asp10k, cos_asp10k,                                     # 2 — regional aspect
        float(station_info["station_height"]),                      # 1 — sensor elevation
        float(station_info["dem"]),                                 # 1 — DEM elevation
        float(station_info["TPI_2000M"]),                           # 1 — valley(−) vs ridge(+)
        float(station_info["SLOPE_2000M_SIGRATIO1"]),               # 1 — local slope steepness
        float(station_info["SLOPE_10000M_SIGRATIO1"]),              # 1 — regional slope steepness
        float(station_info["SN_DERIVATIVE_2000M_SIGRATIO1"]),       # 1 — S-N gradient local
        float(station_info["SN_DERIVATIVE_10000M_SIGRATIO1"]),      # 1 — S-N gradient regional
        float(station_info["WE_DERIVATIVE_2000M_SIGRATIO1"]),       # 1 — W-E gradient (Föhn)
    ]   # total: 18

    return torch.tensor(features, dtype=torch.float32)


def encode_temporal(ts: pd.Timestamp) -> float:
    """
    Encode a timestamp as hours elapsed since 1970-01-01 00:00 UTC.

    This scalar is the input to TemporalEmbedding, which expands it into
    multi-scale Fourier features spanning 10 min → 1 year wavelengths.

    Args:
        ts: pd.Timestamp — timezone-aware (UTC) or naive (assumed UTC).

    Returns:
        float: hours since Unix epoch.
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return float((ts - epoch).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# nn.Module embeddings
# ---------------------------------------------------------------------------

class SpatialEmbedding(nn.Module):
    """
    Projects static station metadata (18-dim) into d_model space via a 2-layer MLP.

    Usage note:
        Spatial features are fixed per station. Pre-encode all stations with
        encode_spatial_static(), normalise the scalar features across the
        station population, then call this module once to obtain a cached
        (N, d_model) tensor that is reused every forward pass.

    Args:
        d_model:    Transformer model dimension.
        input_dim:  Dimensionality of encode_spatial_static output (default 18).
    """

    def __init__(self, d_model: int, input_dim: int = SPATIAL_INPUT_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: (N, 18) or (B, N, 18)
                              Should be normalised before calling.
        Returns:
            (N, d_model) or (B, N, d_model)
        """
        return self.proj(spatial_features)


class TemporalEmbedding(nn.Module):
    """
    Multi-scale Fourier temporal embedding (inspired by Aurora, Price et al. 2024).

    Encodes time as *hours since the Unix epoch* using log-spaced sinusoidal
    features, then projects to d_model via a 2-layer MLP.

    Using `fourier_dim // 2` wavelengths log-spaced between λ_min and λ_max,
    the model can jointly learn sub-daily, weekly, monthly, and seasonal
    patterns without any hard-coded time cycles.

    Reference (Supplementary B.4):
        Emb(x) = [cos(2πx/λ_i), sin(2πx/λ_i)]  for i = 0..D/2-1
        λ_i log-spaced in [λ_min, λ_max]
        x = hours since 1970-01-01 00:00 UTC

    Args:
        d_model:     Transformer model dimension.
        fourier_dim: Total Fourier feature dimension (must be even; default 32).
                     Gives fourier_dim // 2 distinct wavelengths.
        lambda_min:  Shortest wavelength in hours (default 1/6 ≈ 10 min).
        lambda_max:  Longest  wavelength in hours (default 365.25 × 24 ≈ 1 year).
    """

    def __init__(
        self,
        d_model:     int   = 128,
        fourier_dim: int   = TEMPORAL_FOURIER_DIM,
        lambda_min:  float = 1.0 / 6.0,          # 10 minutes in hours
        lambda_max:  float = 365.25 * 24.0,       # 1 year in hours
    ):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even"
        self.fourier_dim = fourier_dim

        # Non-trainable log-spaced wavelengths: (fourier_dim // 2,)
        n_wl = fourier_dim // 2
        lambdas = torch.exp(
            torch.linspace(math.log(lambda_min), math.log(lambda_max), n_wl)
        )
        self.register_buffer("lambdas", lambdas)

        # MLP: fourier_dim → d_model → d_model
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _fourier(self, hours: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features for arbitrary-shape input.

        Args:
            hours: (...) float tensor of hours-since-epoch.
        Returns:
            (..., fourier_dim) float tensor.
        """
        x      = hours.unsqueeze(-1)                        # (..., 1)
        angles = 2.0 * math.pi * x / self.lambdas           # (..., n_wl)
        return torch.cat([torch.cos(angles),
                          torch.sin(angles)], dim=-1)       # (..., fourier_dim)

    def forward(self, hours: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hours: (B,) or (B, W) float tensor of hours since Unix epoch.
        Returns:
            (B, d_model) or (B, W, d_model) — same leading shape as input.
        """
        return self.proj(self._fourier(hours))              # (..., d_model)


class DeltaTimeEmbedding(nn.Module):
    """
    Learned embedding table for discrete forecast lead-time steps.

    step = 0  →  reconstruction  (target time == input time t)
    step = k  →  forecast k × 10 minutes ahead of t

    Used exclusively in the decoder. The encoder never uses this.

    Args:
        d_model:    Transformer model dimension.
        max_steps:  Maximum forecast horizon in 10-minute steps (default 36 = 6 h).
    """

    def __init__(self, d_model: int, max_steps: int = 36):
        super().__init__()
        self.embedding = nn.Embedding(max_steps + 1, d_model)  # +1 for step=0
        self.max_steps = max_steps

    def forward(self, delta_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_steps: (B,) integer tensor, values in [0, max_steps].
        Returns:
            (B, d_model)
        """
        assert delta_steps.max() <= self.max_steps, (
            f"delta_steps contains {delta_steps.max()} > max_steps={self.max_steps}"
        )
        return self.embedding(delta_steps)


class VariableProjection(nn.Module):
    """
    Projects raw meteorological measurements into d_model space.

    Each variable has its own scalar-to-d_model linear projection plus a
    learned type embedding encoding variable identity. Contributions from
    present variables are summed and averaged, making the token representation
    invariant to how many sensors a station has.

    Missing variables (mask == 0) are excluded so the model cannot confuse
    a true zero measurement with an absent sensor.

    Args:
        num_vars:  Number of meteorological variables (default 6).
        d_model:   Transformer model dimension.
    """

    def __init__(self, num_vars: int = NUM_VARIABLES, d_model: int = 256):
        super().__init__()
        self.num_vars = num_vars
        self.d_model  = d_model

        # One linear projection per variable: scalar → d_model
        self.var_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_vars)
        ])

        # Learned type embedding: encodes which variable this is
        self.var_type_embedding = nn.Embedding(num_vars, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, N, V)  Measurement values. 0.0 where sensor absent.
            mask: (B, N, V)  Float tensor: 1.0 present, 0.0 absent.

        Returns:
            (B, N, d_model)
        """
        B, N, V = x.shape
        device  = x.device

        out    = torch.zeros(B, N, self.d_model, device=device)
        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)   # (B, N, 1)

        for v in range(self.num_vars):
            present  = mask[..., v]                               # (B, N)
            value    = x[..., v].unsqueeze(-1)                    # (B, N, 1)
            proj     = self.var_projections[v](value)             # (B, N, d_model)
            idx      = torch.full((B, N), v, dtype=torch.long, device=device)
            type_emb = self.var_type_embedding(idx)               # (B, N, d_model)
            out     += present.unsqueeze(-1) * (proj + type_emb)

        return out / counts                                       # (B, N, d_model)


# ---------------------------------------------------------------------------
# Normalisation helper  (used during preprocessing, not training)
# ---------------------------------------------------------------------------

def compute_spatial_normalization(
    all_spatial_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean and std across all stations for normalisation.

    Args:
        all_spatial_features: (N_stations, 18) from encode_spatial_static.

    Returns:
        mean: (18,)
        std:  (18,) — clamped to avoid division by zero.
    """
    mean = all_spatial_features.mean(dim=0)
    std  = all_spatial_features.std(dim=0).clamp(min=1e-6)
    return mean, std
