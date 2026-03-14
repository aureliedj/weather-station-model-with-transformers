"""
model/embeddings.py

All embedding modules for Station-MAE:
  - SpatialEmbedding     : static station metadata (lat/lon/topo) → d_model
  - TemporalEmbedding    : cyclic timestamp features              → d_model
  - DeltaTimeEmbedding   : forecast lead-time steps               → d_model  [decoder only]
  - VariableProjection   : per-variable measurement values        → d_model

Spatial features are static per station and should be precomputed and cached.
Temporal features are shared across all stations at a given timestep.
DeltaTimeEmbedding is only used in the decoder.
"""

import math
import torch
import torch.nn as nn
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of meteorological variables used in this project.
# wind_speed, wind_direction, wind_gust excluded:
#   - wind_speed / wind_direction replaced by wind_u / wind_v (compute_uv=True)
#   - wind_gust dropped by design choice
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
# Aspects are stored as raw degrees here; sin/cos encoding is done inside encode_spatial_static.
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
#   lat        → 2 (sin/cos of radians)
#   lon        → 2 (sin/cos of radians)
#   aspect_2k  → 2 (sin/cos of degrees)
#   aspect_10k → 2 (sin/cos of degrees)
#   scalars    → 8 (station_height, dem, tpi, slope_2k, slope_10k, sn_2k, sn_10k, we_2k)
SPATIAL_INPUT_DIM = 18

# Output dimension of encode_temporal:
#   sin/cos 24h cycle  → 2
#   sin/cos 365d cycle → 2
TEMPORAL_INPUT_DIM = 4


# ---------------------------------------------------------------------------
# Static encoding helpers  (CPU / preprocessing time, not nn.Module)
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

    # Geographic position
    sin_lat, cos_lat   = _sincos_rad(math.radians(station_info["swiss_easting"]))
    sin_lon, cos_lon   = _sincos_rad(math.radians(station_info["swiss_northing"]))

    # Aspect — cyclic in [0, 360] degrees
    sin_asp2k,  cos_asp2k  = _sincos_deg(station_info["ASPECT_2000M_SIGRATIO1"])
    sin_asp10k, cos_asp10k = _sincos_deg(station_info["ASPECT_10000M_SIGRATIO1"])

    # Scalar topographic features (normalise externally)
    features = [
        sin_lat,    cos_lat,                                        # 2 — geographic position
        sin_lon,    cos_lon,                                        # 2
        sin_asp2k,  cos_asp2k,                                      # 2 — local aspect
        sin_asp10k, cos_asp10k,                                     # 2 — regional aspect
        float(station_info["station_height"]),                      # 1 — sensor elevation
        float(station_info["dem"]),                                 # 1 — DEM elevation
        float(station_info["TPI_2000M"]),                           # 1 — valley(-) vs ridge(+)
        float(station_info["SLOPE_2000M_SIGRATIO1"]),               # 1 — local slope steepness
        float(station_info["SLOPE_10000M_SIGRATIO1"]),              # 1 — regional slope steepness
        float(station_info["SN_DERIVATIVE_2000M_SIGRATIO1"]),       # 1 — S-N gradient local
        float(station_info["SN_DERIVATIVE_10000M_SIGRATIO1"]),      # 1 — S-N gradient regional
        float(station_info["WE_DERIVATIVE_2000M_SIGRATIO1"]),       # 1 — W-E gradient (Föhn)
    ]   # total: 18

    return torch.tensor(features, dtype=torch.float32)


def encode_temporal(ts: pd.Timestamp) -> torch.Tensor:
    """
    Encode a UTC-aware pandas Timestamp into a 4-dimensional cyclic feature vector.

    Two cycles are captured:
      - 24h  diurnal cycle  (sin/cos) — captures hour-of-day including day/night
      - 365d seasonal cycle (sin/cos) — captures time of year

    The minute component is included so 10-minute resolution timestamps
    within the same hour produce distinct encodings.

    Args:
        ts: pd.Timestamp with timezone info (UTC expected).

    Returns:
        torch.Tensor of shape (4,), dtype float32.
    """
    hour_of_day = ts.hour + ts.minute / 60.0   # fractional hour in [0, 24)
    day_of_year = float(ts.day_of_year)

    two_pi = 2.0 * math.pi

    features = [
        math.sin(two_pi * hour_of_day / 24.0),     # diurnal
        math.cos(two_pi * hour_of_day / 24.0),
        math.sin(two_pi * day_of_year / 365.25),    # seasonal
        math.cos(two_pi * day_of_year / 365.25),
    ]   # total: 4

    return torch.tensor(features, dtype=torch.float32)


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
                              Should be normalised (zero-mean, unit-variance
                              for scalar features) before calling.
        Returns:
            (N, d_model) or (B, N, d_model)
        """
        return self.proj(spatial_features)


class TemporalEmbedding(nn.Module):
    """
    Projects cyclic timestamp features (4-dim) into d_model space via a 2-layer MLP.

    At a given timestep all stations share the same temporal context, so the
    4-dim vector is computed once per timestep and broadcast across stations
    before being passed here.

    Args:
        d_model:    Transformer model dimension.
        input_dim:  Dimensionality of encode_temporal output (default 4).
    """

    def __init__(self, d_model: int, input_dim: int = TEMPORAL_INPUT_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: (B, 4) or (B, W, N, 4)
        Returns:
            (B, d_model) or (B, W, N, d_model)
        """
        return self.proj(temporal_features)


class DeltaTimeEmbedding(nn.Module):
    """
    Learned embedding table for discrete forecast lead-time steps.

    step = 0  →  reconstruction  (target time == input time t)
    step = k  →  forecast k * 10 minutes ahead of t

    Used exclusively in the decoder. The encoder never uses this.

    Args:
        d_model:    Transformer model dimension.
        max_steps:  Maximum forecast horizon in 10-minute steps (default 36 = 6h).
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