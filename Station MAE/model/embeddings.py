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
NUM_VARIABLES = len(VARIABLE_NAMES)   # 6  — all variables used as *input*

# Variables the model is asked to *predict*.
# Precipitation is kept as an input signal but excluded from the loss / output head
# because it is intermittent and heavily zero-inflated, making MSE a poor objective.
TARGET_VARIABLE_NAMES = [v for v in VARIABLE_NAMES if v != "precipitation"]
NUM_TARGET_VARIABLES  = len(TARGET_VARIABLE_NAMES)   # 5

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
    "WE_DERIVATIVE_10000M_SIGRATIO1",   # regional W-E gradient (Föhn / valley flow)
]

# Output dimension of encode_spatial_static:
#   swiss_easting / swiss_northing → 1 each (plain scalar, normalised)
#   aspect 2km / 10km              → 2 each (sin/cos — genuinely cyclic angles)
#   scalars                        → 9 (station_height, dem, tpi, 2×slope, 2×sn_deriv, 2×we_deriv)
# Total: 2 + 4 + 9 = 15
#
# NOTE: Swiss LV95/LV03 coordinates are Cartesian (metres), not angles.
# Sin/cos encoding would be meaningless at that scale (~2.6M m easting → ~45k rad).
# Simple zero-mean unit-variance normalisation is the correct treatment.
SPATIAL_INPUT_DIM = 15

# Fourier feature dimension for TemporalEmbedding.
# Must be even: TEMPORAL_FOURIER_DIM // 2 wavelengths are used.
# 32 → 16 wavelengths spanning ~10 min to ~1 year.
TEMPORAL_FOURIER_DIM = 32

# Fourier feature dimension for DeltaTimeEmbedding.
# Smaller than TEMPORAL_FOURIER_DIM — lead-time range (0–6 h) is much narrower
# than the temporal range (10 min → 1 year), so fewer basis functions suffice.
# 16 → 8 wavelengths spanning 10 min to 8 h.
DELTA_FOURIER_DIM = 16


# ---------------------------------------------------------------------------
# Static encoding helpers  (preprocessing / CPU, not nn.Module)
# ---------------------------------------------------------------------------

def encode_spatial_static(station_info: dict) -> torch.Tensor:
    """
    Encode static station metadata into a 15-dimensional feature vector.

    Cyclic variables (latitude, longitude, aspects) are sin/cos encoded.
    Scalar topographic features are returned as-is and should be normalised
    (zero-mean, unit-variance across stations) before being passed to
    SpatialEmbedding.

    Args:
        station_info: dict-like with keys matching SPATIAL_FEATURE_NAMES
                      (e.g. a row from ds.stations_table).

    Returns:
        torch.Tensor of shape (15,), dtype float32.
    """
    def _sincos_deg(deg: float):
        rad = math.radians(float(deg))
        return math.sin(rad), math.cos(rad)

    # Aspect angles are genuinely cyclic (compass direction → sin/cos correct)
    sin_asp2k,  cos_asp2k  = _sincos_deg(station_info["ASPECT_2000M_SIGRATIO1"])
    sin_asp10k, cos_asp10k = _sincos_deg(station_info["ASPECT_10000M_SIGRATIO1"])

    features = [
        # Geographic position — Swiss LV95/LV03 Cartesian coordinates in metres.
        # These are NOT angles: sin/cos would be meaningless at ~2.6 M metre scale.
        # Kept as plain scalars; normalised (zero-mean, unit-var) across the station
        # population by the caller (build_spatial_features / compute_spatial_normalization).
        float(station_info["swiss_easting"]),                        # 1 — CH1903 easting  (m)
        float(station_info["swiss_northing"]),                       # 1 — CH1903 northing (m)
        # Aspect angles — sin/cos encoding is correct for cyclic compass directions
        sin_asp2k,  cos_asp2k,                                       # 2 — local aspect
        sin_asp10k, cos_asp10k,                                      # 2 — regional aspect
        # Scalar topographic features (all normalised by caller)
        float(station_info["station_height"]),                       # 1 — sensor elevation
        float(station_info["dem"]),                                  # 1 — DEM elevation
        float(station_info["TPI_2000M"]),                            # 1 — valley(−) vs ridge(+)
        float(station_info["SLOPE_2000M_SIGRATIO1"]),                # 1 — local slope steepness
        float(station_info["SLOPE_10000M_SIGRATIO1"]),               # 1 — regional slope steepness
        float(station_info["SN_DERIVATIVE_2000M_SIGRATIO1"]),        # 1 — S-N gradient local
        float(station_info["SN_DERIVATIVE_10000M_SIGRATIO1"]),       # 1 — S-N gradient regional
        float(station_info["WE_DERIVATIVE_2000M_SIGRATIO1"]),        # 1 — W-E gradient local (Föhn)
        float(station_info["WE_DERIVATIVE_10000M_SIGRATIO1"]),       # 1 — W-E gradient regional
    ]   # total: 2 + 4 + 9 = 15

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
    Projects static station metadata (15-dim) into d_model space via a 2-layer MLP.

    Usage note:
        Spatial features are fixed per station. Pre-encode all stations with
        encode_spatial_static(), normalise the scalar features across the
        station population, then call this module once to obtain a cached
        (N, d_model) tensor that is reused every forward pass.

    Args:
        d_model:    Transformer model dimension.
        input_dim:  Dimensionality of encode_spatial_static output (default 15).
    """

    def __init__(self, d_model: int, input_dim: int = SPATIAL_INPUT_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),   # stabilises scale before second linear
            nn.Linear(d_model, d_model),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: (N, 15) or (B, N, 15)
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
            nn.LayerNorm(d_model),   # stabilises scale before second linear
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
    Fourier-based lead-time embedding for forecast horizons.

    Replaces the previous discrete lookup table with a continuous sinusoidal
    encoding that is consistent with TemporalEmbedding's design.

    Each integer delta_steps is first converted to hours
    (hours = delta_steps × step_size_h), then encoded with log-spaced Fourier
    features spanning [lambda_min, lambda_max], and projected to d_model
    via a 2-layer MLP.

    Advantages over a lookup table:
      - Generalises to unseen horizons (extrapolation beyond max_steps).
      - Consistent encoding philosophy with TemporalEmbedding.
      - No upper bound on delta_steps baked into the architecture.

    step = 0  →  reconstruction  (target time == input time t)
    step = k  →  forecast k × 10 minutes ahead of t

    Used exclusively in the decoder. The encoder never uses this.

    Args:
        d_model:      Transformer model dimension.
        fourier_dim:  Total Fourier feature dimension (must be even; default 16).
                      Gives fourier_dim // 2 distinct wavelengths.
        lambda_min:   Shortest wavelength in hours (default 10 min = 1/6 h).
        lambda_max:   Longest  wavelength in hours (default 8 h — beyond 6 h horizon).
        step_size_h:  Duration of one step in hours (default 1/6 for 10-min steps).
    """

    def __init__(
        self,
        d_model:      int   = 128,
        fourier_dim:  int   = DELTA_FOURIER_DIM,
        lambda_min:   float = 1.0 / 6.0,    # 10 minutes in hours
        lambda_max:   float = 8.0,           # 8 hours — buffer beyond 6 h max horizon
        step_size_h:  float = 1.0 / 6.0,    # each step = 10 minutes = 1/6 h
    ):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even"
        self.fourier_dim  = fourier_dim
        self.step_size_h  = step_size_h

        # Non-trainable log-spaced wavelengths: (fourier_dim // 2,)
        n_wl = fourier_dim // 2
        lambdas = torch.exp(
            torch.linspace(math.log(lambda_min), math.log(lambda_max), n_wl)
        )
        self.register_buffer("lambdas", lambdas)

        # MLP: fourier_dim → d_model → d_model  (matches TemporalEmbedding structure)
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, delta_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_steps: (B,) integer tensor of lead-time steps (≥ 0).
        Returns:
            (B, d_model)
        """
        # Convert discrete steps to continuous hours
        hours  = delta_steps.float() * self.step_size_h          # (B,)
        x      = hours.unsqueeze(-1)                              # (B, 1)
        angles = 2.0 * math.pi * x / self.lambdas                # (B, n_wl)
        fourier = torch.cat([torch.cos(angles),
                             torch.sin(angles)], dim=-1)          # (B, fourier_dim)
        return self.proj(fourier)                                  # (B, d_model)


class VariableProjection(nn.Module):
    """
    Projects raw meteorological measurements into d_model space.

    Each variable has its own scalar-to-d_model linear projection plus a
    learned type embedding encoding variable identity.  Contributions from
    present variables are summed and averaged, making the token representation
    invariant to how many sensors a station has.

    Missing variables (mask == 0) are excluded so the model cannot confuse
    a true zero measurement with an absent sensor.

    Implementation note:
        Variables are projected in a single vectorised operation:
            proj[b, n, v] = x[b, n, v] * var_weights[v] + var_biases[v]
        This replaces the previous Python loop over V separate Linear modules,
        reducing the number of kernel dispatches from V to 1.

        NOTE: parameter names changed from ``var_projections.{v}.{weight,bias}``
        to ``var_weights`` / ``var_biases`` — checkpoints saved before this
        change are not directly compatible.

    Args:
        num_vars:  Number of meteorological variables (default 6).
        d_model:   Transformer model dimension.
    """

    def __init__(self, num_vars: int = NUM_VARIABLES, d_model: int = 256):
        super().__init__()
        self.num_vars = num_vars
        self.d_model  = d_model

        # Batched per-variable linear: var_weights[v] projects x[..., v] → d_model
        # Equivalent to num_vars independent Linear(1, d_model) modules but vectorised.
        self.var_weights = nn.Parameter(torch.empty(num_vars, d_model))
        self.var_biases  = nn.Parameter(torch.zeros(num_vars, d_model))
        nn.init.xavier_uniform_(self.var_weights)

        # Learned type embedding: one d_model vector per variable identity
        self.var_type_embedding = nn.Embedding(num_vars, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, N, V)  Measurement values. 0.0 where sensor absent.
            mask: (B, N, V)  Float tensor: 1.0 present, 0.0 absent.

        Returns:
            (B, N, d_model)
        """
        # Per-variable linear: (B, N, V, 1) * (V, d_model) → (B, N, V, d_model)
        proj = x.unsqueeze(-1) * self.var_weights + self.var_biases   # (B, N, V, d_model)

        # Add variable-identity embedding; var_type_embedding.weight is (V, d_model)
        proj = proj + self.var_type_embedding.weight                   # (B, N, V, d_model)

        # Zero out absent sensors and average over present variables
        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)        # (B, N, 1)
        out    = (proj * mask.unsqueeze(-1)).sum(dim=-2) / counts      # (B, N, d_model)

        return out


# ---------------------------------------------------------------------------
# Normalisation helper  (used during preprocessing, not training)
# ---------------------------------------------------------------------------

def compute_spatial_normalization(
    all_spatial_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean and std across all stations for normalisation.

    Args:
        all_spatial_features: (N_stations, 15) from encode_spatial_static.

    Returns:
        mean: (15,)
        std:  (15,) — clamped to avoid division by zero.
    """
    mean = all_spatial_features.mean(dim=0)
    std  = all_spatial_features.std(dim=0).clamp(min=1e-6)
    return mean, std
