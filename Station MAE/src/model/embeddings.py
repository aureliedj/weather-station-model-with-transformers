"""
model/embeddings.py

Embedding modules for Station-MAE.

Each token is the sum of four independent components:

  p1  PositionalEmbedding  : 2-D station coordinates (easting, northing)   → d_model
                              Fourier-based: same philosophy as TemporalEmbedding
                              but over space rather than time.
  p2  StationEmbedding     : 13-D topographic / physical characteristics    → d_model
                              (aspect sin/cos, elevation, slope, TPI, gradients)
                              Plain 2-layer MLP — features are heterogeneous scalars.
  v   VariableProjection   : per-variable measurements                      → d_model
  t   TemporalEmbedding    : multi-scale Fourier time encoding               → d_model

Separating p1 from p2 lets the model learn pure location-based priors (e.g.
Föhn gaps in the Alps vs. Plateau stations) independently from the physical
characteristics of each site, and uses a principled continuous encoding for
the geographic position rather than treating coordinates as plain scalars.

Fourier encoding design (shared by PositionalEmbedding, TemporalEmbedding,
DeltaTimeEmbedding — inspired by Aurora, Price et al. 2024 Section B.4):

    Emb(x) = [cos(2πx/λ_i), sin(2πx/λ_i)]  for i = 0 … D/2 − 1
    λ_i = exp(log λ_min + i·(log λ_max − log λ_min) / (D/2 − 1))

    PositionalEmbedding : x = normalised easting or northing (each independently)
    TemporalEmbedding   : x = hours since 1970-01-01 00:00 UTC
    DeltaTimeEmbedding  : x = forecast lead-time in hours
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Layout of the spatial feature tensor (N, 15) produced by build_spatial_features:
#
#   Columns 0:2   — swiss_easting, swiss_northing  → consumed by PositionalEmbedding (p1)
#   Columns 2:15  — topographic characteristics     → consumed by StationEmbedding   (p2)
#       2,3   sin/cos aspect 2km
#       4,5   sin/cos aspect 10km
#       6     station_height
#       7     dem
#       8     TPI_2000M
#       9     SLOPE_2000M
#       10    SLOPE_10000M
#       11    SN_DERIVATIVE_2000M
#       12    SN_DERIVATIVE_10000M
#       13    WE_DERIVATIVE_2000M
#       14    WE_DERIVATIVE_10000M
#
# NOTE: Swiss LV95/LV03 coordinates are Cartesian (metres), not angles.
# After normalisation they live in roughly [-3, 3]; Fourier features in that
# domain cover spatial scales from ~10 km (local) to ~850 km (supra-national).
SPATIAL_INPUT_DIM  = 15   # total spatial features per station
POSITION_DIM       = 2    # columns 0:2  — easting, northing
STATION_CHAR_DIM   = 13   # columns 2:15 — topographic characteristics

# Fourier feature dimension for PositionalEmbedding (per coordinate).
# 16 features per coordinate (8 wavelengths), two coordinates → 32 total Fourier
# features fed to the projection MLP.
# λ_min=0.1 ≈ 8.5 km, λ_max=10 ≈ 850 km in normalised-coordinate space.
POSITION_FOURIER_DIM = 16

# Fourier feature dimension for TemporalEmbedding.
# Must be even: TEMPORAL_FOURIER_DIM // 2 wavelengths are used.
# 32 → 16 wavelengths (see TEMPORAL_WAVELENGTHS_H).
TEMPORAL_FOURIER_DIM = 32

# ── Temporal wavelengths, in hours ───────────────────────────────────────────
# EXPLICIT, not log-spaced. The previous version drew 16 wavelengths
# log-spaced from 1/6 h to 8766 h, which placed them at
#
#     0.167  0.344  0.710  1.466  3.025  6.245  12.889  26.605
#     54.9   113.4  234.0  482.9  996.8  2057.5 4246.9  8766.0
#
# The annual cycle landed EXACTLY on lambda_max, but the DIURNAL cycle — the
# dominant periodicity in surface weather, and the strongest predictable
# signal in temperature and humidity at the 0-6 h horizons this project
# forecasts — had no basis function. The nearest were 26.605 h (+10.9%) and
# 12.889 h (+7.4%), both incommensurate with 24 h, so no combination of the
# 32 features could express a 24-hour-periodic function. Measured by least
# squares against a pure diurnal signal:
#
#     fit of a  24 h cycle from the old basis:   R^2 = 0.005
#     fit of an annual cycle from the old basis: R^2 = 1.000
#
# The downstream MLP cannot rescue this: a 24 h-periodic embedding would
# require mapping f(t) and f(t+24h) — genuinely distinct points — to the same
# output across the whole domain. The model was therefore forced to
# re-derive time-of-day from the 12 h observation window on every forward
# pass, while being handed the annual cycle for free.
#
# Below, the periods are chosen for what surface weather actually does, in
# the spirit of ClimaX / GraphCast (which encode the physically meaningful
# cycles rather than a blind geometric sweep):
#
#   diurnal harmonics 24/k (k=1..6) — the diurnal curve is near-sinusoidal
#       with asymmetry (fast morning rise, slow evening fall); harmonics 1-3
#       carry most of the shape, 4-6 sharpen it. 12 h additionally captures
#       the semi-diurnal atmospheric pressure tide, which is a real signal
#       in the pressure channel.
#   synoptic -> monthly — weather systems are aperiodic, so these are a
#       log-spaced basis for slow drift rather than true cycles.
#   annual harmonics 8766/k (k=3,2,1) — seasonality, with the 2nd and 3rd
#       harmonics carrying the spring/autumn asymmetry a pure sinusoid misses.
#
# NOTE: nothing below ~4 h. Absolute sub-hourly phase ("22 minutes past the
# hour, measured from 1970") carries no meteorological signal. Relative
# position inside the input window is StepIndexEmbedding's job and is encoded
# exactly there; lead time is DeltaTimeEmbedding's. This module supplies
# absolute WHEN — time of day and time of year — and nothing else.
TEMPORAL_WAVELENGTHS_H = (
    # diurnal family: 24 / k for k = 1..6
    24.0, 12.0, 8.0, 6.0, 4.8, 4.0,
    # synoptic -> monthly drift (2 d, 3 d, 5 d, 8 d, 14 d, 30 d, 60 d)
    48.0, 72.0, 120.0, 192.0, 336.0, 720.0, 1440.0,
    # annual family: 8766 / k for k = 3, 2, 1
    2922.0, 4383.0, 8766.0,
)
assert 2 * len(TEMPORAL_WAVELENGTHS_H) == TEMPORAL_FOURIER_DIM

# ── v18 value embedding (PLR — Gorishniy et al., NeurIPS 2022) ───────────────
# Observations are per-station z-scored, so the value axis is in sigma units.
# lambda_min 0.25 resolves an eighth of a standard deviation; lambda_max 10
# spans the bulk of the distribution. Values outside that range do occur
# (precipitation reaches +26 sigma, wind +-11 sigma) which is why the RAW value
# is kept as a feature alongside the periodic code — it keeps the map monotone
# and injective everywhere.
VALUE_FOURIER_K   = 8                        # wavelengths
VALUE_FEATURE_DIM = 1 + 2 * VALUE_FOURIER_K  # raw value + cos/sin = 17
VALUE_LAMBDA_MIN  = 0.25                     # sigma
VALUE_LAMBDA_MAX  = 10.0                     # sigma
# Linear init gain. Fourier features are bounded, so this alone fixes the output
# scale — it does not depend on the observation distribution. GELU roughly
# halves the std, so a gain above 1 is needed; 2.2 was calibrated on the real
# observations to land every variable at std ~0.46-0.55 (v17 branch: 0.53,
# metadata branches: ~0.58).
VALUE_INIT_GAIN   = 2.2

# ── v18 value embedding, MLP variant (PLE-like — the DEFAULT v18 choice) ─────
# scalar -> Linear(1, H) -> GELU -> Linear(H, d).  Each hidden unit is a GELU
# threshold at x = -b_i/a_i, so the H units form a piecewise basis over the
# value axis.  Measured against the Fourier variant on held-out targets:
#
#     target              v17 linear   MLP   Fourier
#     |x|                     0.000   1.000    0.000
#     1[x > 1] threshold      0.499   0.950    0.000
#     x^2 / sin(3x)           0.000   1.000    1.000
#
# The MLP wins on KINKS and THRESHOLDS, which is what meteorological structure
# looks like (saturation, freezing, calm/gust).  Fourier only won on angular
# quantities — wind direction — which is not represented here.
VALUE_MLP_HIDDEN  = 32
# First layer: |a| = 1 exactly with random sign, b ~ U(-3, 3), so the H
# thresholds land at -b/a, spread UNIFORMLY over +-3 sigma.  PyTorch's default
# (a, b ~ U(-1,1)) makes -b/a Cauchy-distributed: measured 26.9/32 thresholds
# inside +-3 sigma versus 32/32 here, i.e. a sixth of the basis wasted.
VALUE_MLP_THRESH  = 3.0
# Second layer gain, /sqrt(H). Calibrated on the real observations against the
# SIGNAL std (the branch with its constant removed), not RMS: an earlier value
# of 0.65 matched RMS while 91% of that RMS was the uncentred GELU constant,
# which collapsed content fraction from 18.9% to 6.5%. Calibrate variance
# across tokens, never RMS.
VALUE_MLP_GAIN    = 1.47

# Init scale for var_absent_embedding — the "sensor missing" code. Matched to a
# typical PRESENT contribution (~0.5) so that "missing" is distinctive rather
# than merely small.
ABSENT_INIT_STD   = 0.5

# ── v21: static features INSIDE the variable block (Aurora-style) ────────────
# Aurora "incorporate[s] static variables (orography, land-sea mask, and
# soil-type mask) by treating them as extra surface-level variables", so terrain
# is scaled by the same per-variable mechanism as the weather instead of forming
# a separate additive branch whose magnitude is set independently.
#
# The GROUPING matters more than the idea here. Aurora has ~3 statics against
# many variables; this project has 15 statics against 6 weather variables, so
# one slot per static feature would give the weather only 6/21 of the block.
# Measured weather share of the token at init:
#
#     one slot per static feature   21 slots    9.6%   <- WORSE than today
#     position + topography          8 slots   25.1%   <- default
#     all 15 in one slot             7 slots   28.7%
#     (current v18, separate branches)         18.3%
#
# Column groups of the 15-D spatial vector. Mirrors the existing p1/p2 split:
# 0:2 easting/northing, 2:15 topography.
STATIC_SLOTS_DEFAULT = ((0, 2), (2, 15))

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

    The output layout matches SPATIAL_INPUT_DIM = 15:
        columns  0:2  → easting, northing  (consumed by PositionalEmbedding p1)
        columns 2:15  → characteristics    (consumed by StationEmbedding     p2)

    Aspect angles are sin/cos encoded (genuinely cyclic).
    All other features are plain scalars and should be normalised
    (zero-mean, unit-variance across stations) before being passed to the
    embedding modules.

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
        float(station_info["swiss_easting"]),                        # 1 — CH1903 easting  (m) **
        float(station_info["swiss_northing"]),                       # 1 — CH1903 northing (m) **
        # Aspect angles — sin/cos encoding is correct for cyclic compass directions
        sin_asp2k,  cos_asp2k,                                       # 2 — local aspect
        sin_asp10k, cos_asp10k,                                      # 2 — regional aspect
        # Scalar topographic features (all normalised by caller)
        float(station_info["station_height"]),                       # 1 — sensor elevation ***
        float(station_info["dem"]),                                  # 1 — DEM elevation    ***
        float(station_info["TPI_2000M"]),                            # 1 — valley(−) vs ridge(+) ***
        float(station_info["SLOPE_2000M_SIGRATIO1"]),                # 1 — local slope steepness ***
        float(station_info["SLOPE_10000M_SIGRATIO1"]),               # 1 — regional slope steepness ***
        float(station_info["SN_DERIVATIVE_2000M_SIGRATIO1"]),        # 1 — S-N gradient local   ***
        float(station_info["SN_DERIVATIVE_10000M_SIGRATIO1"]),       # 1 — S-N gradient regional    ***
        float(station_info["WE_DERIVATIVE_2000M_SIGRATIO1"]),        # 1 — W-E gradient local (Föhn) ***
        float(station_info["WE_DERIVATIVE_10000M_SIGRATIO1"]),       # 1 — W-E gradient regional ***
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

class PositionalEmbedding(nn.Module):
    """
    Fourier-based 2-D positional embedding for Swiss LV95 station coordinates (p1).

    After normalisation across the station population, easting and northing
    are each independently expanded into log-spaced Fourier features
    (POSITION_FOURIER_DIM features per coordinate).  The two sets of features
    are concatenated and projected to d_model via a 2-layer MLP.

    This mirrors TemporalEmbedding's design: a continuous scalar domain
    (normalised metres) mapped to sinusoids at multiple spatial scales,
    letting the model learn pure location-based priors (e.g. Föhn gaps,
    Alpine vs. Plateau regimes) without encoding them into the topographic
    characteristics embedding.

    Wavelengths span λ_min ≈ 8.5 km to λ_max ≈ 850 km in normalised-coordinate
    space (assuming σ_easting ≈ 85 km, σ_northing ≈ 55 km).

    The spatial tensor layout is:
        spatial[:, 0]  = normalised easting
        spatial[:, 1]  = normalised northing
        spatial[:, 2:] = topographic characteristics  (→ StationEmbedding)

    Args:
        d_model:      Transformer model dimension.
        fourier_dim:  Fourier features per coordinate (must be even; default 16).
                      Total input to the MLP = 2 × fourier_dim.
        lambda_min:   Shortest wavelength in normalised-coordinate units (default 0.1).
        lambda_max:   Longest  wavelength in normalised-coordinate units (default 10.0).
    """

    def __init__(
        self,
        d_model:     int   = 128,
        fourier_dim: int   = POSITION_FOURIER_DIM,
        lambda_min:  float = 0.1,    # ≈ 8.5 km (local station-spacing scale)
        lambda_max:  float = 10.0,   # ≈ 850 km (supra-national scale)
        dropout:     float = 0.0,
    ):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even"
        self.fourier_dim = fourier_dim

        # Non-trainable log-spaced wavelengths: (fourier_dim // 2,)
        n_wl = fourier_dim // 2
        lambdas = torch.exp(
            torch.linspace(math.log(lambda_min), math.log(lambda_max), n_wl)
        )
        self.register_buffer("lambdas", lambdas)  # (n_wl,)

        # MLP: (2 × fourier_dim) → d_model → d_model
        # Two coordinates × fourier_dim features each = 2 × fourier_dim input
        self.proj = nn.Sequential(
            nn.Linear(2 * fourier_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def _fourier_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand a scalar coordinate into Fourier features.

        Args:
            x: (...) float tensor of normalised coordinate values.
        Returns:
            (..., fourier_dim) — [cos, sin] interleaved over wavelengths.
        """
        x = x.unsqueeze(-1)                                  # (..., 1)
        angles = 2.0 * math.pi * x / self.lambdas            # (..., n_wl)
        return torch.cat([torch.cos(angles),
                          torch.sin(angles)], dim=-1)         # (..., fourier_dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (..., 2) float tensor — normalised [easting, northing].
                 Typically (N, 2) or (B, N, 2).
        Returns:
            (..., d_model)
        """
        east_feat  = self._fourier_1d(pos[..., 0])            # (..., fourier_dim)
        north_feat = self._fourier_1d(pos[..., 1])            # (..., fourier_dim)
        fourier    = torch.cat([east_feat, north_feat], dim=-1)  # (..., 2*fourier_dim)
        return self.proj(fourier)                              # (..., d_model)


class StationEmbedding(nn.Module):
    """
    Projects the 13-D station characteristic features into d_model space (p2).

    This covers the heterogeneous topographic and physical scalars:
        sin/cos aspect (2km & 10km), station_height, dem, TPI, slope (2km & 10km),
        S-N derivative (2km & 10km), W-E derivative (2km & 10km).

    A 2-layer MLP with an intermediate LayerNorm is used rather than Fourier
    encoding because the features are heterogeneous (different units, ranges,
    and semantics) and do not share a common 1-D continuous domain.

    Args:
        d_model:    Transformer model dimension.
        input_dim:  Number of station characteristics (default STATION_CHAR_DIM = 13).
    """

    def __init__(self, d_model: int, input_dim: int = STATION_CHAR_DIM,
                 dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),   # stabilises scale before second linear
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, char_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_features: (..., 13) normalised station characteristic features.
                           Typically (N, 13) or (B, N, 13).
        Returns:
            (..., d_model)
        """
        return self.proj(char_features)


class TemporalEmbedding(nn.Module):
    """
    Multi-scale Fourier temporal embedding (Aurora, Price et al. 2024 §B.4),
    over an EXPLICIT set of physically meaningful periods.

    Encodes absolute time as *hours since the Unix epoch*, then projects to
    d_model via a 2-layer MLP:

        Emb(x) = [cos(2πx/λ_i), sin(2πx/λ_i)]  for i = 0 … D/2−1
        x = hours since 1970-01-01 00:00 UTC

    The λ_i are TEMPORAL_WAVELENGTHS_H — diurnal harmonics, a synoptic band,
    and annual harmonics — not a log-spaced sweep. See the constant for the
    measurement that motivated it: a geometric spacing from 1/6 h to 1 year
    placed wavelengths at 12.889 h and 26.605 h and therefore could not
    represent the 24 h cycle at all (R² = 0.005), while landing exactly on
    the annual one.

    This module carries ABSOLUTE time only — time of day and time of year.
    Relative position inside the input window is StepIndexEmbedding's job and
    forecast lead time is DeltaTimeEmbedding's; neither is duplicated here.

    Args:
        d_model:     Transformer model dimension.
        fourier_dim: Total Fourier feature dimension (must be even; default 32).
                     Must equal 2 × len(wavelengths).
        wavelengths: Tuple of periods in hours. Defaults to
                     TEMPORAL_WAVELENGTHS_H.
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

        # EXPLICIT wavelengths (hours), not a log-spaced sweep — see
        # TEMPORAL_WAVELENGTHS_H for why. The old geometric spacing had no
        # 24 h component, so the diurnal cycle was unrepresentable (R^2=0.005
        # for a least-squares fit of a pure diurnal signal from the basis).
        wl = tuple(wavelengths) if wavelengths is not None else TEMPORAL_WAVELENGTHS_H
        assert 2 * len(wl) == fourier_dim, (
            f"fourier_dim={fourier_dim} needs {fourier_dim // 2} wavelengths, "
            f"got {len(wl)}")
        # float64: these are divisors of an ~4.9e5 input, see _fourier().
        self.register_buffer("lambdas", torch.tensor(wl, dtype=torch.float64))

        # MLP: fourier_dim → d_model → d_model
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),   # stabilises scale before second linear
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def _fourier(self, hours: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features for arbitrary-shape input.

        Computed in float64. This is NOT defensive over-engineering — the
        input is hours since 1970 (~4.9e5 in 2026), so even the 4 h wavelength
        gives an angle of 2*pi*4.9e5/4 ~= 7.7e5 radians. float32 carries a
        24-bit mantissa, so the absolute rounding error there is
        ~7.7e5 * 6e-8 ~= 0.046 rad, and it grows as lambda shrinks.

        With the OLD log-spaced bank (lambda_min = 1/6 h) the angle reached
        1.85e7 rad and the error 2.2 rad — a phase error of 126 degrees at
        the shortest wavelength, i.e. pure noise, with cos() off by 0.64 on a
        [-1, 1] range. Roughly a third of that bank was degraded. Those
        wavelengths are gone now (see TEMPORAL_WAVELENGTHS_H), but the input
        magnitude is unchanged, so the float64 computation is still required.

        float64 puts the error at ~1e-5 rad everywhere. cos/sin outputs are
        bounded in [-1, 1], so casting back afterwards is lossless in the
        relevant sense, and the tensor is tiny ((B, W) -> (B, W, 32)) so the
        cost is negligible against a ~48 GFLOP encoder pass.

        Args:
            hours: (...) float tensor of hours-since-epoch.
        Returns:
            (..., fourier_dim) float tensor, in the input's dtype.
        """
        x      = hours.double().unsqueeze(-1)               # (..., 1)  float64
        angles = 2.0 * math.pi * x / self.lambdas.double()  # (..., n_wl)
        feats  = torch.cat([torch.cos(angles),
                            torch.sin(angles)], dim=-1)     # (..., fourier_dim)
        return feats.to(hours.dtype)

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
        # lambda_min was 1/6 h and produced a PROVABLY CONSTANT feature pair:
        # fixed_grid horizons are multiples of 0.5 h, so every lead time was an
        # exact integer number of wavelengths (cos = 1, sin = 0 always).
        #
        # The first correction moved it to 1.0 h — the Nyquist wavelength of
        # the 0.5 h grid — which was HALF wrong: at exactly Nyquist the cosine
        # alternates +-1 but the SINE is identically zero (sin(pi*k) = 0 for
        # every integer k), so one of the two channels stayed dead. Caught by
        # tests/test_embedding_numerics.py on the first torch run.
        #
        # lambda_min must sit STRICTLY ABOVE Nyquist. 1.25 h gives phase steps
        # of 144 degrees on the 0.5 h grid — a 5-point cycle where both cos and
        # sin vary — and is verified alive on the 10-min grid too (random-delta
        # mode). Keep lambda_min > 2 * (delta_grid_stride * 10 min), never =.
        lambda_min:   float = 1.25,          # strictly above the 0.5 h-grid Nyquist
        lambda_max:   float = 8.0,           # 8 hours — buffer beyond 6 h max horizon
        step_size_h:  float = 1.0 / 6.0,    # each step = 10 minutes = 1/6 h
        dropout:      float = 0.0,
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
            nn.Dropout(dropout),
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


class StepIndexEmbedding(nn.Module):
    """
    Fourier embedding over integer step indices (0, 1, 2, …).

    Provides explicit within-window positional information that is consistent
    across the encoder and decoder:

      • Encoder  — input token at step w  (w = 0 … W-1) gets embedding[w].
      • Decoder  — query token for delta_step d (d = 0, 3, 6, …, 36) gets
                   embedding[d] using the SAME Fourier basis.

    This ensures that "input step 3 = 30 min in" and "output at delta_step 3 =
    30 min lead" map to the same sinusoidal features before the MLP projection,
    giving the model a principled shortcut to express 'copy + small correction'
    for short horizons.

    Motivation: TemporalEmbedding encodes *absolute* hours since 1970, so it
    naturally captures time-of-day, seasonality, and long-range trends but
    does not cleanly encode the *relative position within a 12 h window*.
    StepIndexEmbedding fills that gap with wavelengths tuned to the window
    scale (1 step ≈ 10 min to max_steps × 10 min), mirroring how ViT / BERT
    add a learned position embedding on top of content embeddings.

    Wavelengths: log-spaced from λ_min = 1 step (finest; resolves individual
    steps) to λ_max = max_steps steps (coarsest; covers the full window).
    Same design as TemporalEmbedding and DeltaTimeEmbedding but in step-count
    space rather than hours.

    Args:
        d_model:     Transformer model dimension.
        fourier_dim: Total Fourier feature dimension (must be even; default 32).
                     Gives fourier_dim // 2 distinct wavelengths.
        max_steps:   Upper bound on step index; sets the longest wavelength.
                     Should be ≥ W + max_delta so both encoder (0..W-1) and
                     decoder (0..max_delta) step indices fall in-range.
                     Default 512 is a generous buffer for typical configs.
        dropout:     Dropout rate on the MLP output (default 0.0).
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
        # Log-spaced wavelengths in step counts: 2.5 steps → max_steps.
        #
        # The lower bound was 1.0 and produced a PROVABLY CONSTANT feature
        # pair: the input is an INTEGER step index k, so at lambda = 1 every
        # input gives cos(2*pi*k) = 1 and sin(2*pi*k) = 0 — the same defect
        # class as the removed var_type_embedding.
        #
        # The first correction moved it to 2.0 (Nyquist), which was HALF
        # wrong: at exactly Nyquist cos(pi*k) alternates +-1 but sin(pi*k) is
        # identically ZERO — the sine channel stayed dead. Caught by
        # tests/test_embedding_numerics.py on the first torch run.
        #
        # lambda_min must sit STRICTLY ABOVE Nyquist: 2.5 steps gives phase
        # increments of 144 degrees — a 5-point cycle where both channels
        # vary. Verified: all 32 features alive over the used range 0..107.
        lambdas = torch.exp(
            torch.linspace(math.log(2.5), math.log(float(max_steps)), n_wl)
        )
        self.register_buffer("lambdas", lambdas)   # (n_wl,)

        # MLP: fourier_dim → d_model → d_model  (matches other embedding modules)
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def _fourier(self, steps: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features for arbitrary-shape step-index input.

        Args:
            steps: (...) integer or float tensor of step indices.
        Returns:
            (..., fourier_dim) float tensor.
        """
        x      = steps.float().unsqueeze(-1)                    # (..., 1)
        angles = 2.0 * math.pi * x / self.lambdas               # (..., n_wl)
        return torch.cat([torch.cos(angles),
                          torch.sin(angles)], dim=-1)           # (..., fourier_dim)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            steps: (W,)     integer step indices 0..W-1  — encoder usage, or
                   (B,)     integer delta steps           — decoder single-delta, or
                   (B, K)   integer delta steps           — decoder multi-delta.
        Returns:
            Same leading shape as steps, last dim = d_model.
        """
        return self.proj(self._fourier(steps))                  # (..., d_model)


class _CenteredMLP(nn.Module):
    """
    Linear -> GELU -> Linear, with the activation CENTRED so that e(0) = 0.

        h = GELU(a*x + b) - GELU(b)
        e = W2 @ h

    Why the subtraction. A plain GELU MLP is non-zero at x = 0: GELU(b) is a
    fixed vector, so W2 @ GELU(b) is added identically to EVERY token. That is
    magnitude without information — the same defect as the removed
    var_type_embedding, and it is invisible in an RMS check because RMS counts
    the constant. Measured on the real observations, uncentred:

        RMS 0.543 = constant 0.493 + signal 0.293   ->  content fraction 6.5%
        (v17 linear, for comparison: constant 0.000, signal 0.528, 18.9%)

    Subtracting GELU(b) removes the constant exactly and costs nothing in
    expressiveness: span{GELU(a_i x + b_i) - c_i} + bias = span{GELU(a_i x +
    b_i)} + bias, so the representable set is unchanged.
    """

    def __init__(self, l1: nn.Linear, l2: nn.Linear):
        super().__init__()
        self.l1, self.l2 = l1, l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.l1(x)) - F.gelu(self.l1.bias)
        return self.l2(h)


class _CenteredPLR(nn.Module):
    """
    GELU(lin(feat)) − GELU(lin(feat₀)): centres the PLR/fourier value
    embedding so that e(0) = 0 exactly — the same treatment _CenteredMLP
    gives the MLP mode, for the same reason.

    Without it the fourier branch carried a token-independent constant of
    1.11x the signal std (measured at init, tests/test_token_balance.py):
    at x = 0 the PLR features are [x=0, cos(0)=1 ... , sin(0)=0 ...], so
    GELU(lin(·)) of that fixed vector was added identically to EVERY token —
    the third instance of the uncentred-constant defect in this project,
    after var_type_embedding and the uncentred v18 MLP.

    feat₀ is independent of the wavelengths (cos(0) = 1 for any λ), so it is
    a fixed non-persistent buffer. Subtracting a constant does not change the
    variance across tokens, so VALUE_INIT_GAIN's calibration is unaffected,
    and the representable set is unchanged (a bias absorbs the shift).
    """

    def __init__(self, lin: nn.Linear, n_values: int, k: int = VALUE_FOURIER_K):
        super().__init__()
        self.lin = lin
        f0 = torch.tensor([0.0] + [1.0] * k + [0.0] * k).repeat(n_values)
        self.register_buffer("feat0", f0, persistent=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.lin(feat)) - F.gelu(self.lin(self.feat0))


class VariableProjection(nn.Module):
    """
    Projects raw meteorological measurements into d_model space.

    Per variable v, per station, per timestep:

        sensor present :  x_v * var_weights[v] + var_biases[v]
        sensor absent  :  var_absent_embedding[v]

    the V contributions are summed and divided by sqrt(V).

    Present sensors contribute their projected value; ABSENT sensors contribute
    a learned per-variable "absent" embedding (BERT-mask-style) instead of being
    dropped.  The model therefore sees an explicit, learnable signal for
    "sensor missing" and cannot confuse it with a true zero measurement.

    The V contributions are divided by a CONSTANT (not by the per-station
    present count).  Dividing by the count made the embedding magnitude of the
    same physical value depend on how many unrelated sensors the station
    carries (1/6 at a full station vs 1/2 at a wind-only one) — a
    station-dependent gain the network had to undo.  (cf. ClimaX, Nguyen et
    al. 2023, which uses attention aggregation partly to avoid this.)

    SCALE (see "Rebalance observation-token initialization")
    -------------------------------------------------------
    This branch is one of FIVE summed into the encoder token; the other four
    (position, station topography, absolute time, step index) are all
    MLP-with-LayerNorm and land at RMS ~0.58 per token.  For the token to
    encode the weather at all, this branch has to arrive at a comparable
    scale.  With the sqrt(V) divisor:

        obs_d = (1/sqrt(V)) * sum_v x_v * w_vd
        Var(obs_d) = (1/V) * V * sigma_x^2 * sigma_w^2 = sigma_x^2 * sigma_w^2
        => std(obs) = sigma_x * sigma_w                    -- V CANCELS

    so the branch scale is independent of the variable count: adding a seventh
    variable needs no re-tuning.  Dividing by V instead of sqrt(V) does NOT
    have that property and over-suppresses by sqrt(V).

    With sigma_x ~ 0.92 (per-station normalised observations) and
    var_weights ~ U(-1,1) (sigma_w = 0.577), std(obs) ~ 0.53 — parity with the
    other four branches.

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

    def __init__(
        self,
        num_vars:        int  = NUM_VARIABLES,
        d_model:         int  = 256,
        value_embedding: str  = "linear",       # "linear" (v17) | "fourier" (v18)
        wind_pair:       "tuple | None" = None, # e.g. (3, 4) -> encode u,v jointly
        static_slots:    "tuple | None" = None, # v21: column groups of `spatial`
        static_dim:      int  = 15,
    ):
        """
        value_embedding
            "linear"  — v17: e_v = x_v * var_weights[v] + var_biases[v].
                        Rank 1 per variable, no nonlinearity. DEFAULT, unchanged.
            "mlp"     — v18: e_v = Linear_H->d( GELU( Linear_1->H(x_v) ) ).
                        A piecewise basis over the value axis; best on kinks
                        and thresholds, which is what meteorological structure
                        looks like. The recommended v18 setting.
            "fourier" — v18: PLR (Gorishniy et al., NeurIPS 2022):
                        e_v = GELU( Linear( [x_v, cos(2*pi*x_v/lam), sin(...)] ) ).
                        The raw value is kept alongside the periodic code so the
                        map stays monotone and injective outside the wavelength
                        range — real observations reach +26 sigma (precipitation)
                        and +-11 sigma (wind), well beyond lambda_max.

        wind_pair
            Indices of variables encoded by ONE shared map instead of two
            independent ones. Intended for (wind_u, wind_v), which are the two
            components of a single vector: a joint encoder can form speed and
            direction, which no per-variable map can. Verified on the data:
            the wind_u and wind_v masks are identical in 100.000% of
            (sample, station) pairs, so there is no partial-availability case.
        """
        super().__init__()
        assert value_embedding in ("linear", "mlp", "fourier"), value_embedding
        self.num_vars        = num_vars
        self.d_model         = d_model
        self.value_embedding = value_embedding

        # ── Slots: usually one per variable; a wind pair occupies one slot ──
        # Order is deterministic: variables in index order, with the pair
        # collapsed at the position of its first member.
        self.wind_pair = tuple(wind_pair) if wind_pair else None
        slots: "list[tuple[int, ...]]" = []
        if self.wind_pair:
            assert len(self.wind_pair) == 2 and all(0 <= i < num_vars for i in self.wind_pair)
            skip = set(self.wind_pair)
            for v in range(num_vars):
                if v == min(self.wind_pair):
                    slots.append(self.wind_pair)
                elif v not in skip:
                    slots.append((v,))
        else:
            slots = [(v,) for v in range(num_vars)]
        self.slots     = slots
        self.num_slots = len(slots)
        # One index buffer per slot, so forward() never builds a tensor.
        # persistent=False: derivable from the config, not model state.
        for si, members in enumerate(slots):
            self.register_buffer(f"_slot_{si}",
                                 torch.tensor(members, dtype=torch.long),
                                 persistent=False)

        # Batched per-variable linear: var_weights[v] projects x[..., v] → d_model.
        # This tensor stores num_vars independent Linear(1, d_model) maps; it is
        # NOT one (num_vars → d_model) layer.
        #
        # It used to be initialised with xavier_uniform_, which reads the tensor
        # SHAPE (fan_in=d_model, fan_out=num_vars) and returns bound
        # sqrt(6/(V+d)) = 0.124 → std 0.072 at d=384.  The Linear(1, d_model)
        # modules this replaced have fan_in = 1 → bound 1.0 → std 0.577: eight
        # times larger, for exactly the same function.  Xavier's
        # variance-preservation argument does not apply to a shape that does not
        # describe the map being computed.  Measured effect of the old init: the
        # observation branch arrived at RMS 0.027 against ~0.58 for each of the
        # four metadata branches it is summed with.
        # Is every slot a single variable? Then the whole branch is a BATCHED
        # per-variable map and needs no Python loop — see forward().
        self.batched = all(len(s) == 1 for s in slots)

        # ── v21: static slots — extra "variables" that never change in time ──
        # Their contribution is computed ONCE on (N, static_dim) and broadcast
        # over the B*W leading dimension, so they cost nothing per timestep.
        self.static_slots = tuple(tuple(g) for g in static_slots) if static_slots else None
        if self.static_slots:
            per = 1 if value_embedding == "linear" else VALUE_FEATURE_DIM
            mods = []
            for lo, hi in self.static_slots:
                n = hi - lo
                if value_embedding == "mlp":
                    H  = VALUE_MLP_HIDDEN
                    l1 = nn.Linear(n, H)
                    with torch.no_grad():
                        nn.init.uniform_(l1.weight, -1.0, 1.0)
                        nn.init.uniform_(l1.bias, -VALUE_MLP_THRESH, VALUE_MLP_THRESH)
                    l2 = nn.Linear(H, d_model)
                    b  = VALUE_MLP_GAIN / math.sqrt(H)
                    nn.init.uniform_(l2.weight, -b, b); nn.init.zeros_(l2.bias)
                    mods.append(_CenteredMLP(l1, l2))
                else:
                    lin = nn.Linear(n * (1 if value_embedding == "linear" else per), d_model)
                    if value_embedding == "fourier":
                        b = VALUE_INIT_GAIN / math.sqrt(lin.in_features)
                        nn.init.uniform_(lin.weight, -b, b); nn.init.zeros_(lin.bias)
                        # centred like the slot maps — same uncentred-GELU defect
                        mods.append(_CenteredPLR(lin, n))
                    else:
                        nn.init.uniform_(lin.weight, -1.0, 1.0); nn.init.zeros_(lin.bias)
                        mods.append(lin)
            self.static_maps = nn.ModuleList(mods)
        self.total_slots = self.num_slots + (len(self.static_slots) if self.static_slots else 0)

        if value_embedding == "linear" and self.wind_pair is None:
            self.var_weights = nn.Parameter(torch.empty(num_vars, d_model))
            self.var_biases  = nn.Parameter(torch.zeros(num_vars, d_model))
            nn.init.uniform_(self.var_weights, -1.0, 1.0)   # Linear(1, d) scale
        elif value_embedding == "mlp" and self.batched:
            # ── Batched MLP: V independent (1 -> H -> d) maps, stored as one
            # set of tensors and applied with two fused ops.
            #
            # PERFORMANCE. The first implementation used an nn.ModuleList and a
            # Python loop over slots, with getattr(self, f"_slot_{si}") to fetch
            # the index buffer. That is ~30 kernel launches per forward instead
            # of 2, and — worse under --compile — dynamic attribute access by
            # formatted string and ModuleList indexing inside a loop are classic
            # graph-break causes, so the compiled graph was repeatedly falling
            # back to eager. Measured effect on the v18 run: ~20 min/epoch.
            # The FLOPs are unchanged (3.3 GMAC either way, against ~48 GFLOP
            # for the encoder); the cost was launch overhead and recompilation.
            H = VALUE_MLP_HIDDEN
            self.mlp_w1 = nn.Parameter(torch.empty(num_vars, H))
            self.mlp_b1 = nn.Parameter(torch.empty(num_vars, H))
            self.mlp_w2 = nn.Parameter(torch.empty(num_vars, H, d_model))
            self.mlp_b2 = nn.Parameter(torch.zeros(num_vars, d_model))
            with torch.no_grad():
                # |a| = 1 with random sign, b ~ U(-T, T): thresholds at -b/a
                # spread uniformly over the value range.
                self.mlp_w1.copy_(torch.randint(0, 2, self.mlp_w1.shape).float() * 2 - 1)
                nn.init.uniform_(self.mlp_b1, -VALUE_MLP_THRESH, VALUE_MLP_THRESH)
                bound = VALUE_MLP_GAIN / math.sqrt(H)
                nn.init.uniform_(self.mlp_w2, -bound, bound)
        else:
            # One map per SLOT. For "linear" the map is Linear(|slot| -> d);
            # for "fourier" it is Linear(|slot| * VALUE_FEATURE_DIM -> d) + GELU.
            maps = []
            for s in slots:
                n = len(s)
                if value_embedding == "linear":
                    maps.append(nn.Linear(n, d_model))
                elif value_embedding == "mlp":
                    H   = VALUE_MLP_HIDDEN
                    l1  = nn.Linear(n, H)
                    # |a| = 1 with random sign, b ~ U(-3, 3): thresholds at
                    # -b/a spread uniformly over the value range instead of
                    # PyTorch's Cauchy-distributed default.
                    with torch.no_grad():
                        l1.weight.copy_(torch.randint(0, 2, l1.weight.shape).float() * 2 - 1)
                        nn.init.uniform_(l1.bias, -VALUE_MLP_THRESH, VALUE_MLP_THRESH)
                    l2 = nn.Linear(H, d_model)
                    bound = VALUE_MLP_GAIN / math.sqrt(H)
                    nn.init.uniform_(l2.weight, -bound, bound)
                    nn.init.zeros_(l2.bias)
                    maps.append(_CenteredMLP(l1, l2))
                else:                                     # "fourier"
                    lin   = nn.Linear(n * VALUE_FEATURE_DIM, d_model)
                    # Fourier features are bounded, so this init alone fixes the
                    # output scale — it does not depend on the observations'
                    # spread. GELU roughly halves the std, hence a gain above 1.
                    bound = VALUE_INIT_GAIN / math.sqrt(lin.in_features)
                    nn.init.uniform_(lin.weight, -bound, bound)
                    nn.init.zeros_(lin.bias)
                    # centred: e(0) = 0 exactly, see _CenteredPLR
                    maps.append(_CenteredPLR(lin, n))
            self.slot_maps = nn.ModuleList(maps)
            if value_embedding == "fourier":
                self.register_buffer(
                    "value_lambdas",
                    torch.exp(torch.linspace(math.log(VALUE_LAMBDA_MIN),
                                             math.log(VALUE_LAMBDA_MAX),
                                             VALUE_FOURIER_K)),
                )

        # NOTE: ``var_type_embedding`` (one learned vector per variable, added to
        # BOTH the present and the absent branch) has been REMOVED.  It was
        # provably redundant: with
        #     present = x_v*w_v + b_v + t_v      absent = a_v + t_v
        # the substitution b'_v = b_v + t_v, a'_v = a_v + t_v gives an identical
        # function, so b, t and a were three per-variable vectors spanning two
        # degrees of freedom.  Variable identity is already carried by the
        # DIRECTION of var_weights[v] in the present branch and by
        # var_absent_embedding[v] in the absent branch.
        #
        # It also happened to be the only parameter in the model left at
        # nn.Embedding's default N(0,1) — every other token-level parameter here
        # uses trunc_normal_(std=0.02).  Summed over V=6 and divided by V, it
        # produced a token-independent offset of RMS sqrt(V)/V = 0.41, i.e. large
        # magnitude carrying exactly zero information.
        #
        # Checkpoints containing var_proj.var_type_embedding.weight are migrated
        # automatically and exactly — see _load_from_state_dict below.

        # Learned per-variable ABSENT embedding, used in place of the value
        # projection when a sensor is missing (mask == 0).  Replaces the
        # encoder-level sensor_mask_token, whose fill value was silently
        # cancelled by the mask re-application here.
        #
        # Initialised at the project convention (0.02) rather than at zero.  With
        # both var_biases and var_absent_embedding starting at zero, the two
        # codes were bit-identical at step 0:
        #     sensor reads zero :  0*w_v + b_v  =  0
        #     sensor absent     :          a_v  =  0
        # i.e. the exact confusion this parameter exists to prevent.  Because
        # normalisation is per-station, x = 0 means "reads its own long-run
        # mean" — the MODAL value, not an edge case — so the collision sat on
        # the most common reading.
        # SCALE (corrected): initialise at the scale of a TYPICAL PRESENT
        # contribution (~0.5), not at 0.02. A present sensor contributes
        # std ~0.53; at 0.02 the absent code was 26x weaker, so "missing"
        # was a barely-visible signal rather than a distinctive one. Those
        # are different things: the code should be DISTINCTIVE in direction
        # at comparable magnitude, not small.
        self.var_absent_embedding = nn.Parameter(
            torch.zeros(self.num_slots, d_model))
        nn.init.trunc_normal_(self.var_absent_embedding, std=ABSENT_INIT_STD)

    # ------------------------------------------------------------------
    # Value features
    # ------------------------------------------------------------------

    def _value_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        (..., n) values -> (..., n * VALUE_FEATURE_DIM) PLR features.

        Per value: [x, cos(2*pi*x/lam_i), sin(2*pi*x/lam_i)] for the
        VALUE_FOURIER_K log-spaced wavelengths. Keeping the raw x means the
        map is monotone and injective for values outside [lambda_min,
        lambda_max] — necessary here because precipitation reaches +26 sigma.
        """
        ang  = 2.0 * math.pi * x.unsqueeze(-1) / self.value_lambdas   # (..., n, K)
        feat = torch.cat([x.unsqueeze(-1), ang.cos(), ang.sin()], dim=-1)  # (..., n, 1+2K)
        return feat.flatten(-2)                                        # (..., n*(1+2K))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                static: "torch.Tensor | None" = None) -> torch.Tensor:
        """
        Args:
            x:    (B, N, V)  Measurement values. 0.0 where sensor absent.
            mask: (B, N, V)  Float tensor: 1.0 present, 0.0 absent.

        Returns:
            (B, N, d_model)
        """
        # ── v17 fast path: byte-identical to the released implementation ───
        if self.value_embedding == "linear" and self.wind_pair is None:
            proj   = x.unsqueeze(-1) * self.var_weights + self.var_biases
            m      = mask.unsqueeze(-1)
            mixed  = proj * m + self.var_absent_embedding * (1.0 - m)
            # sqrt(S), NOT S. Summing S roughly-independent contributions scales
            # the variance by S and the std by sqrt(S); dividing by S
            # over-suppresses by a further sqrt(S). sqrt(S) also makes
            # std(obs) = sigma_x * sigma_w, independent of the slot count.
            return self._finish(mixed.sum(dim=-2), static)

        # ── v18 batched MLP: no Python loop, two fused ops ──────────────────
        if self.value_embedding == "mlp" and self.batched:
            m = mask.unsqueeze(-1)                                # (B, N, V, 1)
            # h = GELU(a*x + b) - GELU(b), centred so e(0) = 0 exactly and the
            # branch carries no token-independent constant.
            h = F.gelu(x.unsqueeze(-1) * self.mlp_w1 + self.mlp_b1) \
                - F.gelu(self.mlp_b1)                             # (B, N, V, H)
            e = torch.einsum("...vh,vhd->...vd", h, self.mlp_w2) + self.mlp_b2
            e = e * m + self.var_absent_embedding * (1.0 - m)
            return self._finish(e.sum(dim=-2), static)

        # ── v18 general path: one map per SLOT (used only for wind pairing) ─
        # A slot is one variable, or the wind pair. A slot counts as present
        # only when every member is present (u and v always agree in this
        # dataset, but the rule is enforced rather than assumed).
        acc = None
        for si, members in enumerate(self.slots):
            # Buffer, not new_tensor(): this is the training hot loop, and
            # building an index tensor per call forces a host->device copy.
            idx  = getattr(self, f"_slot_{si}")
            xv   = x.index_select(-1, idx)                      # (B, N, |slot|)
            mv   = mask.index_select(-1, idx).amin(-1, keepdim=True)   # (B, N, 1)
            feat = self._value_features(xv) if self.value_embedding == "fourier" else xv
            e    = self.slot_maps[si](feat)     # GELU lives inside the Sequential
            e = e * mv + self.var_absent_embedding[si] * (1.0 - mv)
            acc = e if acc is None else acc + e

        # Divide by sqrt(number of SLOTS): the same reasoning as sqrt(V), but
        # a paired wind slot is one contribution, not two.
        return self._finish(acc, static)

    # ------------------------------------------------------------------
    # Static slots + the shared divisor
    # ------------------------------------------------------------------

    def _finish(self, acc: torch.Tensor,
                static: "torch.Tensor | None") -> torch.Tensor:
        """
        Add the static slots (v21) and divide by sqrt(total slot count).

        The static contribution depends only on the station, so it is computed
        ONCE on (N, static_dim) and broadcast over the leading B*W dimension —
        it costs nothing per timestep, exactly like the separate pos_emb /
        station_emb branches it replaces.
        """
        if self.static_slots is not None:
            if static is None:
                raise ValueError(
                    "VariableProjection was built with static_slots but forward() "
                    "received static=None. The encoder must pass `spatial`.")
            if static.dim() == 3:          # (B, N, S) -> statics do not vary in B
                static = static[0]
            for si, (lo, hi) in enumerate(self.static_slots):
                sv = static[..., lo:hi]                          # (N, n)
                if self.value_embedding == "fourier":
                    sv = self._value_features(sv)
                acc = acc + self.static_maps[si](sv)             # broadcasts over B*W
        return acc / math.sqrt(self.total_slots)

    # ------------------------------------------------------------------
    # Checkpoint migration
    # ------------------------------------------------------------------

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Load pre-rebalance checkpoints EXACTLY, folding away var_type_embedding.

        Old form (divisor V, with a type vector t_v):
            out_old = (1/V)      * sum_v [ (x_v*w_v + b_v + t_v)*m + (a_v + t_v)*(1-m) ]
        New form (divisor sqrt(V), no type vector):
            out_new = (1/sqrt(V))* sum_v [ (x_v*w'_v + b'_v)*m + a'_v*(1-m) ]

        Setting
            w' = w / sqrt(V)      b' = (b + t) / sqrt(V)      a' = (a + t) / sqrt(V)
        makes out_new == out_old identically, for every x and every mask.  So an
        existing v15 checkpoint keeps its exact behaviour and stays evaluable;
        only FRESH runs get the rebalanced initialisation.

        This is deliberately loud about what it did: a silent strict=False is
        what invalidated every test number from v9 to v13 in this project.
        """
        t_key = prefix + "var_type_embedding.weight"
        if t_key in state_dict and not (self.value_embedding == "linear"
                                        and self.wind_pair is None):
            # A pre-rebalance checkpoint cannot be folded into a v18 layout:
            # the parameter set is different (slot_maps vs var_weights) and the
            # value map is no longer linear, so no algebraic rewrite exists.
            raise RuntimeError(
                f"VariableProjection at '{prefix}': this checkpoint predates the "
                f"observation-token rebalance (it carries var_type_embedding), "
                f"but the model is configured with value_embedding="
                f"'{self.value_embedding}' / wind_pair={self.wind_pair}. "
                f"Old checkpoints can only be loaded into the v17 layout "
                f"(value_embedding='linear', wind_pair=None). Evaluate them "
                f"with that configuration.")
        if t_key in state_dict:
            t     = state_dict.pop(t_key)
            scale = math.sqrt(self.num_vars)
            for name, add_t in (("var_weights", False),
                                ("var_biases", True),
                                ("var_absent_embedding", True)):
                k = prefix + name
                if k in state_dict:
                    v = state_dict[k]
                    state_dict[k] = ((v + t) if add_t else v) / scale
            warnings.warn(
                f"VariableProjection: migrated a pre-rebalance checkpoint at "
                f"'{prefix}' — var_type_embedding folded into var_biases / "
                f"var_absent_embedding and all three tensors scaled by "
                f"1/sqrt({self.num_vars}). The loaded model is numerically "
                f"IDENTICAL to the one that was trained.",
                RuntimeWarning, stacklevel=2,
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


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
