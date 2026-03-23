"""
data/dataset.py

PyTorch Dataset for the Station-MAE project, built on top of PeakWeather.

Design:
    Each sample contains a multi-step input window and a single target snapshot:
        x           : (W, N, V)  — input window of W timesteps
        x_mask      : (W, N, V)  — sensor availability mask for input (1=present, 0=absent)
        x_hours     : (W,)       — hours-since-epoch per input step (→ TemporalEmbedding)
        y           : (N, V)     — target snapshot at t + delta_steps
        y_mask      : (N, V)     — sensor availability mask for target
        y_hours     : ()         — hours-since-epoch for target step (→ TemporalEmbedding)
        spatial     : (N, 18)    — normalised static station features (same for all samples)
        delta_steps : int        — forecast lead time in 10-min steps (0 = reconstruction)

    Where:
        W = window_size  (number of input timesteps)
        N = number of stations
        V = 6            (temperature, pressure, humidity, wind_u, wind_v, precipitation)

Variables:
    temperature, pressure, humidity, wind_u, wind_v, precipitation

Usage:
    from peakweather.dataset import PeakWeatherDataset
    from data.dataset import StationMAEDataset, load_peakweather

    ds_peak        = load_peakweather(root="path/to/data")
    train_dataset  = StationMAEDataset(ds_peak, window_size=12, delta_steps=6, split="train")
    val_dataset    = StationMAEDataset(ds_peak, window_size=12, delta_steps=6, split="val")
    loader         = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
"""

import math
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from peakweather.dataset import PeakWeatherDataset


# ---------------------------------------------------------------------------
# Temporal helper (mirrored from model/embeddings.py to avoid circular import)
# ---------------------------------------------------------------------------

def _hours_since_epoch(ts: pd.Timestamp) -> float:
    """
    Return hours elapsed since 1970-01-01 00:00 UTC.

    This scalar feeds TemporalEmbedding, which expands it into
    multi-scale Fourier features spanning 10 min → 1 year wavelengths.

    Args:
        ts: pd.Timestamp — UTC-aware or naive (assumed UTC).
    Returns:
        float: hours since Unix epoch.
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return float((ts - epoch).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# Constants — must stay in sync with model/embeddings.py
# ---------------------------------------------------------------------------

VARIABLE_NAMES = [
    "temperature",
    "pressure",
    "humidity",
    "wind_u",
    "wind_v",
    "precipitation",
]
NUM_VARIABLES = len(VARIABLE_NAMES)   # 6

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

# Train / val / test split by year
TRAIN_YEARS = list(range(2017, 2022))   # 2017–2021
VAL_YEARS   = [2022]
TEST_YEARS  = [2023, 2024]


# ---------------------------------------------------------------------------
# PeakWeather loader
# ---------------------------------------------------------------------------

def load_peakweather(root: str) -> PeakWeatherDataset:
    """
    Load PeakWeatherDataset with the correct parameters for this project.

    Loads meteo stations only, computes wind_u/wind_v from speed + direction,
    and attaches DEM topographic variables for spatial embeddings.
    wind_speed, wind_direction and wind_gust are excluded from the final
    variable set — only wind_u and wind_v are used by the model.

    Args:
        root: Path to the local PeakWeather data directory.

    Returns:
        PeakWeatherDataset instance.
    """
    return PeakWeatherDataset(
        root=root,
        parameters=[
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",       # required by compute_uv internally
            "wind_direction",   # required by compute_uv internally
            "precipitation",
        ],
        compute_uv=True,        # adds wind_u, wind_v columns
        station_type="meteo_station",
        freq="10min",
    )


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def build_spatial_features(
    ds: PeakWeatherDataset,
) -> tuple[torch.Tensor, dict]:
    """
    Encode and normalise static station metadata into a spatial feature tensor.

    Cyclic variables (lat, lon, aspects) are sin/cos encoded.
    Scalar features are normalised to zero-mean unit-variance across stations.

    Args:
        ds: PeakWeatherDataset instance (must have extended_topo_vars="DEM").

    Returns:
        spatial_norm : (N, 18) normalised float32 tensor
        stats        : {"mean": (18,), "std": (18,)} — keep for inference normalisation
    """
    stations = ds.stations_table
    rows = []

    for _, row in stations.iterrows():
        def sd(deg):
            r = math.radians(float(deg)); return math.sin(r), math.cos(r)

        sl, cl   = sd(row["swiss_easting"])
        slo, clo = sd(row["swiss_northing"])
        sa2, ca2 = sd(row["ASPECT_2000M_SIGRATIO1"])
        sa10,ca10= sd(row["ASPECT_10000M_SIGRATIO1"])

        rows.append([
            sl,   cl,                                           # lat  (2)
            slo,  clo,                                          # lon  (2)
            sa2,  ca2,                                          # aspect 2km  (2)
            sa10, ca10,                                         # aspect 10km (2)
            float(row["station_height"]),                       # elevation   (1)
            float(row["dem"]),                                  # DEM         (1)
            float(row["TPI_2000M"]),                            # TPI         (1)
            float(row["SLOPE_2000M_SIGRATIO1"]),                # slope 2km   (1)
            float(row["SLOPE_10000M_SIGRATIO1"]),               # slope 10km  (1)
            float(row["SN_DERIVATIVE_2000M_SIGRATIO1"]),        # S-N 2km     (1)
            float(row["SN_DERIVATIVE_10000M_SIGRATIO1"]),       # S-N 10km    (1)
            float(row["WE_DERIVATIVE_2000M_SIGRATIO1"]),        # W-E 2km     (1)
        ])   # 18 features total

    features = torch.tensor(rows, dtype=torch.float32)         # (N, 18)
    mean     = features.mean(dim=0)                            # (18,)
    std      = features.std(dim=0).clamp(min=1e-6)             # (18,)

    return (features - mean) / std, {"mean": mean, "std": std}


def build_observations(
    ds: PeakWeatherDataset,
) -> tuple[torch.Tensor, torch.Tensor, list[pd.Timestamp]]:
    """
    Extract the full observation array for the 6 selected variables.

    wind_speed, wind_direction and wind_gust are excluded here even if
    present in the underlying dataset.

    Returns:
        obs       : (T, N, V) float32  — NaN where sensor absent, value otherwise
        mask      : (T, N, V) float32  — 1.0 present, 0.0 absent
        timestamps: list of T pd.Timestamp objects
    """
    # Pull observations for the 6 model variables only
    raw = ds.get_observations(parameters=VARIABLE_NAMES)   # MultiIndex DataFrame

    # Shape: (T, N*V) → pivot to (T, N, V)
    stations   = ds.stations_table.index.tolist()
    T          = len(raw)
    N          = len(stations)
    V          = NUM_VARIABLES

    obs  = np.full((T, N, V), np.nan, dtype=np.float32)
    for v_idx, var in enumerate(VARIABLE_NAMES):
        for n_idx, stn in enumerate(stations):
            col = (stn, var)
            if col in raw.columns:
                obs[:, n_idx, v_idx] = raw[col].values.astype(np.float32)

    mask = (~np.isnan(obs)).astype(np.float32)
    obs  = np.nan_to_num(obs, nan=0.0)                     # replace NaN with 0

    timestamps = raw.index.tolist()

    return (
        torch.from_numpy(obs),     # (T, N, V)
        torch.from_numpy(mask),    # (T, N, V)
        timestamps,
    )


def normalise_observations(
    obs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Normalise each variable to zero-mean unit-variance using only present values.

    Args:
        obs:  (T, N, V) raw observations (0.0 where absent)
        mask: (T, N, V) presence mask

    Returns:
        obs_norm : (T, N, V) normalised observations
        stats    : {"mean": (V,), "std": (V,)} — save for denormalisation at inference
    """
    means = []
    stds  = []

    obs_norm = obs.clone()
    for v in range(NUM_VARIABLES):
        vals = obs[:, :, v][mask[:, :, v] == 1.0]
        m    = vals.mean()
        s    = vals.std().clamp(min=1e-6)
        obs_norm[:, :, v] = (obs[:, :, v] - m) / s
        # zero out absent sensors again after normalisation
        obs_norm[:, :, v] *= mask[:, :, v]
        means.append(m)
        stds.append(s)

    return obs_norm, {
        "mean": torch.stack(means),   # (V,)
        "std":  torch.stack(stds),    # (V,)
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StationMAEDataset(Dataset):
    """
    Sliding-window dataset over PeakWeather observations for Station-MAE.

    Each sample packages:
        x           (W, N, V)  normalised input window
        x_mask      (W, N, V)  sensor availability for input
        x_hours     (W,)       hours-since-epoch per input step (→ TemporalEmbedding)
        y           (N, V)     normalised target snapshot
        y_mask      (N, V)     sensor availability for target
        y_hours     ()         hours-since-epoch for target step (→ TemporalEmbedding)
        spatial     (N, 18)    normalised static station features
        delta_steps int        forecast lead time in 10-min steps

    The encoder receives the full window [W, N, V], attending across both
    stations and timesteps (multi-step mode). The decoder receives the
    target queries enriched with their spatial + temporal + Δt embeddings.

    Args:
        ds:           PeakWeatherDataset instance from load_peakweather().
        window_size:  Number of input timesteps W (default 12 = 2 hours).
        delta_steps:  Forecast horizon in 10-min steps (default 6 = 1 hour).
                      Use 0 for pure reconstruction (gap-filling).
        split:        One of "train", "val", "test". Selects years accordingly.
        obs_stats:    Optional pre-computed normalisation stats dict
                      {"mean": (V,), "std": (V,)}. If None, computed from
                      the split's data. Pass training stats to val/test splits.
    """

    def __init__(
        self,
        ds:          PeakWeatherDataset,
        window_size: int  = 12,
        delta_steps: int  = 6,
        split:       str  = "train",
        obs_stats:   dict = None,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val' or 'test', got '{split}'"
        assert delta_steps >= 0, "delta_steps must be >= 0"

        self.window_size  = window_size
        self.delta_steps  = delta_steps
        self.split        = split

        # ------------------------------------------------------------------
        # 1. Spatial features — static, shared across all samples
        # ------------------------------------------------------------------
        self.spatial, self.spatial_stats = build_spatial_features(ds)
        # (N, 18), normalised

        # ------------------------------------------------------------------
        # 2. Full observation tensor
        # ------------------------------------------------------------------
        obs_full, mask_full, timestamps_full = build_observations(ds)
        # obs_full:  (T, N, V)
        # mask_full: (T, N, V)

        # ------------------------------------------------------------------
        # 3. Split by year
        # ------------------------------------------------------------------
        year_map = {"train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS}
        keep_years = year_map[split]

        indices = [
            i for i, ts in enumerate(timestamps_full)
            if ts.year in keep_years
        ]
        assert len(indices) > 0, f"No timesteps found for split='{split}'"

        obs       = obs_full[indices]       # (T_split, N, V)
        mask      = mask_full[indices]      # (T_split, N, V)
        timestamps = [timestamps_full[i] for i in indices]

        # ------------------------------------------------------------------
        # 4. Normalise observations
        # ------------------------------------------------------------------
        if obs_stats is None:
            obs, self.obs_stats = normalise_observations(obs, mask)
        else:
            # Apply pre-computed stats (use training stats for val/test)
            self.obs_stats = obs_stats
            obs_norm = obs.clone()
            for v in range(NUM_VARIABLES):
                obs_norm[:, :, v] = (
                    (obs[:, :, v] - obs_stats["mean"][v]) / obs_stats["std"][v]
                ) * mask[:, :, v]
            obs = obs_norm

        self.obs        = obs         # (T_split, N, V)
        self.mask       = mask        # (T_split, N, V)
        self.timestamps = timestamps  # list of T_split pd.Timestamp

        # ------------------------------------------------------------------
        # 5. Pre-compute hours-since-epoch for every timestep
        #    TemporalEmbedding internally expands this scalar into
        #    multi-scale Fourier features (10 min → 1 year wavelengths).
        # ------------------------------------------------------------------
        self.hours = torch.tensor(
            [_hours_since_epoch(ts) for ts in timestamps],
            dtype=torch.float32,
        )   # (T_split,)

        # ------------------------------------------------------------------
        # 6. Valid window start indices
        #    A window [i : i+W] with target at i+W-1+delta_steps must fit
        #    within the split. We also skip windows that span year boundaries
        #    to avoid mixing context from different years.
        # ------------------------------------------------------------------
        T            = len(timestamps)
        max_start    = T - window_size - delta_steps
        self.indices = []

        for i in range(max_start + 1):
            win_start_year  = timestamps[i].year
            target_year     = timestamps[i + window_size - 1 + delta_steps].year
            if win_start_year == target_year:   # no year boundary crossing
                self.indices.append(i)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i   = self.indices[idx]
        W   = self.window_size
        dt  = self.delta_steps

        # Input window indices: [i, i+W)
        x       = self.obs[i : i + W]       # (W, N, V)
        x_mask  = self.mask[i : i + W]      # (W, N, V)
        x_hours = self.hours[i : i + W]     # (W,)  — hours-since-epoch per step

        # Target snapshot index: i + W - 1 + delta_steps
        t_idx   = i + W - 1 + dt
        y       = self.obs[t_idx]            # (N, V)
        y_mask  = self.mask[t_idx]           # (N, V)
        y_hours = self.hours[t_idx]          # ()   — scalar for target step

        return {
            "x":           x,                             # (W, N, V)
            "x_mask":      x_mask,                        # (W, N, V)
            "x_hours":     x_hours,                       # (W,)
            "y":           y,                             # (N, V)
            "y_mask":      y_mask,                        # (N, V)
            "y_hours":     y_hours,                       # ()
            "spatial":     self.spatial,                  # (N, 18)
            "delta_steps": torch.tensor(dt, dtype=torch.long),
        }