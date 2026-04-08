"""
data/dataset.py

PyTorch Dataset for the Station-MAE project, built on top of PeakWeather.

Design:
    Each sample contains a multi-step input window and one or more target snapshots:
        x           : (W, N, V)        — input window of W timesteps
        x_mask      : (W, N, V)        — sensor availability mask for input
        x_hours     : (W,)             — hours-since-epoch per input step
        y           : (N, V)           — target snapshot  [single-delta mode]
                      (K, N, V)        — K target snapshots [multi-delta mode]
        y_mask      : (N, V) or (K, N, V)
        y_hours     : ()   or (K,)
        spatial     : (N, 15)          — normalised static station features
        delta_steps : ()   or (K,)     — lead-time(s) in 10-min steps

Multi-delta mode
----------------
Set ``num_delta_per_sample > 1`` to return K randomly chosen lead-times per
sample in [1, max_delta_steps].  The encoder runs once per sample while the
decoder runs K times — amortising the expensive O(L²) encoder attention cost.

Data caching
------------
The first call to ``build_observations`` reads every row from the underlying
PeakWeather HDF/parquet files and can take several minutes.  Pass ``cache_dir``
to save the result as a single ``.pt`` file; subsequent runs load in seconds.

    StationMAEDataset(..., cache_dir="/path/to/cache")

Delete the cache file to force a rebuild after the underlying data changes.

Variables:
    temperature, pressure, humidity, wind_u, wind_v, precipitation
"""

import math
import os
import time

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from peakweather.dataset import PeakWeatherDataset


# ---------------------------------------------------------------------------
# Temporal helper
# ---------------------------------------------------------------------------

def _hours_since_epoch(ts: pd.Timestamp) -> float:
    """Return hours elapsed since 1970-01-01 00:00 UTC."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return float((ts - epoch).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# Constants
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
    "WE_DERIVATIVE_10000M_SIGRATIO1",
]

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
            "wind_speed",
            "wind_direction",
            "precipitation",
        ],
        compute_uv=True,
        station_type="meteo_station",
        imputation_method=None,
        freq="10min",
    )


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def build_spatial_features(
    ds: PeakWeatherDataset,
) -> tuple[torch.Tensor, dict]:
    """
    Encode and normalise static station metadata into a (N, 15) feature tensor.

    Swiss LV95 coordinates are Cartesian metres, normalised as plain scalars.
    Aspect angles receive sin/cos encoding (genuinely cyclic compass values).

    Returns:
        spatial_norm : (N, 15) normalised float32 tensor
        stats        : {"mean": (15,), "std": (15,)}
    """
    stations = ds.stations_table
    rows = []

    def _sincos_deg(deg):
        r = math.radians(float(deg))
        return math.sin(r), math.cos(r)

    for _, row in stations.iterrows():
        sa2,  ca2  = _sincos_deg(row["ASPECT_2000M_SIGRATIO1"])
        sa10, ca10 = _sincos_deg(row["ASPECT_10000M_SIGRATIO1"])
        rows.append([
            float(row["swiss_easting"]),
            float(row["swiss_northing"]),
            sa2,  ca2,
            sa10, ca10,
            float(row["station_height"]),
            float(row["dem"]),
            float(row["TPI_2000M"]),
            float(row["SLOPE_2000M_SIGRATIO1"]),
            float(row["SLOPE_10000M_SIGRATIO1"]),
            float(row["SN_DERIVATIVE_2000M_SIGRATIO1"]),
            float(row["SN_DERIVATIVE_10000M_SIGRATIO1"]),
            float(row["WE_DERIVATIVE_2000M_SIGRATIO1"]),
            float(row["WE_DERIVATIVE_10000M_SIGRATIO1"]),
        ])   # 2 + 4 + 9 = 15 features total

    features = torch.tensor(rows, dtype=torch.float32)   # (N, 15)
    mean     = features.mean(dim=0)
    std      = features.std(dim=0).clamp(min=1e-6)
    return (features - mean) / std, {"mean": mean, "std": std}


def build_observations(
    ds: PeakWeatherDataset,
) -> tuple[torch.Tensor, torch.Tensor, list[pd.Timestamp]]:
    """
    Extract the full (T, N, V) observation array for the 6 model variables.

    wind_speed, wind_direction and wind_gust are excluded; only wind_u/wind_v.

    Returns:
        obs        : (T, N, V) float32 — 0.0 where sensor absent
        mask       : (T, N, V) float32 — 1.0 present, 0.0 absent
        timestamps : list of T pd.Timestamp objects
    """
    raw      = ds.get_observations(parameters=VARIABLE_NAMES)
    stations = ds.stations_table.index.tolist()
    T        = len(raw)
    N        = len(stations)
    V        = NUM_VARIABLES

    obs = np.full((T, N, V), np.nan, dtype=np.float32)
    for v_idx, var in enumerate(VARIABLE_NAMES):
        for n_idx, stn in enumerate(stations):
            col = (stn, var)
            if col in raw.columns:
                obs[:, n_idx, v_idx] = raw[col].values.astype(np.float32)

    mask = (~np.isnan(obs)).astype(np.float32)
    obs  = np.nan_to_num(obs, nan=0.0)

    return (
        torch.from_numpy(obs),
        torch.from_numpy(mask),
        raw.index.tolist(),
    )


def normalise_observations(
    obs:  torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Normalise each variable to zero-mean unit-variance using only present values.

    Args:
        obs:  (T, N, V) raw observations (0.0 where absent)
        mask: (T, N, V) presence mask

    Returns:
        obs_norm : (T, N, V) normalised
        stats    : {"mean": (V,), "std": (V,)}
    """
    means, stds = [], []
    obs_norm = obs.clone()
    for v in range(NUM_VARIABLES):
        vals = obs[:, :, v][mask[:, :, v] == 1.0]
        m    = vals.mean()
        s    = vals.std().clamp(min=1e-6)
        obs_norm[:, :, v] = (obs[:, :, v] - m) / s
        obs_norm[:, :, v] *= mask[:, :, v]
        means.append(m)
        stds.append(s)
    return obs_norm, {"mean": torch.stack(means), "std": torch.stack(stds)}


def compute_obs_stats(
    ds:          PeakWeatherDataset,
    obs_full:    "torch.Tensor | None" = None,
    mask_full:   "torch.Tensor | None" = None,
    timestamps:  "list | None"         = None,
    train_years: "list[int] | None"    = None,
) -> dict:
    """
    Compute normalisation statistics from the training split.

    Two calling conventions::

        compute_obs_stats(ds)
            → builds observations from scratch (slow first time)

        compute_obs_stats(ds, obs_full, mask_full, timestamps)
            → reuses pre-built tensors (fast, use after cache load)

    Args:
        train_years: Years to use as the training split for computing stats.
                     Defaults to TRAIN_YEARS (2017–2021). Pass a subset list
                     (e.g. [2020, 2021]) to match a subset training run.

    Returns:
        dict with "mean": (V,) and "std": (V,) float32 tensors.
    """
    if obs_full is None or mask_full is None or timestamps is None:
        obs_full, mask_full, timestamps = build_observations(ds)

    years      = train_years if train_years is not None else TRAIN_YEARS
    train_idx  = [i for i, ts in enumerate(timestamps) if ts.year in years]
    obs_train  = obs_full[train_idx]
    mask_train = mask_full[train_idx]
    _, stats   = normalise_observations(obs_train, mask_train)
    return stats


# ---------------------------------------------------------------------------
# Disk cache for preprocessed tensors
# ---------------------------------------------------------------------------

_CACHE_FILENAME = "peakweather_obs_cache.pt"


def _load_or_build_cache(
    ds:        PeakWeatherDataset,
    cache_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, list, torch.Tensor, dict]:
    """
    Load preprocessed observation + spatial tensors from a cache file,
    building and saving them on the first call.

    Cache: ``{cache_dir}/peakweather_obs_cache.pt``

    Stores: obs_full (T,N,V), mask_full (T,N,V), timestamps list,
            spatial (N,15), spatial_stats dict.

    Delete the file to force a rebuild (e.g. after updating the raw data).

    Returns:
        obs_full, mask_full, timestamps, spatial, spatial_stats
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _CACHE_FILENAME)

    if os.path.exists(cache_path):
        t0 = time.time()
        print(f"[Cache] Loading from {cache_path} …", flush=True)
        payload = torch.load(cache_path, weights_only=False)
        elapsed = time.time() - t0
        obs_shape = tuple(payload["obs_full"].shape)
        print(f"[Cache] Loaded in {elapsed:.1f}s  (obs {obs_shape})", flush=True)
        return (
            payload["obs_full"],
            payload["mask_full"],
            payload["timestamps"],
            payload["spatial"],
            payload["spatial_stats"],
        )

    # ── First run: build from raw data and save ─────────────────────────
    print("[Cache] First run — building tensors from raw data "
          "(may take several minutes)…", flush=True)
    t0 = time.time()
    obs_full, mask_full, timestamps = build_observations(ds)
    spatial, spatial_stats          = build_spatial_features(ds)
    elapsed = time.time() - t0
    print(f"[Cache] Built in {elapsed:.1f}s  — saving to {cache_path}", flush=True)

    torch.save({
        "obs_full":      obs_full,
        "mask_full":     mask_full,
        "timestamps":    timestamps,
        "spatial":       spatial,
        "spatial_stats": spatial_stats,
    }, cache_path)
    print("[Cache] Saved.", flush=True)

    return obs_full, mask_full, timestamps, spatial, spatial_stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StationMAEDataset(Dataset):
    """
    Sliding-window dataset over PeakWeather observations for Station-MAE.

    Single-delta mode  (num_delta_per_sample == 1)
    -----------------------------------------------
    Each sample packages a single (x, y) pair with one lead-time.  The
    lead-time is either fixed at ``delta_steps`` or drawn randomly from
    ``[1, max_delta_steps]`` when those two values differ.

    Multi-delta mode  (num_delta_per_sample == K > 1)
    --------------------------------------------------
    K distinct lead-times are drawn per sample from ``[1, max_delta_steps]``.
    Batch shapes gain a K dimension:
        y / y_mask   → (K, N, V)
        y_hours      → (K,)
        delta_steps  → (K,)  long tensor

    The training loop calls ``model.forward_multi_delta()``, which runs the
    encoder once and the decoder K times — amortising O(L²) attention cost.

    Data loading optimisations
    --------------------------
    * ``share_memory_()`` is called on obs, mask, and hours tensors so that
      all DataLoader worker processes share the same memory pages without
      copying, keeping per-worker RAM usage negligible.
    * Pass ``cache_dir`` to persist the expensive ``build_observations()``
      result across Python sessions — subsequent starts load in < 5 s.
    * Use ``persistent_workers=True`` and ``prefetch_factor=4`` in your
      DataLoader to keep workers alive between epochs and pipeline I/O.

    Args:
        ds:                   PeakWeatherDataset from ``load_peakweather()``.
        window_size:          Input timesteps W (default 288 = 48 h at 10 min).
        delta_steps:          Fixed lead-time in 10-min steps (default 18 = 3 h).
                              Used as-is when max_delta_steps is None.
        split:                "train", "val", or "test".
        obs_stats:            Normalisation stats dict {"mean": (V,), "std": (V,)}.
                              If None, computed from training years of obs_full.
                              Always pass training-split stats to val/test splits
                              to prevent data leakage.
        num_delta_per_sample: K lead-times returned per sample (default 1).
                              When K > 1, max_delta_steps must be provided.
        max_delta_steps:      Upper bound for randomly sampled lead-times.
                              Required when num_delta_per_sample > 1.
        cache_dir:            Directory for the preprocessed tensor cache.
                              First run builds and saves; subsequent runs load
                              from disk in seconds.
    """

    def __init__(
        self,
        ds:                   PeakWeatherDataset,
        window_size:          int  = 288,
        delta_steps:          int  = 18,
        split:                str  = "train",
        obs_stats:            "dict | None" = None,
        num_delta_per_sample: int  = 1,
        max_delta_steps:      "int | None" = None,
        cache_dir:            "str | None" = None,
        train_years:          "list[int] | None" = None,
        shared_memory:        bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val' or 'test', got '{split}'"
        assert delta_steps >= 0, "delta_steps must be >= 0"
        assert num_delta_per_sample >= 1, "num_delta_per_sample must be >= 1"
        if num_delta_per_sample > 1:
            assert max_delta_steps is not None, \
                "max_delta_steps must be set when num_delta_per_sample > 1"

        self.window_size          = window_size
        self.delta_steps          = delta_steps
        self.split                = split
        self.num_delta_per_sample = num_delta_per_sample
        # Effective upper bound on lead-time — governs valid index calculation
        # and random sampling in __getitem__
        self.max_delta_steps = max_delta_steps if max_delta_steps is not None \
                               else delta_steps

        # ------------------------------------------------------------------
        # 1. Load (or build + cache) raw tensors
        # ------------------------------------------------------------------
        if cache_dir is not None:
            obs_full, mask_full, timestamps_full, spatial, spatial_stats = \
                _load_or_build_cache(ds, cache_dir)
        else:
            obs_full, mask_full, timestamps_full = build_observations(ds)
            spatial, spatial_stats               = build_spatial_features(ds)

        self.spatial       = spatial          # (N, 15)
        self.spatial_stats = spatial_stats

        # ------------------------------------------------------------------
        # 1b. Log1p-transform precipitation before normalisation
        #
        # Precipitation is zero-inflated and heavy-tailed (most values are 0,
        # rare events can be very large).  A direct z-score normalisation on the
        # raw mm values produces a highly skewed distribution that is hard for
        # the variable projection to embed meaningfully.
        #
        # log1p(x) = log(1 + x) compresses the tail while mapping:
        #   0   mm → 0.0        (exact: no-rain is preserved as zero)
        #   1   mm → 0.693
        #   10  mm → 2.398
        #   100 mm → 4.615
        #
        # Applied here to obs_full BEFORE stats are computed, so the
        # normalisation statistics are in log-space.  The mask is respected:
        # missing values are already 0 in obs_full and log1p(0)=0, so the
        # zero-filling for absent sensors is preserved exactly.
        # ------------------------------------------------------------------
        _PRECIP_IDX = VARIABLE_NAMES.index("precipitation")   # = 5
        obs_full = obs_full.clone()   # don't modify the cached tensor in-place
        obs_full[:, :, _PRECIP_IDX] = torch.log1p(
            obs_full[:, :, _PRECIP_IDX].clamp(min=0.0)
        )

        # ------------------------------------------------------------------
        # 2. Normalisation statistics — always from TRAINING years only
        # ------------------------------------------------------------------
        # train_years controls which years are used for normalisation stats.
        # For a subset run (e.g. train_years=[2020,2021]) this ensures stats
        # are computed from the SAME data the model was trained on, avoiding
        # scale drift if you later run a full (2017–2021) training job.
        effective_train_years = train_years if train_years is not None else TRAIN_YEARS
        if obs_stats is None:
            self.obs_stats = compute_obs_stats(
                ds,
                obs_full     = obs_full,
                mask_full    = mask_full,
                timestamps   = timestamps_full,
                train_years  = effective_train_years,
            )
        else:
            self.obs_stats = obs_stats

        # ------------------------------------------------------------------
        # 3. Slice to split years
        # ------------------------------------------------------------------
        year_map   = {"train": effective_train_years, "val": VAL_YEARS, "test": TEST_YEARS}
        keep_years = year_map[split]
        split_idx  = [i for i, ts in enumerate(timestamps_full)
                      if ts.year in keep_years]
        assert len(split_idx) > 0, (
            f"No timesteps found for split='{split}' with years={keep_years}. "
            f"Check that the dataset covers these years."
        )

        obs        = obs_full[split_idx]
        mask       = mask_full[split_idx]
        timestamps = [timestamps_full[i] for i in split_idx]

        # ------------------------------------------------------------------
        # 4. Normalise split observations using training-split statistics
        # ------------------------------------------------------------------
        obs_norm = obs.clone()
        for v in range(NUM_VARIABLES):
            obs_norm[:, :, v] = (
                (obs[:, :, v] - self.obs_stats["mean"][v]) / self.obs_stats["std"][v]
            ) * mask[:, :, v]

        # Optionally place tensors in shared memory so DataLoader workers share
        # the pages rather than each forking a private copy.
        # Only useful when num_workers > 0 on Linux/CUDA.
        # On macOS MPS, unified memory makes this redundant and the OS-level
        # shared-memory segment limit (kern.sysv.shmmax) can cause OOM errors
        # for large datasets — leave shared_memory=False (default) on Apple Silicon.
        if shared_memory:
            obs_norm.share_memory_()
            mask.share_memory_()
        self.obs  = obs_norm   # (T_split, N, V)
        self.mask = mask       # (T_split, N, V)

        # ------------------------------------------------------------------
        # 5. Hours-since-epoch for every split timestep
        # ------------------------------------------------------------------
        hours = torch.tensor(
            [_hours_since_epoch(ts) for ts in timestamps],
            dtype=torch.float32,
        )
        if shared_memory:
            hours.share_memory_()
        self.hours = hours   # (T_split,)

        self.timestamps = timestamps

        # ------------------------------------------------------------------
        # 6. Valid window start indices
        #
        #    A window at index i is valid iff:
        #      - i + W - 1 + max_delta_steps < T_split   (fits in the split)
        #      - timestamps[i].year ==
        #        timestamps[i + W - 1 + max_delta_steps].year
        #        (no year-boundary crossing, even for the longest lead-time)
        #
        #    Enforcing this for max_delta guarantees validity for all deltas
        #    in [1, max_delta_steps] without checking each one individually.
        # ------------------------------------------------------------------
        T         = len(timestamps)
        max_delta = self.max_delta_steps
        max_start = T - window_size - max_delta

        self.indices: list[int] = []
        for i in range(max(0, max_start + 1)):
            start_year  = timestamps[i].year
            target_year = timestamps[i + window_size - 1 + max_delta].year
            if start_year == target_year:
                self.indices.append(i)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        W = self.window_size
        K = self.num_delta_per_sample

        # ── Input window — shared across all K targets ─────────────────
        x       = self.obs[i : i + W]      # (W, N, V)
        x_mask  = self.mask[i : i + W]     # (W, N, V)
        x_hours = self.hours[i : i + W]    # (W,)

        # ── Target snapshot(s) ─────────────────────────────────────────
        if K == 1:
            # Single lead-time per sample
            # Random mode  : max_delta_steps != fixed delta_steps
            # Fixed mode   : both equal (original behaviour)
            if self.max_delta_steps != self.delta_steps:
                dt = int(torch.randint(1, self.max_delta_steps + 1, ()).item())
            else:
                dt = self.delta_steps

            t_idx = i + W - 1 + dt
            return {
                "x":           x,
                "x_mask":      x_mask,
                "x_hours":     x_hours,
                "y":           self.obs[t_idx],           # (N, V)
                "y_mask":      self.mask[t_idx],          # (N, V)
                "y_hours":     self.hours[t_idx],         # scalar
                "spatial":     self.spatial,              # (N, 15)
                "delta_steps": torch.tensor(dt, dtype=torch.long),
            }

        else:
            # Multi-delta: K distinct lead-times from [1, max_delta_steps]
            max_dt = self.max_delta_steps
            if K <= max_dt:
                # Guarantee distinct values via randperm
                deltas, _ = (torch.randperm(max_dt)[:K] + 1).sort()
            else:
                # More deltas than range — allow repeats
                deltas, _ = torch.randint(1, max_dt + 1, (K,)).sort()

            ys, y_masks, y_hrs = [], [], []
            for dt in deltas.tolist():
                t_idx = i + W - 1 + dt
                ys.append(self.obs[t_idx])
                y_masks.append(self.mask[t_idx])
                y_hrs.append(self.hours[t_idx])

            return {
                "x":           x,
                "x_mask":      x_mask,
                "x_hours":     x_hours,
                "y":           torch.stack(ys),        # (K, N, V)
                "y_mask":      torch.stack(y_masks),   # (K, N, V)
                "y_hours":     torch.stack(y_hrs),     # (K,)
                "spatial":     self.spatial,            # (N, 15)
                "delta_steps": deltas.long(),           # (K,)
            }
