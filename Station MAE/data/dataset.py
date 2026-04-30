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


# ---------------------------------------------------------------------------
# Fast local cache  (numpy memmap — direct worker access, no IPC overhead)
# ---------------------------------------------------------------------------

_FAST_CACHE_VERSION = "v1"
# Bump this when the on-disk format changes to force a rebuild.


def _fast_split_paths(fast_dir: str, split: str, train_years: list) -> dict:
    """
    Return a dict of expected file paths for one split's fast cache.

    Files are keyed by split + training years so that different training
    configurations (subset vs full) coexist in the same directory.
    """
    years_key = "_".join(str(y) for y in sorted(train_years))
    prefix    = f"{split}_{years_key}"
    return {
        "obs":     os.path.join(fast_dir, f"{prefix}_obs.npy"),
        "mask":    os.path.join(fast_dir, f"{prefix}_mask.npy"),
        "hours":   os.path.join(fast_dir, f"{prefix}_hours.npy"),
        "spatial": os.path.join(fast_dir, "spatial.pt"),   # shared across splits
        "meta":    os.path.join(fast_dir, f"{prefix}_meta.pt"),
    }


def fast_cache_save(
    fast_dir:        str,
    split:           str,
    train_years:     list,
    obs_norm:        torch.Tensor,   # (T_split, N, V)  normalised
    mask:            torch.Tensor,   # (T_split, N, V)
    hours:           torch.Tensor,   # (T_split,)
    spatial:         torch.Tensor,   # (N, 15)
    spatial_stats:   dict,
    obs_stats:       dict,
    indices:         list,
    window_size:     int,
    max_delta_steps: int,
) -> None:
    """
    Save split-specific normalised tensors as numpy .npy files.

    Subsequent calls to ``fast_cache_load`` will mmap these files, so each
    DataLoader worker reads directly from the OS page cache without any
    inter-process tensor transfer for the source data.

    Files are written to ``fast_dir`` (typically ``/tmp/station_mae_cache``
    which is a tmpfs RAM disk on Linux).  Workers independently open the
    same mmap files; the OS shares a single copy of the pages in the page
    cache across all processes — zero duplication, no /dev/shm exhaustion.
    """
    os.makedirs(fast_dir, exist_ok=True)
    paths = _fast_split_paths(fast_dir, split, train_years)
    print(f"[FastCache] Saving '{split}' split to {fast_dir} …", flush=True)
    t0 = time.time()

    np.save(paths["obs"],   obs_norm.numpy())
    np.save(paths["mask"],  mask.numpy())
    np.save(paths["hours"], hours.numpy())

    # spatial is the same for all splits — only write once
    if not os.path.exists(paths["spatial"]):
        torch.save(spatial, paths["spatial"])

    torch.save({
        "version":         _FAST_CACHE_VERSION,
        "obs_stats":       obs_stats,
        "spatial_stats":   spatial_stats,
        "indices":         indices,
        "window_size":     window_size,
        "max_delta_steps": max_delta_steps,
    }, paths["meta"])

    print(f"[FastCache] Saved '{split}' in {time.time() - t0:.1f}s", flush=True)


def fast_cache_load(
    fast_dir:    str,
    split:       str,
    train_years: list,
) -> "dict | None":
    """
    Load split-specific tensors as memory-mapped numpy arrays.

    Returns a dict with keys obs / mask / hours / spatial / obs_stats /
    spatial_stats / indices, or None if the cache files are absent or
    outdated (caller should rebuild and call fast_cache_save).

    The returned obs / mask / hours arrays use mmap_mode='c' (copy-on-write):
    read pages are served from the OS page cache shared across all worker
    processes; private writes go to a per-process in-memory buffer.
    """
    paths = _fast_split_paths(fast_dir, split, train_years)

    if not all(os.path.exists(paths[k]) for k in paths):
        return None   # cache miss — caller will build from scratch

    t0 = time.time()
    print(f"[FastCache] Loading '{split}' split from {fast_dir} …", flush=True)

    meta = torch.load(paths["meta"], weights_only=False)
    if meta.get("version") != _FAST_CACHE_VERSION:
        print(f"[FastCache] Version mismatch — will rebuild '{split}'.", flush=True)
        return None

    # mmap_mode='c' — copy-on-write:
    #   • Workers mmap the same .npy file; reads hit the OS page cache (one shared copy).
    #   • No /dev/shm consumed for source data — only the small per-batch slice tensors
    #     travel through the DataLoader queue.
    obs  = np.load(paths["obs"],   mmap_mode="c")
    mask = np.load(paths["mask"],  mmap_mode="c")
    hrs  = np.load(paths["hours"], mmap_mode="c")
    spatial = torch.load(paths["spatial"], weights_only=False)

    print(f"[FastCache] Loaded '{split}' in {time.time() - t0:.2f}s  "
          f"obs={obs.shape}", flush=True)

    return {
        "obs":           obs,
        "mask":          mask,
        "hours":         hrs,
        "spatial":       spatial,
        "obs_stats":     meta["obs_stats"],
        "spatial_stats": meta["spatial_stats"],
        "meta":          meta,   # includes indices, window_size, max_delta_steps, version
    }


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
        fast_cache_dir:       Optional fast local directory (e.g. ``/tmp/station_mae_cache``)
                              for split-specific numpy memmap files.  When set, each
                              DataLoader worker mmaps the .npy files directly from the
                              OS page cache instead of receiving tensors through the
                              inter-process queue — eliminating the file_system IPC
                              overhead for source data.  On Linux, ``/tmp`` is usually
                              a tmpfs (RAM-backed), so page faults are served at memory
                              speed.  First call builds and saves the .npy files; all
                              subsequent calls (and all workers) load instantly.
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
        fast_cache_dir:       "str | None" = None,
        exclude_stations:     "list | None" = None,
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

        effective_train_years = train_years if train_years is not None else TRAIN_YEARS

        # ------------------------------------------------------------------
        # 0. Fast cache (numpy memmap on local storage)  —  try first
        #
        # Each DataLoader worker independently mmaps the same .npy files.
        # Read pages are served from the OS page cache — one shared RAM copy
        # across all processes, no /dev/shm consumed for source data.
        # On Linux, /tmp is a tmpfs (RAM-backed), so reads are at memory speed.
        # ------------------------------------------------------------------
        if fast_cache_dir is not None:
            cached = fast_cache_load(fast_cache_dir, split, effective_train_years)

            # Validate: cache must be built for the same window / horizon
            _cache_ok = (
                cached is not None
                and cached["meta"].get("window_size")     == window_size
                and cached["meta"].get("max_delta_steps") == self.max_delta_steps
            )
            # For val/test: normalisation stats must match the train dataset's
            if _cache_ok and obs_stats is not None:
                cached_stats = cached["obs_stats"]
                _cache_ok = (
                    torch.allclose(cached_stats["mean"], obs_stats["mean"])
                    and torch.allclose(cached_stats["std"],  obs_stats["std"])
                )

            if _cache_ok:
                self.obs_stats     = cached["obs_stats"]
                self.spatial       = cached["spatial"]
                self.spatial_stats = cached["spatial_stats"]
                self.indices       = cached["meta"]["indices"]

                # Keep numpy mmap arrays alive (torch.from_numpy holds a weak ref
                # to the numpy array; storing explicitly prevents GC).
                self._obs_np    = cached["obs"]    # (T_split, N, V) mmap
                self._mask_np   = cached["mask"]   # (T_split, N, V) mmap
                self._hours_np  = cached["hours"]  # (T_split,)      mmap

                # Torch tensors backed by the mmap file on /tmp.
                # mmap_mode='c' returns a copy-on-write memmap — writeable flag
                # is set, so torch.from_numpy() accepts it.  Workers that fork
                # will each trigger page faults only on the pages they actually
                # touch; the kernel serves them from the shared page cache.
                self.obs   = torch.from_numpy(self._obs_np)
                self.mask  = torch.from_numpy(self._mask_np)
                self.hours = torch.from_numpy(self._hours_np)
                # Apply station exclusion before returning from fast-cache path
                if exclude_stations:
                    keep = StationMAEDataset._resolve_keep_indices(
                        ds, exclude_stations)
                    self._apply_station_exclusion(keep, exclude_stations)
                return   # skip the build path entirely

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
        # 2. Normalisation statistics — ALWAYS from TRAINING years only
        #
        # Invariant that MUST be respected across all three splits:
        #   • Training dataset  : obs_stats=None  → computed here from training
        #                         years AFTER log1p has been applied above.
        #   • Val / Test        : obs_stats=train_ds.obs_stats  → same stats,
        #                         no recomputation.  Log1p is still applied
        #                         (see step 1b above) before normalisation.
        #
        # Never pass obs_stats computed from raw (non-log1p) observations, or
        # stats from a different training split — that would create a scale
        # mismatch between the model's training distribution and inference.
        #
        # train_years must match the training run so that stats are anchored
        # to the exact same data the model saw, avoiding drift between a
        # subset run (e.g. [2020,2021]) and a full-data (2017–2021) job.
        # ------------------------------------------------------------------
        if obs_stats is None:
            self.obs_stats = compute_obs_stats(
                ds,
                obs_full     = obs_full,       # already log1p-transformed above
                mask_full    = mask_full,
                timestamps   = timestamps_full,
                train_years  = effective_train_years,
            )
        else:
            # Stats passed in — assumed to be from train_ds.obs_stats
            # (log1p-aware, same training years).
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
        # 7. Save fast cache for future runs (if fast_cache_dir is set)
        # ------------------------------------------------------------------
        if fast_cache_dir is not None:
            fast_cache_save(
                fast_dir      = fast_cache_dir,
                split         = split,
                train_years   = effective_train_years,
                obs_norm      = self.obs,
                mask          = self.mask,
                hours         = self.hours,
                spatial       = spatial,
                spatial_stats = spatial_stats,
                obs_stats     = self.obs_stats,
                indices       = self.indices,
                window_size   = window_size,
                max_delta_steps = self.max_delta_steps,
            )

        # ── Station exclusion (normal build path) ─────────────────────────────
        if exclude_stations:
            keep = StationMAEDataset._resolve_keep_indices(ds, exclude_stations)
            self._apply_station_exclusion(keep, exclude_stations)

    # ------------------------------------------------------------------
    # Station exclusion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_keep_indices(
        ds: PeakWeatherDataset,
        exclude_stations: list,
    ) -> list:
        """
        Return a sorted list of station *positions* to keep after dropping
        ``exclude_stations``.

        Each entry in ``exclude_stations`` is matched (case-insensitive) against:
          1. The stations_table index  (station ID / numeric code)
          2. The ``name`` column        (full station name)
          3. The ``abbr`` column        (short abbreviation)

        Raises a warning if no stations were matched.
        """
        stns       = ds.stations_table
        excl_upper = {str(s).upper() for s in exclude_stations}
        drop       = set()

        for pos, (idx, row) in enumerate(stns.iterrows()):
            candidates = {str(idx).upper()}
            for col in ("name", "abbr", "station_name"):
                if col in row.index and not pd.isna(row[col]):
                    candidates.add(str(row[col]).upper())
            if candidates & excl_upper:
                drop.add(pos)

        if not drop:
            import warnings
            warnings.warn(
                f"[StationFilter] No stations matched {exclude_stations} — "
                f"nothing was excluded.  Check against ds.stations_table.",
                UserWarning, stacklevel=4,
            )

        return [i for i in range(len(stns)) if i not in drop]

    def _apply_station_exclusion(
        self,
        keep: list,
        exclude_stations: list,
    ) -> None:
        """
        Drop excluded stations from obs, mask, and spatial in-place.

        Uses advanced indexing which always returns a new contiguous tensor,
        so the result is safe regardless of the backing storage (mmap or RAM).
        """
        keep_t       = torch.tensor(keep, dtype=torch.long)
        n_before     = self.obs.shape[1]
        self.obs     = self.obs    [:, keep_t, :].contiguous()
        self.mask    = self.mask   [:, keep_t, :].contiguous()
        self.spatial = self.spatial[keep_t, :]   .contiguous()
        dropped      = n_before - len(keep)
        if dropped > 0:
            print(
                f"[StationFilter] Excluded {dropped} station(s) "
                f"{exclude_stations}  → N={len(keep)} (was {n_before})",
                flush=True,
            )

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
        # .clone() gives each slice its own storage rather than a view of the
        # full backing tensor.  This is critical for the DataLoader's file_system
        # sharing strategy: without it, the entire obs storage (hundreds of MB)
        # may be written to /tmp for every batch instead of just the slice.
        x       = self.obs  [i : i + W].clone()   # (W, N, V)
        x_mask  = self.mask [i : i + W].clone()   # (W, N, V)
        x_hours = self.hours[i : i + W].clone()   # (W,)

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
