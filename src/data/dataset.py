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

# Donor-similarity weighting (see normalise_observations).
SIM_HEIGHT_IDX    = slice(6, 8)   # station_height, dem in the 15-d spatial vector
SIM_HEIGHT_WEIGHT = 3.0           # elevation counts 3x a terrain descriptor

TRAIN_YEARS = list(range(2017, 2022))   # 2017–2021
VAL_YEARS   = [2022]
TEST_YEARS  = [2023, 2024]
# Both test.py and test_lstm.py read this same constant, so the LSTM and transformer
# are guaranteed to evaluate on the identical test loader / windows.


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
        freq="10min"
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
    obs:          torch.Tensor,
    mask:         torch.Tensor,
    per_station:  bool = True,
    coords:       "torch.Tensor | None" = None,
    station_ids:  "list | None" = None,
    alt_km_per_m: float = 0.2,
    verbose:      bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Normalise each (station, variable) pair to zero-mean unit-variance using
    only present values for that specific station and variable.

    Per-station normalisation (vs. global per-variable) removes the systematic
    altitude-driven offset so that every station's distribution is centred at
    zero — the model learns the inter-station differences purely from the spatial
    and topographic embeddings rather than the raw signal level.

    Sparse (station, variable) pairs
    -------------------------------
    A pair with fewer than MIN_OBS training observations has no usable statistics
    of its own — GES pressure, for example, has ZERO (the sensor was installed in
    2023, after the training period). Such pairs previously fell back to the
    cross-station GLOBAL mean and std, which is badly wrong for altitude-driven
    variables: the network spans ~200–3600 m, so a global mean pressure is
    meaningless at any individual station.

    We now borrow from the NEAREST station that does have data for that variable,
    where "nearest" combines horizontal distance with an altitude penalty
    (``alt_km_per_m``). Altitude dominates pressure (~12 hPa/100 m) and strongly
    affects temperature (~0.65 °C/100 m), so a donor 5 km away but 500 m higher is
    a worse choice than one 50 km away at the same elevation. The default
    0.2 km/m makes 100 m of elevation cost as much as 20 km of distance.

    Global stats remain the last resort, used only when no station has data for
    that variable at all.

    Args:
        obs:          (T, N, V) raw observations (0.0 where absent)
        mask:         (T, N, V) presence mask (1.0 present, 0.0 absent)
        per_station:  False reproduces the old global-only behaviour.
        coords:       (N, 3) easting[m], northing[m], height[m]. Required for
                      nearest-donor substitution; without it we fall back to
                      global stats and say so.
        station_ids:  optional names, used only to make the log readable.
        alt_km_per_m: altitude penalty in km-equivalent per metre of Δheight.
        verbose:      print every substitution (there are few, and they matter).

    Returns:
        obs_norm : (T, N, V) normalised (absent values zeroed)
        stats    : {"mean": (N, V), "std": (N, V), "donor": (N, V) long,
                    "n_obs": (N, V)}
                   donor[i, v] = index of the station whose stats station i used
                   for variable v; -1 = its own, -2 = global fallback.
    """
    T, N, V = obs.shape

    # ── Global normalization (old behaviour, one mean/std per variable) ────────
    if not per_station:
        means_g, stds_g = [], []
        obs_norm = obs.clone()
        for v in range(V):
            vals = obs[:, :, v][mask[:, :, v] == 1.0]
            m = vals.mean() if len(vals) > 0 else torch.tensor(0.0)
            s = vals.std().clamp(min=1e-6) if len(vals) > 1 else torch.tensor(1.0)
            obs_norm[:, :, v] = (obs[:, :, v] - m) / s * mask[:, :, v]
            means_g.append(m)
            stds_g.append(s)
        return obs_norm, {"mean": torch.stack(means_g), "std": torch.stack(stds_g)}

    # Minimum number of training observations required to use per-station stats.
    # Stations with fewer observations fall back to cross-station global stats,
    # preventing near-zero stds (and their reciprocals → astronomical normalised
    # values) for stations that were inactive during the training period.
    MIN_OBS = 50

    # Donor similarity is measured in the z-scored spatial-feature space built by
    # build_spatial_features(): [easting, northing, sin/cos aspect x2,
    # station_height, dem, TPI, slope x2, SN/WE derivatives x2] = 15 dims.
    # Elevation (station_height, dem -> indices 6,7) is up-weighted because it
    # governs pressure (~12 hPa/100 m) and temperature (~0.65 degC/100 m); the
    # terrain descriptors matter, but far less.

    raw_count = mask.sum(dim=0)                                    # (N, V) unclipped
    count     = raw_count.clamp(min=1.0)

    # ── Per-station mean ───────────────────────────────────────────────────────
    means = (obs * mask).sum(dim=0) / count                        # (N, V)

    # ── Global (cross-station) fallback mean per variable ─────────────────────
    _g_count = mask.sum(dim=(0, 1)).clamp(min=1)                   # (V,)
    _g_mean  = (obs * mask).sum(dim=(0, 1)) / _g_count             # (V,)

    # ── Per-station std ────────────────────────────────────────────────────────
    diff_sq = ((obs - means.unsqueeze(0)) ** 2) * mask             # (T, N, V)
    stds    = (diff_sq.sum(dim=0) / count.clamp(min=2.0)).sqrt()   # (N, V)

    # ── Global fallback std per variable ──────────────────────────────────────
    _g_diff = ((obs - _g_mean[None, None, :]) ** 2) * mask        # (T, N, V)
    _g_std  = (_g_diff.sum(dim=(0, 1)) / _g_count.clamp(min=2)).sqrt().clamp(min=1e-6)

    # ── Replace sparse-station stats: nearest donor, then global ──────────────
    # donor[i, v]: -1 = station i used its own stats, >=0 = index of the donor
    # station it borrowed from, -2 = no donor existed so global stats were used.
    _sparse = raw_count < MIN_OBS                                   # (N, V) bool
    donor   = torch.full((N, V), -1, dtype=torch.long)

    if _sparse.any():
        n_bad = int(_sparse.sum())
        if coords is None:
            print(f"  [norm] {n_bad} sparse (station,variable) pair(s) but no station "
                  f"features supplied → global stats used (worse; see docstring)")
            means = torch.where(_sparse, _g_mean.unsqueeze(0).expand_as(means), means)
            stds  = torch.where(_sparse, _g_std.unsqueeze(0).expand_as(stds),   stds)
            donor = torch.where(_sparse, torch.full_like(donor, -2), donor)
        else:
            c = coords.to(torch.float64)
            if c.shape[1] == 3:
                # Legacy path: raw easting[m], northing[m], height[m].
                dxy  = torch.cdist(c[:, :2], c[:, :2]) / 1000.0        # (N, N) km
                dz   = (c[:, 2:3] - c[:, 2:3].T).abs() * alt_km_per_m  # km-equivalent
                dist = torch.sqrt(dxy ** 2 + dz ** 2)
            else:
                # Similarity over the full station character: z-scored spatial
                # features (position, elevation, DEM, slope, aspect, TPI, …) with
                # elevation up-weighted, since it drives pressure and temperature
                # far more than the terrain descriptors do.
                w = torch.ones(c.shape[1], dtype=torch.float64)
                w[:SIM_HEIGHT_IDX.start] = 1.0
                w[SIM_HEIGHT_IDX] = SIM_HEIGHT_WEIGHT
                dist = torch.cdist(c * w, c * w)
            dist.fill_diagonal_(float("inf"))                          # never self

            subs = []
            for v in range(V):
                healthy = ~_sparse[:, v]                              # (N,) donors
                if not healthy.any():
                    # nobody has this variable — global is genuinely the only option
                    sel = _sparse[:, v]
                    means[sel, v] = _g_mean[v]
                    stds[sel, v]  = _g_std[v]
                    donor[sel, v] = -2
                    continue
                d_v = dist.clone()
                d_v[:, ~healthy] = float("inf")                       # only healthy donors
                nearest = d_v.argmin(dim=1)                           # (N,)
                for i in torch.nonzero(_sparse[:, v]).flatten().tolist():
                    j = int(nearest[i])
                    means[i, v] = means[j, v]
                    stds[i, v]  = stds[j, v]
                    donor[i, v] = j
                    subs.append((i, j, v, float(dist[i, j]),
                                 float(c[i, 2] - c[j, 2]), int(raw_count[i, v])))

            if verbose and subs:
                print(f"  [norm] {len(subs)} sparse (station,variable) pair(s) took "
                      f"stats from their most similar station")
            _still = int((donor == -2).sum())
            if _still:
                # This one is always worth saying: it means the variable is
                # missing network-wide, so the fallback really is global.
                print(f"  [norm] {_still} (station,variable) pair(s) had no donor "
                      f"anywhere → global stats used")

    stds = stds.clamp(min=1e-6)

    # ── Normalise and zero-out absent entries ──────────────────────────────────
    obs_norm = (obs - means.unsqueeze(0)) / stds.unsqueeze(0)      # (T, N, V)
    obs_norm = obs_norm * mask                                      # zero absent

    return obs_norm, {"mean": means, "std": stds,
                      "donor": donor, "n_obs": raw_count}


def compute_obs_stats(
    ds:          PeakWeatherDataset,
    obs_full:    "torch.Tensor | None" = None,
    mask_full:   "torch.Tensor | None" = None,
    timestamps:  "list | None"         = None,
    train_years: "list[int] | None"    = None,
    per_station: bool = True,
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

    # Station geometry, so sparse (station, variable) pairs can borrow stats from
    # the nearest comparable station instead of the altitude-blind global mean.
    # Full z-scored station character — position, elevation, DEM, slope, aspect,
    # TPI — so a station with too little training data borrows from the station
    # it most resembles, not from a network-wide average that describes nowhere.
    coords, station_ids = None, None
    try:
        coords, _ = build_spatial_features(ds)               # (N, 15), z-scored
        coords = coords.to(torch.float64)
        station_ids = ds.stations_table.index.tolist()
        if coords.shape[0] != obs_train.shape[1]:            # never silently misalign
            print(f"  [norm] stations_table has {coords.shape[0]} rows but obs has "
                  f"{obs_train.shape[1]} stations — similar-station lookup disabled")
            coords, station_ids = None, None
    except Exception as e:                                   # noqa: BLE001
        print(f"  [norm] could not build station features ({type(e).__name__}) — "
              f"similar-station lookup disabled")

    _, stats = normalise_observations(
        obs_train, mask_train, per_station=per_station,
        coords=coords, station_ids=station_ids,
    )
    return stats


# ---------------------------------------------------------------------------
# Disk cache for preprocessed tensors
# ---------------------------------------------------------------------------

_CACHE_FILENAME = "peakweather_obs_cache.pt"


# ---------------------------------------------------------------------------
# Fast local cache  (numpy memmap — direct worker access, no IPC overhead)
# ---------------------------------------------------------------------------

# Bump whenever the CONTENT of a cached file would change: the cache stores
# NORMALISED observations, so any change to the normalisation algorithm must
# invalidate it or stale values are served silently.
#   v5: log1p transform on precipitation
#   v6: nearest-station stats for sparse (station,variable) pairs
#       (previously the altitude-blind global fallback)
_FAST_CACHE_VERSION = "v6"
# Bump this when the on-disk format changes to force a rebuild.
# v2: stores "all_valid_indices" (pre-mode pool) instead of post-stride
#     "indices", so the windowing strategy can be changed without rebuilding.
# v3: obs_stats["mean"] / ["std"] are now (N, V) per-station-per-variable
#     tensors instead of (V,) global tensors.
# v4: sparse-station fallback — stations with < 50 training observations use
#     cross-station global stats to prevent near-zero stds and astronomical
#     normalised values.


def _fast_split_paths(fast_dir: str, split: str, train_years: list,
                      per_station: bool = True,
                      exclude_stations: "list | None" = None) -> dict:
    """
    Return a dict of expected file paths for one split's fast cache.

    The key must contain EVERY input that changes the cached bytes, because the
    files hold NORMALISED observations, not raw ones:

      * split, train_years  — which rows, and which stats were fitted
      * per_station         — per-station vs global normalisation produce
                              completely different values. Sharing one file
                              between the two silently mixes them, which is how
                              tw6 and tw12 ended up incomparable earlier.

    _FAST_CACHE_VERSION covers changes to the normalisation ALGORITHM itself.

    exclude_stations is ALSO in the key, but not for the reason this docstring
    used to give ("changes N, so the arrays have a different shape"). That is
    not true: the cached arrays are written at FULL N. fast_cache_save() is
    called before _apply_station_exclusion(), and on the load path exclusion is
    likewise applied after the arrays come back — so the cached bytes are
    identical whether or not stations are excluded, and the key only produces
    duplicate copies of the same data under different names.

    It is kept anyway, deliberately: it is cheap, and a filename that records
    the run's exclusion list is easier to audit than one that does not. Do not
    "optimise" it away expecting a behaviour change — there is none to gain, and
    removing it invalidates every cache already on disk.

    What this means for the station axis, which is the part that actually
    matters: every split holds the same N_full stations in the same order,
    spatial.pt is one shared file across splits, and exclusion is a sorted
    index-select applied identically everywhere. obs_stats stays at (N_full, V)
    on purpose (see _apply_station_exclusion) so val/test can compare against
    train_ds.obs_stats without re-deriving it; slice it through _keep_indices.
    """
    years_key = "_".join(str(y) for y in sorted(train_years))
    norm_key  = "ps" if per_station else "gl"
    excl_key  = ("_x" + "-".join(sorted(str(e) for e in exclude_stations))
                 ) if exclude_stations else ""
    prefix    = f"{split}_{years_key}_{norm_key}{excl_key}"
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
    per_station:     bool = True,
    exclude_stations: "list | None" = None,
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
    paths = _fast_split_paths(fast_dir, split, train_years, per_station, exclude_stations)
    print(f"[FastCache] Saving '{split}' split to {fast_dir} …", flush=True)
    t0 = time.time()

    np.save(paths["obs"],   obs_norm.numpy())
    np.save(paths["mask"],  mask.numpy())
    np.save(paths["hours"], hours.numpy())

    # spatial is the same for all splits — only write once
    if not os.path.exists(paths["spatial"]):
        torch.save(spatial, paths["spatial"])

    torch.save({
        "version":           _FAST_CACHE_VERSION,
        "obs_stats":         obs_stats,
        "spatial_stats":     spatial_stats,
        "all_valid_indices": indices,   # pre-mode pool; mode is applied on load
        "window_size":       window_size,
        "max_delta_steps":   max_delta_steps,
    }, paths["meta"])

    print(f"[FastCache] Saved '{split}' in {time.time() - t0:.1f}s", flush=True)


def fast_cache_load(
    fast_dir:    str,
    split:       str,
    train_years: list,
    per_station: bool = True,
    exclude_stations: "list | None" = None,
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
    paths = _fast_split_paths(fast_dir, split, train_years, per_station, exclude_stations)

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

    Windowing strategies  (index_mode)
    ------------------------------------
    All three strategies first build a *pool* of contiguity-valid start indices
    — positions where the full span ``[i, i + W + max_delta - 1]`` contains no
    temporal gaps (checked vectorially via nanosecond timestamp differences).
    This replaces the old year-boundary heuristic which incorrectly rejected
    valid cross-year windows and missed mid-series instrument outages.

    ``"sliding"``  (Strategy C — default, GraphDOP / most baselines):
        Every contiguity-valid start, optionally thinned by ``train_stride`` on
        the training split.  DataLoader ``shuffle=True`` gives random-without-
        replacement ordering each epoch.  Maximal data coverage.

    ``"blocks"``   (Strategy B — PatchTST / iTransformer):
        Greedy non-overlapping selection: a window is accepted only when its
        start is at least W steps ahead of the previous accepted start.  No two
        windows share any input timestep.  Cleanest gradient signal, smallest
        epoch size.

    ``"random"``   (Strategy A — Aurora / W-MAE / VideoMAE):
        Full pool stored; ``__getitem__`` ignores the DataLoader index and
        samples uniformly at random from the pool on every call.  Gives
        per-item independent sampling (with replacement) — different windows
        every epoch with no systematic coverage bias.
        Epoch length is controlled by ``random_epoch_size`` (default:
        ``len(pool) // window_size`` ≈ number of non-overlapping blocks)
        to avoid oversampling the pool.

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
        obs_stats:            Normalisation stats dict {"mean": (N, V), "std": (N, V)}.
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
        index_mode:           Windowing strategy for the training split (see class
                              docstring).  One of "sliding" (default), "blocks", or
                              "random".  Val/test always use "sliding" with stride=1.
        train_stride:         Uniform thinning step applied to the "sliding" train
                              index list (1 = every start, 6 = hourly, 12 = 2-h).
                              Ignored for "blocks" and "random" modes.
        delta_mode:           How lead-times are selected per sample.
                              ``"fixed_grid"`` (default): always return K targets at
                              uniformly-spaced horizons 0, delta_grid_stride,
                              2·delta_grid_stride, …, max_delta_steps steps.
                              E.g. max_delta=36, stride=3 → 0, 30 min, 1 h, … 6 h
                              (K=13).  The encoder runs once; the decoder runs K times.
                              ``"random"``: each call draws num_delta_per_sample
                              distinct lead-times uniformly from [1, max_delta_steps].
        delta_grid_stride:    Spacing between fixed-grid horizons in 10-min steps
                              (default 3 = 30 min).  Only used when delta_mode="fixed_grid".
        random_epoch_size:    Number of samples per epoch when index_mode="random".
                              Default: len(pool) // window_size (≈ non-overlapping blocks).
                              Tune upward if the model under-trains, downward to cut
                              epoch time.
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
        train_stride:         int  = 1,
        index_mode:           str  = "sliding",
        delta_mode:           str  = "fixed_grid",
        delta_grid_stride:    int  = 3,
        random_epoch_size:    "int | None" = None,
        global_norm:          bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val' or 'test', got '{split}'"
        assert delta_steps >= 0, "delta_steps must be >= 0"
        assert num_delta_per_sample >= 1, "num_delta_per_sample must be >= 1"
        if num_delta_per_sample > 1:
            assert max_delta_steps is not None, \
                "max_delta_steps must be set when num_delta_per_sample > 1"
        assert train_stride >= 1, "train_stride must be >= 1"
        assert index_mode in ("sliding", "blocks", "random"), (
            f"index_mode must be 'sliding', 'blocks', or 'random', got '{index_mode}'"
        )
        assert delta_mode in ("fixed_grid", "random"), (
            f"delta_mode must be 'fixed_grid' or 'random', got '{delta_mode}'"
        )
        assert delta_grid_stride >= 1, "delta_grid_stride must be >= 1"

        self.window_size          = window_size
        self.delta_steps          = delta_steps
        self.split                = split
        self.num_delta_per_sample = num_delta_per_sample
        self.train_stride         = train_stride
        self.index_mode           = index_mode
        self.delta_mode           = delta_mode
        self.delta_grid_stride    = delta_grid_stride
        # Effective upper bound on lead-time — governs valid index calculation
        # and random sampling in __getitem__
        self.max_delta_steps = max_delta_steps if max_delta_steps is not None \
                               else delta_steps

        # Fixed-grid lead-times: stride, 2·stride, …, max_delta_steps
        # delta=0 is the reconstruction / inpainting horizon: the target is the
        # last input timestep itself.  Loss is computed on MASKED stations only
        # (see forward_multi_delta in mae.py) — visible stations see their own
        # values directly in the encoder and can trivially copy them, so
        # supervising them at delta=0 contributes near-zero gradient.
        # Restricting to masked stations gives a pure gap-filling signal.
        # E.g. max_delta=36, stride=3 → [0, 3, 6, …, 36]  (K=13 horizons)
        if delta_mode == "fixed_grid":
            self.delta_grid = list(range(0, self.max_delta_steps + 1, delta_grid_stride))
        else:
            self.delta_grid = []

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
            cached = fast_cache_load(fast_cache_dir, split, effective_train_years,
                                     per_station=not global_norm,
                                     exclude_stations=exclude_stations)

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

                # Apply windowing strategy to the pre-mode pool stored in cache.
                # Mode can be changed freely without rebuilding the cache.
                _all_valid = cached["meta"]["all_valid_indices"]
                self.indices = StationMAEDataset._apply_index_mode(
                    _all_valid, window_size, train_stride, index_mode, split,
                )
                # Epoch size for random strategy
                self._random_epoch_size = (
                    random_epoch_size if random_epoch_size is not None
                    else max(1, len(_all_valid) // window_size)
                )

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
                obs_full     = obs_full,
                mask_full    = mask_full,
                timestamps   = timestamps_full,
                train_years  = effective_train_years,
                per_station  = not global_norm,
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
        #
        # obs_stats["mean"] / ["std"] are (N, V) per-station-per-variable
        # tensors.  Broadcasting: (T, N, V) - (1, N, V) / (1, N, V).
        # Absent values (mask==0) are zeroed after normalisation; the model
        # handles them via VariableProjection.var_absent_embedding (the zeros
        # themselves are never read — contributions are mask-gated).
        # ------------------------------------------------------------------
        _mean = self.obs_stats["mean"]
        _std  = self.obs_stats["std"]
        if _mean.dim() == 1:
            # Global (V,) stats — broadcast over T and N automatically
            obs_norm = (obs - _mean) / _std
        else:
            # Per-station (N, V) stats — unsqueeze for T dimension
            obs_norm = (obs - _mean.unsqueeze(0)) / _std.unsqueeze(0)
        obs_norm = obs_norm * mask                               # zero absent

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
        # 6. Valid window start indices — vectorised contiguity check
        #
        # A window starting at i is valid iff the entire span
        #   [i, i + W - 1 + max_delta]
        # contains no temporal gaps (no missing 10-min steps).
        #
        # This replaces the previous year-boundary heuristic which:
        #   (a) rejected valid windows that cross 31 Dec → 1 Jan within the
        #       same training split (e.g. 2020-12-31 → 2021-01-01), wasting
        #       several hundred valid starts per year boundary; and
        #   (b) did not catch mid-year instrument outages, which produce
        #       windows with inflated zero-fill proportions in x_mask.
        #
        # Implementation:
        #   1. Compute nanosecond differences between consecutive timestamps
        #      (vectorised via pd.DatetimeIndex.asi8 → np.diff).
        #   2. Build a prefix-sum array of "bad gap" counts so each window's
        #      validity can be tested in O(1).
        #   3. Apply the chosen windowing strategy (_apply_index_mode).
        #
        # Note: timestamps are treated as UTC-equivalent for the diff
        # comparison.  If the source data uses local time (CET/CEST), windows
        # that span a DST transition will show a 50-min or 70-min gap for
        # exactly one step and be correctly rejected.
        # ------------------------------------------------------------------
        T         = len(timestamps)
        max_delta = self.max_delta_steps

        # ── Vectorised gap mask ────────────────────────────────────────────
        _step_ns  = int(pd.Timedelta("10min").value)          # expected gap (ns)
        _ts_ns    = pd.DatetimeIndex(timestamps).asi8         # (T,) int64 ns
        _diffs    = np.diff(_ts_ns)                           # (T-1,) ns between steps
        _bad      = (_diffs != _step_ns).astype(np.int32)    # 1 where gap is wrong

        # bad_count[i] = number of bad gaps strictly before position i
        # bad_count[i+k] - bad_count[i] == 0  ↔  span [i, i+k-1] is gap-free
        _bad_count = np.zeros(T + 1, dtype=np.int32)
        _bad_count[1:T] = np.cumsum(_bad)   # (T-1,) diffs → positions 1..T-1
        # _bad_count[T] is never queried (max index used is T-1); leave as zero.

        # A window needs the span [i, i + (W-1) + max_delta] to be gap-free,
        # i.e. no bad gaps among positions i, i+1, …, i + (W-1) + max_delta - 1.
        _need     = window_size - 1 + max_delta     # number of steps past start
        _max_i    = max(0, T - window_size - max_delta)

        all_valid: list[int] = [
            i for i in range(_max_i)
            if _bad_count[i + _need] == _bad_count[i]
        ]

        # Apply chosen windowing strategy (see _apply_index_mode docstring).
        self.indices = StationMAEDataset._apply_index_mode(
            all_valid, window_size, train_stride, index_mode, split,
        )
        # Epoch size for random strategy: default = number of non-overlapping blocks
        # in the pool, which covers the data without oversampling.
        self._random_epoch_size = (
            random_epoch_size if random_epoch_size is not None
            else max(1, len(all_valid) // window_size)
        )

        # ------------------------------------------------------------------
        # 7. Save fast cache for future runs (if fast_cache_dir is set)
        #
        # We persist the pre-mode all_valid pool, not the post-mode self.indices,
        # so that the windowing strategy can be changed without rebuilding.
        # The mode is applied in _apply_index_mode each time the cache is loaded.
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
                indices       = all_valid,       # ← pre-mode pool, not post-mode
                window_size   = window_size,
                max_delta_steps = self.max_delta_steps,
                # Must match the load key exactly, or the file is written under
                # one name and looked up under another (silent cache miss).
                per_station      = not global_norm,
                exclude_stations = exclude_stations,
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
            # Build candidate set: try both raw string and int-normalised form.
            # Parquet/HDF files often store integer station IDs as floats (e.g.
            # 110.0), so str(idx) gives '110.0' which never matches '110'.
            # We also try int(float(idx)) → '110' to handle this case.
            candidates = {str(idx).upper()}
            # Handle float-formatted integer indices: '110.0' → '110'
            try:
                candidates.add(str(int(float(str(idx)))).upper())
            except (ValueError, TypeError):
                pass
            # Check all common station name / abbreviation column names,
            # including PeakWeather's nat_abbr column.
            for col in ("name", "abbr", "nat_abbr", "station_name"):
                if col in row.index and not pd.isna(row[col]):
                    val = str(row[col])
                    candidates.add(val.upper())
                    try:
                        candidates.add(str(int(float(val))).upper())
                    except (ValueError, TypeError):
                        pass
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

    @staticmethod
    def _apply_index_mode(
        all_valid:    list,
        window_size:  int,
        train_stride: int,
        index_mode:   str,
        split:        str,
    ) -> list:
        """
        Apply the chosen windowing strategy to the contiguity-valid index pool.

        All three strategies receive the same pool of gap-checked start indices
        and differ only in how they select from it.  Val/test splits always
        use "sliding" with stride=1 regardless of the requested index_mode.

        Strategy A  ("random"):
            Returns the full pool unchanged.  ``__getitem__`` samples uniformly
            at random on every call (ignoring the DataLoader's index), giving
            independent per-item sampling with replacement.  Every epoch draws
            a fresh random subset of size ``len(dataset)`` from the full pool —
            no two epochs see the same sequence of windows.
            Refs: Aurora, W-MAE, VideoMAE.

        Strategy B  ("blocks"):
            Greedy non-overlapping selection.  Scans ``all_valid`` in order;
            accepts index ``i`` only when ``i >= last_accepted + window_size``.
            Guarantees zero input-timestep overlap between any two windows.
            Produces the smallest training set but the most independent samples.
            Refs: PatchTST, iTransformer, TimesNet.

        Strategy C  ("sliding", default):
            Full pool, optionally thinned by ``train_stride`` on training only.
            DataLoader ``shuffle=True`` gives random-without-replacement epoch
            ordering.  Maximum data coverage; recommended default.
            Refs: GraphDOP (ECMWF), most sliding-window baselines.

        Args:
            all_valid:    Contiguity-checked valid start indices.
            window_size:  W — input window length in timesteps.
            train_stride: Thinning step for "sliding" training indices (≥ 1).
            index_mode:   "sliding", "blocks", or "random".
            split:        "train", "val", or "test".  Val/test always → sliding/1.
        """
        # Val/test default: full sliding pool (stride=1) for stable metric estimates.
        # Pass index_mode="blocks" to get non-overlapping windows — useful for
        # fast evaluation runs and final paper metrics (~1,460 windows vs ~105k).
        if split != "train":
            if index_mode == "blocks":
                # Apply a split-specific starting offset so val and test blocks
                # are anchored to different hours of the day.
                #
                # With W=72 (12 h), non-overlapping blocks repeat every 12 h.
                # Without an offset, both val (2022) and test (2023-2024) start
                # at timestep 0 of their split, which typically aligns to midnight
                # → blocks land on {00:00, 12:00} every day.
                #
                # Offset by W//4 = 18 steps = 3 h for the TEST split:
                #   val  → blocks at {00:00, 12:00, 00:00, …}
                #   test → blocks at {03:00, 15:00, 03:00, …}
                #
                # This ensures test evaluation samples different diurnal patterns
                # than validation — relevant for Alpine meteorology where 03:00
                # (cold drainage flows) and 15:00 (convective peak) are physically
                # distinct from midnight / noon.
                offset: int = 0 if split == "val" else window_size // 4
                result: list = []
                next_ok: int = offset
                for i in all_valid:
                    if i >= next_ok:
                        result.append(i)
                        next_ok = i + window_size
                return result
            # sliding (or random) on val/test. For sliding, honour the stride so the
            # forecast-origin spacing is controllable (e.g. train_stride=9 → every
            # 90 min = rolling-origin evaluation, denser than blocks).
            if index_mode == "sliding" and train_stride > 1:
                return list(all_valid)[::train_stride]
            return list(all_valid)

        if index_mode == "random":
            # Strategy A — return full pool; __getitem__ samples randomly
            return list(all_valid)

        elif index_mode == "blocks":
            # Strategy B — greedy non-overlapping: accept i only if gap ≥ W
            result: list = []
            next_ok: int = 0
            for i in all_valid:
                if i >= next_ok:
                    result.append(i)
                    next_ok = i + window_size
            return result

        elif index_mode == "sliding":
            # Strategy C — full sliding window, optional uniform stride
            if train_stride > 1:
                return list(all_valid)[::train_stride]
            return list(all_valid)

        else:
            raise ValueError(
                f"index_mode='{index_mode}' is not recognised. "
                f"Choose 'sliding' (Strategy C, default), "
                f"'blocks' (Strategy B), or 'random' (Strategy A)."
            )

    def _apply_station_exclusion(
        self,
        keep: list,
        exclude_stations: list,
    ) -> None:
        """
        Drop excluded stations from obs, mask, and spatial in-place.

        Uses advanced indexing which always returns a new contiguous tensor,
        so the result is safe regardless of the backing storage (mmap or RAM).

        Sets ``self._keep_indices`` (LongTensor of kept positions in the
        original N_full station list) so callers can slice obs_stats for
        logging / unnormalization in physical units.
        """
        keep_t       = torch.tensor(keep, dtype=torch.long)
        self._keep_indices = keep_t                              # (N_keep,)
        n_before     = self.obs.shape[1]
        self.obs     = self.obs    [:, keep_t, :].contiguous()
        self.mask    = self.mask   [:, keep_t, :].contiguous()
        self.spatial = self.spatial[keep_t, :]   .contiguous()
        # obs_stats is intentionally kept at (N_full, V) for fast-cache
        # compatibility when val/test compare against train_ds.obs_stats.
        # Callers that need per-kept-station stats should index via _keep_indices.
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
        # Random strategy: return a fixed epoch size rather than the full pool
        # to avoid oversampling (the pool has high temporal overlap).
        if self.index_mode == "random" and self.split == "train":
            return self._random_epoch_size
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        # Strategy A ("random"): ignore DataLoader idx, sample uniformly from pool.
        # This gives per-item independent sampling with replacement — the idx
        # argument is used only to size the epoch (via __len__), not to select data.
        if self.index_mode == "random":
            i = self.indices[int(torch.randint(len(self.indices), (1,)).item())]
        else:
            i = self.indices[idx]
        W = self.window_size

        # ── Input window — shared across all K targets ─────────────────
        # .clone() gives each slice its own storage rather than a view of the
        # full backing tensor.  This is critical for the DataLoader's file_system
        # sharing strategy: without it, the entire obs storage (hundreds of MB)
        # may be written to /tmp for every batch instead of just the slice.
        x       = self.obs  [i : i + W].clone()   # (W, N, V)
        x_mask  = self.mask [i : i + W].clone()   # (W, N, V)
        x_hours = self.hours[i : i + W].clone()   # (W,)

        # ── Target snapshot(s) ─────────────────────────────────────────

        # ── Fixed-grid mode: always return the full horizon grid ───────
        # K targets at 0, stride, 2·stride, …, max_delta_steps steps.
        # E.g. stride=3, max=36 → [0, 3, 6, …, 36], K=13.
        # delta=0 is the reconstruction target (last input timestep).
        if self.delta_mode == "fixed_grid":
            ys, y_masks, y_hrs = [], [], []
            for dt in self.delta_grid:
                t_idx = i + W - 1 + dt
                # .clone() avoids sending the full backing tensor (mmap / shared
                # memory) through the DataLoader queue for every single timestep.
                ys.append(self.obs[t_idx].clone())
                y_masks.append(self.mask[t_idx].clone())
                y_hrs.append(self.hours[t_idx])
            return {
                "x":           x,
                "x_mask":      x_mask,
                "x_hours":     x_hours,
                "y":           torch.stack(ys),                                # (K, N, V)
                "y_mask":      torch.stack(y_masks),                           # (K, N, V)
                "y_hours":     torch.stack(y_hrs),                             # (K,)
                "spatial":     self.spatial,                                   # (N, 15)
                "delta_steps": torch.tensor(self.delta_grid, dtype=torch.long),  # (K,)
            }

        # ── Random delta mode (legacy / ablation) ─────────────────────
        K = self.num_delta_per_sample
        if K == 1:
            # Single lead-time per sample
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
                deltas, _ = (torch.randperm(max_dt)[:K] + 1).sort()
            else:
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
