"""
data/dataset.py

PyTorch dataset over the PeakWeather observations.

Each sample is a 12-hour input window and K target snapshots on a fixed
lead-time grid:

    x           (W, N, V)   normalised observations, 0 where absent
    x_mask      (W, N, V)   1 present / 0 absent
    x_hours     (W,)        hours since epoch per input step
    y           (K, N, V)   targets at leads 0, s, 2s, ..., max_delta steps
    y_mask      (K, N, V)
    y_hours     (K,)
    spatial     (N, 15)     normalised static station features
    delta_steps (K,)        lead times in 10-min steps

Splits: train 2017-2021, validation 2022, test 2023-2024. Normalisation is
per station and per variable with statistics from the training years only;
precipitation is log1p-transformed first.

Two caches are used: ``cache_dir`` holds the raw (T, N, V) tensors as one
.pt file (built once, minutes), ``fast_cache_dir`` holds the normalised
per-split arrays as .npy files that DataLoader workers memory-map.
"""

import math
import os
import time
import warnings

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from peakweather.dataset import PeakWeatherDataset


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

# Donor-station similarity for sparse (station, variable) pairs, see
# normalise_observations: elevation columns of the 15-d spatial vector and
# their weight relative to the other descriptors.
SIM_HEIGHT_IDX    = slice(6, 8)
SIM_HEIGHT_WEIGHT = 3.0

TRAIN_YEARS = list(range(2017, 2022))   # 2017-2021
VAL_YEARS   = [2022]
TEST_YEARS  = [2023, 2024]


def _hours_since_epoch(ts: pd.Timestamp) -> float:
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return float((ts - epoch).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# PeakWeather loader and preprocessing
# ---------------------------------------------------------------------------

def load_peakweather(root: str) -> PeakWeatherDataset:
    """Meteo stations only, 10-min frequency, wind as (u, v), no imputation."""
    return PeakWeatherDataset(
        root=root,
        parameters=["temperature", "pressure", "humidity",
                    "wind_speed", "wind_direction", "precipitation"],
        compute_uv=True,
        station_type="meteo_station",
        imputation_method=None,
        freq="10min",
    )


def build_spatial_features(ds: PeakWeatherDataset) -> tuple[torch.Tensor, dict]:
    """
    (N, 15) standardised static features per station: LV95 easting/northing,
    sin/cos of the two aspect angles, and the nine remaining descriptors of
    SPATIAL_FEATURE_NAMES. Returns (features, {"mean", "std"}).
    """
    rows = []
    for _, row in ds.stations_table.iterrows():
        a2  = math.radians(float(row["ASPECT_2000M_SIGRATIO1"]))
        a10 = math.radians(float(row["ASPECT_10000M_SIGRATIO1"]))
        rows.append([
            float(row["swiss_easting"]),
            float(row["swiss_northing"]),
            math.sin(a2),  math.cos(a2),
            math.sin(a10), math.cos(a10),
            float(row["station_height"]),
            float(row["dem"]),
            float(row["TPI_2000M"]),
            float(row["SLOPE_2000M_SIGRATIO1"]),
            float(row["SLOPE_10000M_SIGRATIO1"]),
            float(row["SN_DERIVATIVE_2000M_SIGRATIO1"]),
            float(row["SN_DERIVATIVE_10000M_SIGRATIO1"]),
            float(row["WE_DERIVATIVE_2000M_SIGRATIO1"]),
            float(row["WE_DERIVATIVE_10000M_SIGRATIO1"]),
        ])
    features = torch.tensor(rows, dtype=torch.float32)
    mean = features.mean(dim=0)
    std  = features.std(dim=0).clamp(min=1e-6)
    return (features - mean) / std, {"mean": mean, "std": std}


def build_observations(ds: PeakWeatherDataset):
    """
    Full (T, N, V) observation array for the six variables.

    Returns obs (0.0 where absent), mask (1.0 present / 0.0 absent), and the
    list of T timestamps.
    """
    raw      = ds.get_observations(parameters=VARIABLE_NAMES)
    stations = ds.stations_table.index.tolist()
    T, N, V  = len(raw), len(stations), NUM_VARIABLES

    obs = np.full((T, N, V), np.nan, dtype=np.float32)
    for v_idx, var in enumerate(VARIABLE_NAMES):
        for n_idx, stn in enumerate(stations):
            if (stn, var) in raw.columns:
                obs[:, n_idx, v_idx] = raw[(stn, var)].values.astype(np.float32)

    mask = (~np.isnan(obs)).astype(np.float32)
    obs  = np.nan_to_num(obs, nan=0.0)
    return torch.from_numpy(obs), torch.from_numpy(mask), raw.index.tolist()


def normalise_observations(
    obs:     torch.Tensor,
    mask:    torch.Tensor,
    coords:  "torch.Tensor | None" = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Per-(station, variable) standardisation using present values only.

    A pair with fewer than MIN_OBS observations borrows the statistics of the
    most similar station that has the variable. Similarity is the Euclidean
    distance in the standardised 15-d static-feature space (``coords``) with
    the two elevation columns weighted SIM_HEIGHT_WEIGHT times, because
    elevation drives the mean and variance of pressure and temperature. If
    no station has the variable, the network-wide statistics are used.

    Returns obs_norm (absent entries zeroed) and
    {"mean": (N, V), "std": (N, V), "donor": (N, V) long, "n_obs": (N, V)}
    where donor = -1 (own statistics), j >= 0 (donor station index) or -2
    (global fallback).
    """
    MIN_OBS = 50
    T, N, V = obs.shape

    raw_count = mask.sum(dim=0)
    count     = raw_count.clamp(min=1.0)
    means     = (obs * mask).sum(dim=0) / count
    stds      = ((((obs - means.unsqueeze(0)) ** 2) * mask).sum(dim=0)
                 / count.clamp(min=2.0)).sqrt()

    g_count = mask.sum(dim=(0, 1)).clamp(min=1)
    g_mean  = (obs * mask).sum(dim=(0, 1)) / g_count
    g_std   = ((((obs - g_mean[None, None, :]) ** 2) * mask).sum(dim=(0, 1))
               / g_count.clamp(min=2)).sqrt().clamp(min=1e-6)

    sparse = raw_count < MIN_OBS
    donor  = torch.full((N, V), -1, dtype=torch.long)

    if sparse.any():
        if coords is None:
            print(f"  [norm] {int(sparse.sum())} sparse (station, variable) pair(s) and no "
                  f"station features supplied: global statistics used")
            means = torch.where(sparse, g_mean.unsqueeze(0).expand_as(means), means)
            stds  = torch.where(sparse, g_std.unsqueeze(0).expand_as(stds), stds)
            donor = torch.where(sparse, torch.full_like(donor, -2), donor)
        else:
            c = coords.to(torch.float64)
            w = torch.ones(c.shape[1], dtype=torch.float64)
            w[SIM_HEIGHT_IDX] = SIM_HEIGHT_WEIGHT
            dist = torch.cdist(c * w, c * w)
            dist.fill_diagonal_(float("inf"))

            n_sub = 0
            for v in range(V):
                healthy = ~sparse[:, v]
                if not healthy.any():
                    sel = sparse[:, v]
                    means[sel, v], stds[sel, v], donor[sel, v] = g_mean[v], g_std[v], -2
                    continue
                d_v = dist.clone()
                d_v[:, ~healthy] = float("inf")
                nearest = d_v.argmin(dim=1)
                for i in torch.nonzero(sparse[:, v]).flatten().tolist():
                    j = int(nearest[i])
                    means[i, v], stds[i, v], donor[i, v] = means[j, v], stds[j, v], j
                    n_sub += 1
            if verbose and n_sub:
                print(f"  [norm] {n_sub} sparse (station, variable) pair(s) took the "
                      f"statistics of their most similar station")
            n_global = int((donor == -2).sum())
            if n_global:
                print(f"  [norm] {n_global} (station, variable) pair(s) had no donor: "
                      f"global statistics used")

    stds = stds.clamp(min=1e-6)
    obs_norm = ((obs - means.unsqueeze(0)) / stds.unsqueeze(0)) * mask
    return obs_norm, {"mean": means, "std": stds, "donor": donor, "n_obs": raw_count}


def compute_obs_stats(
    ds:          PeakWeatherDataset,
    obs_full:    "torch.Tensor | None" = None,
    mask_full:   "torch.Tensor | None" = None,
    timestamps:  "list | None"         = None,
    train_years: "list[int] | None"    = None,
) -> dict:
    """Normalisation statistics from the training years (see normalise_observations)."""
    if obs_full is None or mask_full is None or timestamps is None:
        obs_full, mask_full, timestamps = build_observations(ds)

    years     = train_years if train_years is not None else TRAIN_YEARS
    train_idx = [i for i, ts in enumerate(timestamps) if ts.year in years]

    coords = None
    try:
        coords, _ = build_spatial_features(ds)
        if coords.shape[0] != obs_full.shape[1]:
            print(f"  [norm] stations_table has {coords.shape[0]} rows but obs has "
                  f"{obs_full.shape[1]} stations: similar-station lookup disabled")
            coords = None
    except Exception as e:  # noqa: BLE001
        print(f"  [norm] could not build station features ({type(e).__name__}): "
              f"similar-station lookup disabled")

    _, stats = normalise_observations(obs_full[train_idx], mask_full[train_idx], coords=coords)
    return stats


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_CACHE_FILENAME = "peakweather_obs_cache.pt"

# Bump when the content of the fast cache changes (it stores normalised values).
_FAST_CACHE_VERSION = "v6"


def _fast_split_paths(fast_dir, split, train_years, exclude_stations=None) -> dict:
    years_key = "_".join(str(y) for y in sorted(train_years))
    excl_key  = ("_x" + "-".join(sorted(str(e) for e in exclude_stations))) if exclude_stations else ""
    prefix    = f"{split}_{years_key}_ps{excl_key}"
    return {
        "obs":     os.path.join(fast_dir, f"{prefix}_obs.npy"),
        "mask":    os.path.join(fast_dir, f"{prefix}_mask.npy"),
        "hours":   os.path.join(fast_dir, f"{prefix}_hours.npy"),
        "spatial": os.path.join(fast_dir, "spatial.pt"),
        "meta":    os.path.join(fast_dir, f"{prefix}_meta.pt"),
    }


def fast_cache_save(fast_dir, split, train_years, obs_norm, mask, hours, spatial,
                    spatial_stats, obs_stats, indices, window_size, max_delta_steps,
                    exclude_stations=None) -> None:
    """Write one split's normalised arrays as .npy files plus a metadata .pt."""
    os.makedirs(fast_dir, exist_ok=True)
    paths = _fast_split_paths(fast_dir, split, train_years, exclude_stations)
    t0 = time.time()
    np.save(paths["obs"],   obs_norm.numpy())
    np.save(paths["mask"],  mask.numpy())
    np.save(paths["hours"], hours.numpy())
    if not os.path.exists(paths["spatial"]):
        torch.save(spatial, paths["spatial"])
    torch.save({
        "version":           _FAST_CACHE_VERSION,
        "obs_stats":         obs_stats,
        "spatial_stats":     spatial_stats,
        "all_valid_indices": indices,     # window pool before index_mode is applied
        "window_size":       window_size,
        "max_delta_steps":   max_delta_steps,
    }, paths["meta"])
    print(f"[FastCache] Saved '{split}' to {fast_dir} in {time.time() - t0:.1f}s", flush=True)


def fast_cache_load(fast_dir, split, train_years, exclude_stations=None) -> "dict | None":
    """Memory-map one split's arrays; None if absent or outdated."""
    paths = _fast_split_paths(fast_dir, split, train_years, exclude_stations)
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    meta = torch.load(paths["meta"], weights_only=False)
    if meta.get("version") != _FAST_CACHE_VERSION:
        print(f"[FastCache] Version mismatch, rebuilding '{split}'.", flush=True)
        return None
    t0 = time.time()
    out = {
        "obs":           np.load(paths["obs"],   mmap_mode="c"),
        "mask":          np.load(paths["mask"],  mmap_mode="c"),
        "hours":         np.load(paths["hours"], mmap_mode="c"),
        "spatial":       torch.load(paths["spatial"], weights_only=False),
        "obs_stats":     meta["obs_stats"],
        "spatial_stats": meta["spatial_stats"],
        "meta":          meta,
    }
    print(f"[FastCache] Loaded '{split}' from {fast_dir} in {time.time() - t0:.2f}s  "
          f"obs={out['obs'].shape}", flush=True)
    return out


def _load_or_build_cache(ds: PeakWeatherDataset, cache_dir: str):
    """Raw (T, N, V) tensors and static features, cached as one .pt file."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _CACHE_FILENAME)

    if os.path.exists(cache_path):
        t0 = time.time()
        payload = torch.load(cache_path, weights_only=False)
        print(f"[Cache] Loaded {cache_path} in {time.time() - t0:.1f}s  "
              f"(obs {tuple(payload['obs_full'].shape)})", flush=True)
        return (payload["obs_full"], payload["mask_full"], payload["timestamps"],
                payload["spatial"], payload["spatial_stats"])

    print("[Cache] First run: building tensors from the raw data (several minutes)", flush=True)
    t0 = time.time()
    obs_full, mask_full, timestamps = build_observations(ds)
    spatial, spatial_stats          = build_spatial_features(ds)
    torch.save({"obs_full": obs_full, "mask_full": mask_full, "timestamps": timestamps,
                "spatial": spatial, "spatial_stats": spatial_stats}, cache_path)
    print(f"[Cache] Built in {time.time() - t0:.1f}s, saved to {cache_path}", flush=True)
    return obs_full, mask_full, timestamps, spatial, spatial_stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StationMAEDataset(Dataset):
    """
    Args:
        ds:                PeakWeatherDataset from ``load_peakweather``.
        window_size:       W input steps (72 = 12 h).
        max_delta_steps:   longest lead in 10-min steps (36 = 6 h).
        delta_grid_stride: lead-time spacing in steps (3 = 30 min); the grid is
                           0, stride, ..., max_delta_steps (K = 13 by default).
        split:             "train", "val" or "test".
        obs_stats:         normalisation statistics; None computes them from
                           the training years. Val/test must receive the
                           training dataset's ``obs_stats``.
        cache_dir:         directory of the raw-tensor cache.
        fast_cache_dir:    directory of the memory-mapped per-split cache (optional).
        train_years:       training years (default TRAIN_YEARS).
        exclude_stations:  station abbreviations dropped from every split.
        index_mode:        window selection on the training split:
                           "sliding" (every valid start, thinned by ``train_stride``),
                           "blocks" (non-overlapping windows), or
                           "random" (uniform sampling with replacement from the
                           pool, ``random_epoch_size`` samples per epoch).
                           Val/test use "sliding" (with ``train_stride``) or "blocks".
        train_stride:      thinning step for "sliding".
        random_epoch_size: epoch length for "random".
    """

    def __init__(
        self,
        ds:                PeakWeatherDataset,
        window_size:       int  = 72,
        max_delta_steps:   int  = 36,
        delta_grid_stride: int  = 3,
        split:             str  = "train",
        obs_stats:         "dict | None" = None,
        cache_dir:         "str | None" = None,
        fast_cache_dir:    "str | None" = None,
        train_years:       "list[int] | None" = None,
        exclude_stations:  "list | None" = None,
        index_mode:        str  = "sliding",
        train_stride:      int  = 1,
        random_epoch_size: "int | None" = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), split
        assert index_mode in ("sliding", "blocks", "random"), index_mode
        assert train_stride >= 1 and delta_grid_stride >= 1

        self.window_size       = window_size
        self.max_delta_steps   = max_delta_steps
        self.delta_grid_stride = delta_grid_stride
        self.split             = split
        self.index_mode        = index_mode
        self.train_stride      = train_stride
        self.delta_grid        = list(range(0, max_delta_steps + 1, delta_grid_stride))
        self._keep_indices     = None

        train_years = train_years if train_years is not None else TRAIN_YEARS

        # ── Fast path: memory-mapped per-split cache ─────────────────────
        if fast_cache_dir is not None:
            cached = fast_cache_load(fast_cache_dir, split, train_years, exclude_stations)
            ok = (cached is not None
                  and cached["meta"].get("window_size") == window_size
                  and cached["meta"].get("max_delta_steps") == max_delta_steps)
            if ok and obs_stats is not None:
                ok = (torch.allclose(cached["obs_stats"]["mean"], obs_stats["mean"])
                      and torch.allclose(cached["obs_stats"]["std"], obs_stats["std"]))
            if ok:
                self.obs_stats     = cached["obs_stats"]
                self.spatial       = cached["spatial"]
                self.spatial_stats = cached["spatial_stats"]
                pool = cached["meta"]["all_valid_indices"]
                self.indices = self._apply_index_mode(pool, window_size, train_stride,
                                                      index_mode, split)
                self._random_epoch_size = (random_epoch_size if random_epoch_size is not None
                                           else max(1, len(pool) // window_size))
                self._obs_np, self._mask_np, self._hours_np = cached["obs"], cached["mask"], cached["hours"]
                self.obs   = torch.from_numpy(self._obs_np)
                self.mask  = torch.from_numpy(self._mask_np)
                self.hours = torch.from_numpy(self._hours_np)
                if exclude_stations:
                    self._apply_station_exclusion(
                        self._resolve_keep_indices(ds, exclude_stations), exclude_stations)
                return

        # ── Raw tensors ──────────────────────────────────────────────────
        if cache_dir is not None:
            obs_full, mask_full, timestamps_full, spatial, spatial_stats = \
                _load_or_build_cache(ds, cache_dir)
        else:
            obs_full, mask_full, timestamps_full = build_observations(ds)
            spatial, spatial_stats               = build_spatial_features(ds)
        self.spatial, self.spatial_stats = spatial, spatial_stats

        # Precipitation: log1p before any statistic is computed (0 stays 0).
        p_idx = VARIABLE_NAMES.index("precipitation")
        obs_full = obs_full.clone()
        obs_full[:, :, p_idx] = torch.log1p(obs_full[:, :, p_idx].clamp(min=0.0))

        # ── Normalisation statistics from the training years only ────────
        if obs_stats is None:
            self.obs_stats = compute_obs_stats(ds, obs_full, mask_full, timestamps_full,
                                               train_years=train_years)
        else:
            self.obs_stats = obs_stats

        # ── Slice to the split's years and normalise ─────────────────────
        keep_years = {"train": train_years, "val": VAL_YEARS, "test": TEST_YEARS}[split]
        split_idx  = [i for i, ts in enumerate(timestamps_full) if ts.year in keep_years]
        assert split_idx, f"no timesteps for split='{split}' with years={keep_years}"

        obs        = obs_full[split_idx]
        mask       = mask_full[split_idx]
        timestamps = [timestamps_full[i] for i in split_idx]

        mean, std = self.obs_stats["mean"], self.obs_stats["std"]
        self.obs   = ((obs - mean.unsqueeze(0)) / std.unsqueeze(0)) * mask
        self.mask  = mask
        self.hours = torch.tensor([_hours_since_epoch(ts) for ts in timestamps],
                                  dtype=torch.float32)
        self.timestamps = timestamps

        # ── Valid window starts: the span [i, i+W-1+max_delta] has no gap ─
        T        = len(timestamps)
        step_ns  = int(pd.Timedelta("10min").value)
        diffs    = np.diff(pd.DatetimeIndex(timestamps).asi8)
        bad      = (diffs != step_ns).astype(np.int32)
        bad_cum  = np.zeros(T + 1, dtype=np.int32)
        bad_cum[1:T] = np.cumsum(bad)
        need     = window_size - 1 + max_delta_steps
        max_i    = max(0, T - window_size - max_delta_steps)
        all_valid = [i for i in range(max_i) if bad_cum[i + need] == bad_cum[i]]

        self.indices = self._apply_index_mode(all_valid, window_size, train_stride,
                                              index_mode, split)
        self._random_epoch_size = (random_epoch_size if random_epoch_size is not None
                                   else max(1, len(all_valid) // window_size))

        if fast_cache_dir is not None:
            fast_cache_save(fast_cache_dir, split, train_years, self.obs, self.mask,
                            self.hours, spatial, spatial_stats, self.obs_stats, all_valid,
                            window_size, max_delta_steps, exclude_stations=exclude_stations)

        if exclude_stations:
            self._apply_station_exclusion(
                self._resolve_keep_indices(ds, exclude_stations), exclude_stations)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_keep_indices(ds: PeakWeatherDataset, exclude_stations: list) -> list:
        """
        Station positions to keep after dropping ``exclude_stations``. Entries
        are matched case-insensitively against the stations_table index and
        its name / abbreviation columns.
        """
        stns  = ds.stations_table
        excl  = {str(s).upper() for s in exclude_stations}
        drop  = set()
        for pos, (idx, row) in enumerate(stns.iterrows()):
            cands = {str(idx).upper()}
            try:
                cands.add(str(int(float(str(idx)))).upper())
            except (ValueError, TypeError):
                pass
            for col in ("name", "abbr", "nat_abbr", "station_name"):
                if col in row.index and not pd.isna(row[col]):
                    cands.add(str(row[col]).upper())
            if cands & excl:
                drop.add(pos)
        if not drop:
            warnings.warn(f"No stations matched {exclude_stations}; nothing excluded.",
                          UserWarning, stacklevel=3)
        return [i for i in range(len(stns)) if i not in drop]

    @staticmethod
    def _apply_index_mode(all_valid, window_size, train_stride, index_mode, split) -> list:
        """Select window starts from the gap-free pool (see class docstring)."""
        def _blocks(offset: int) -> list:
            out, next_ok = [], offset
            for i in all_valid:
                if i >= next_ok:
                    out.append(i)
                    next_ok = i + window_size
            return out

        if split != "train":
            if index_mode == "blocks":
                # Test blocks are offset by W/4 (3 h) so that val and test
                # blocks start at different hours of the day.
                return _blocks(0 if split == "val" else window_size // 4)
            return list(all_valid)[::train_stride]

        if index_mode == "random":
            return list(all_valid)
        if index_mode == "blocks":
            return _blocks(0)
        return list(all_valid)[::train_stride]

    def _apply_station_exclusion(self, keep: list, exclude_stations: list) -> None:
        """Drop excluded stations from obs, mask and spatial (obs_stats stays at N_full)."""
        keep_t = torch.tensor(keep, dtype=torch.long)
        self._keep_indices = keep_t
        n_before     = self.obs.shape[1]
        self.obs     = self.obs[:, keep_t, :].contiguous()
        self.mask    = self.mask[:, keep_t, :].contiguous()
        self.spatial = self.spatial[keep_t, :].contiguous()
        if n_before - len(keep) > 0:
            print(f"[StationFilter] Excluded {n_before - len(keep)} station(s) "
                  f"{exclude_stations}: N={len(keep)} (was {n_before})", flush=True)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.index_mode == "random" and self.split == "train":
            return self._random_epoch_size
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        if self.index_mode == "random":
            i = self.indices[int(torch.randint(len(self.indices), (1,)).item())]
        else:
            i = self.indices[idx]
        W = self.window_size

        # .clone(): slices of the memory-mapped arrays must not be sent
        # through the DataLoader queue as views of the whole array.
        x       = self.obs[i:i + W].clone()
        x_mask  = self.mask[i:i + W].clone()
        x_hours = self.hours[i:i + W].clone()

        ys, y_masks, y_hrs = [], [], []
        for dt in self.delta_grid:
            t_idx = i + W - 1 + dt
            ys.append(self.obs[t_idx].clone())
            y_masks.append(self.mask[t_idx].clone())
            y_hrs.append(self.hours[t_idx])

        return {
            "x":           x,
            "x_mask":      x_mask,
            "x_hours":     x_hours,
            "y":           torch.stack(ys),
            "y_mask":      torch.stack(y_masks),
            "y_hours":     torch.stack(y_hrs),
            "spatial":     self.spatial,
            "delta_steps": torch.tensor(self.delta_grid, dtype=torch.long),
        }
