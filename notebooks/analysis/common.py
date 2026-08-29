"""
Shared utilities for the analysis notebooks (notebooks/analysis/01..09).

Design rules
------------
* The repository is the source of truth: variable order, station order, lead
  grid and masks are all READ from the dumps / dataset code, never assumed.
* Baselines (persistence, 24h-lag) are computed on RAW observations from the
  parquet files — no normalise/denormalise round trip.
* Model errors are accumulated in BOTH normalised and physical units; the
  per-station training std converts between them, and correlation needs
  neither (it is affine-invariant).
* Every station×variable exclusion (GES pressure, LAE wind, PFA) is applied
  exactly once, inside `aggregate_run`, so all notebooks inherit it.
* Heavy passes stream the 1.6-2 GB dumps in chunks and cache small summary
  arrays under analysis_outputs/cache/. Delete the cache to force a rebuild.
"""
from __future__ import annotations

import datetime
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))          # repository root
SRC = os.path.join(PROJ, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RESULTS_ROOT = os.path.join(PROJ, "test_results")
OUT = os.path.join(PROJ, "analysis_outputs")
CACHE = os.path.join(OUT, "cache")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
for _d in (OUT, CACHE, FIG, TAB):
    os.makedirs(_d, exist_ok=True)

_DATA_CANDIDATES = [
    "/home/renku/work/PeakWeatherDataset",
    os.path.expanduser("~/Documents/ETH/_DAS Project/PeakWeatherDataset"),
    os.path.join(PROJ, "notebooks", "PeakWeatherDataset"),
    os.path.expanduser("~/PeakWeatherDataset"),
]

# Every year an intact observations file must have; sizes are ~45-58 MB.
_OBS_YEARS = list(range(2017, 2025))
_MIN_OBS_BYTES = 40_000_000


def _root_report(root: str) -> tuple[bool, list[str]]:
    """Validate one dataset root: existence, per-year size, parquet footer."""
    problems = []
    if not os.path.isdir(os.path.join(root, "observations")):
        return False, [f"no observations/ under {root}"]
    for y in _OBS_YEARS:
        p = os.path.join(root, "observations", f"{y}.parquet")
        if not os.path.isfile(p):
            problems.append(f"{y}.parquet missing")
            continue
        size = os.path.getsize(p)
        if size < _MIN_OBS_BYTES:
            problems.append(f"{y}.parquet truncated ({size/1e6:.1f} MB)")
            continue
        with open(p, "rb") as f:                       # parquet footer magic
            f.seek(-4, 2)
            if f.read(4) != b"PAR1":
                problems.append(f"{y}.parquet corrupt (bad footer)")
    for f_ in ("stations.parquet", "parameters.parquet"):
        if not os.path.isfile(os.path.join(root, f_)):
            problems.append(f"{f_} missing")
    return not problems, problems


_DATA_ROOT_CACHE: str | None = None


def data_root(verbose: bool = False) -> str:
    """First candidate root whose files all pass integrity checks.

    This exists because one local copy shipped with a truncated 2018.parquet
    (19.5 MB instead of ~57 MB). 2018 is a TRAINING year, so silently using
    that copy would corrupt every per-station normalisation statistic.
    """
    global _DATA_ROOT_CACHE
    if _DATA_ROOT_CACHE is not None:
        return _DATA_ROOT_CACHE
    reports = []
    for cand in _DATA_CANDIDATES:
        ok, problems = _root_report(cand)
        reports.append((cand, ok, problems))
        if ok:
            if verbose:
                for c, o, pr in reports:
                    flag = "OK " if o else "BAD"
                    print(f"  [{flag}] {c}" + ("" if o else f"  -> {pr}"))
            _DATA_ROOT_CACHE = cand
            return cand
    lines = [f"  {c}: {pr}" for c, o, pr in reports]
    raise FileNotFoundError(
        "No intact PeakWeatherDataset copy found. Candidates:\n" + "\n".join(lines)
        + "\nFix: copy an intact 2018.parquet over the truncated one, e.g.\n"
        "  cp ~/Documents/ETH/_DAS\\ Project/PeakWeatherDataset/observations/"
        "2018.parquet <project>/notebooks/PeakWeatherDataset/observations/"
    )


# ---------------------------------------------------------------------------
# Model registry — verified against checkpoints / training scripts
# ---------------------------------------------------------------------------
MODELS = {
    # run dir            label                              colour     notes
    "lstm-baseline-v1": ("LSTM Baseline",                  "#7B5EA7",
                         "per-station recurrence, spatially blind, no masking"),
    "v27":              ("MAE Transformer",                "#1F5F6B",
                         "spatial attention, trained mask_ratio 0.5, Huber loss"),
    "v30-nll":          ("Probabilistic MAE Transformer",  "#E1A730",
                         "MAE Transformer + heteroscedastic Gaussian NLL head (log_var saved)"),
    "v31":              ("Dense Transformer",               "#D9663D",
                         "MAE Transformer trained at mask_ratio 0"),
    "v32-blind":        ("Spatially Blind Transformer",     "#5FAF5F",
                         "no encoder spatial attn + station-local decoder, mr0"),
}
BASELINE_COLORS = {"persistence": "#888888", "clim": "#CCCCCC"}
# Display names for the non-learned baselines. "clim" is the code's internal
# name (see CLIM_LAG_STEPS below) for what is actually a 24h-lagged
# observation, not a multi-year climatological average - the label reflects
# what it measures, not the variable name.
BASELINE_LABELS = {"persistence": "Last-Value Persistence", "clim": "24h Persistence"}

EXCLUDE = ["PFA"]                       # dropped from the network entirely
DROP_SV = {"GES": ["pressure"], "LAE": ["wind_u", "wind_v"]}

UNITS = {"temperature": "°C", "pressure": "hPa", "humidity": "%",
         "wind_u": "m/s", "wind_v": "m/s"}

CLIM_LAG_STEPS = 144                    # 24 h on the 10-min grid
STEP_H = 1.0 / 6.0                      # one grid step in hours


def discovered_runs() -> dict:
    """{run: {"mr0.00": path, ...}} from what is actually on disk."""
    runs: dict = {}
    for p in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*", "best_mr*",
                                           "predictions.pt"))):
        run = p.split(os.sep)[-3]
        mr = p.split(os.sep)[-2].replace("best_", "")
        runs.setdefault(run, {})[mr] = p
    return runs


def load_dump(run: str, mr: str = "mr0.00"):
    """torch.load one predictions.pt; returns the raw dict (lazy nothing)."""
    import torch
    path = discovered_runs()[run][mr]
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Station table (cached) — order VERIFIED against the dumps' spatial matrix
# ---------------------------------------------------------------------------
def station_table(force: bool = False) -> pd.DataFrame:
    """
    One row per modelled station, in DUMP ORDER (index 0..N-1). Columns:
    abbr, name, latitude, longitude, height, slope, plus density metrics
    (nn_dist_km, n_within_30km) from Swiss LV95 coordinates.

    The order is verified by rebuilding the (N, 15) spatial-feature matrix
    exactly as src/data/dataset.py does and matching it against the matrix
    stored in a dump. Any mismatch raises — a silently wrong station order
    would invalidate every per-station number downstream.
    """
    cache = os.path.join(CACHE, "station_table.parquet")
    # Columns this function is contracted to provide. A cache written by an
    # older version silently lacks new ones and surfaces as a KeyError three
    # notebooks away, so check the schema rather than just the file's presence.
    required = {"abbr", "name", "latitude", "longitude", "height", "slope",
                "easting", "northing", "nn_dist_km", "n_within_30km",
                "n_within_50km", "dem", "rel_height", "relief_2km",
                "relief_10km", "tpi_n2", "tpi_n10", "plateau_dist_km",
                "region", "landform", "terrain_class"}
    if os.path.isfile(cache) and not force:
        cached = pd.read_parquet(cache)
        if required.issubset(cached.columns):
            return cached
        print(f"station_table: cache missing "
              f"{sorted(required - set(cached.columns))} — rebuilding")

    from data.dataset import load_peakweather, StationMAEDataset
    ds = load_peakweather(root=data_root())
    keep = StationMAEDataset._resolve_keep_indices(ds, EXCLUDE)
    stn_ids = [ds.stations_table.index[i] for i in keep]
    t = ds.stations_table.loc[stn_ids]

    # verify against a dump
    runs = discovered_runs()
    first = next(iter(next(iter(runs.values())).values()))
    import torch
    sp = torch.load(first, map_location="cpu",
                    weights_only=False)["spatial"].numpy().astype(np.float64)
    from data.dataset import build_spatial_features
    mine, _ = build_spatial_features(ds)
    mine = mine.numpy().astype(np.float64)[keep]
    dev = np.abs(mine - sp).max()
    assert dev < 1e-3, (
        f"station order mismatch vs dump spatial features (max dev {dev:.2e})")

    e = t["swiss_easting"].astype(float).values
    n = t["swiss_northing"].astype(float).values
    d2 = np.sqrt((e[:, None] - e[None, :]) ** 2
                 + (n[:, None] - n[None, :]) ** 2) / 1000.0     # km
    np.fill_diagonal(d2, np.inf)
    # ── Terrain classification ──────────────────────────────────────────
    # Two independent axes, deliberately kept separate.
    #
    # SHAPE — SD-normalised TPI: TPI_10000M / STD_10000M. Dividing by the
    # LOCAL relief is what makes it altitude-free: a 300 m depression in the
    # Alps and a 30 m dip on the Plateau both read as "valley" relative to
    # their own surroundings. Standardising raw TPI across stations instead
    # put Sion (482 m) and Magadino (203 m) — textbook deep valley floors —
    # in the "open/flat" class.
    #
    # ELEVATION — bands cut on physics, not quantiles. Switzerland's winter
    # inversion caps near 800-1200 m. BELOW it a valley floor decouples from
    # the regional field at night (cold-air pooling, fog); ABOVE it an
    # enclosed site sits above the pool and couples to the free atmosphere,
    # often warmer than the valley below. The same TPI signature means
    # opposite things on either side, so shape alone cannot define the class:
    # without the split, one group ran from Magadino (203 m) to Monte Rosa
    # (2885 m).
    _h    = t["station_height"].astype(float).values
    _dem  = t["dem"].astype(float).values
    _r2   = t["STD_2000M"].astype(float).values
    _r10  = t["STD_10000M"].astype(float).values
    _tpi2 = t["TPI_2000M"].astype(float).values / np.maximum(_r2, 1e-6)
    _tpi10 = t["TPI_10000M"].astype(float).values / np.maximum(_r10, 1e-6)
    _slp  = t["SLOPE_2000M_SIGRATIO1"].astype(float).values

    # Physiographic region: signed perpendicular distance from the Plateau
    # axis (Geneva -> Lake Constance) in LV95 metres. Positive = NW (Jura).
    _p0 = np.array([2_500_000.0, 1_118_000.0])
    _p1 = np.array([2_745_000.0, 1_265_000.0])
    _d  = (_p1 - _p0) / np.linalg.norm(_p1 - _p0)
    _nv = np.array([-_d[1], _d[0]])
    _s  = ((np.c_[e, n] - _p0) @ _nv) / 1000.0
    region = np.where(_s > 40, "Jura",
             np.where(_s > -35, "Plateau",
             np.where(_s > -95, "Alps", "South. Alps")))
    _valley, _ridge = _tpi10 < -0.75, _tpi10 > 0.75
    landform = np.where(_valley, "valley/basin",
               np.where(_ridge, "ridge/summit",
               np.where(_slp > np.median(_slp), "midslope", "open/flat")))
    _alp = np.isin(region, ["Alps", "South. Alps"])

    # ── Four terrain classes, from TWO criteria ─────────────────────────────
    # Deliberately minimal: normalised TPI (shape) and one height threshold.
    # An earlier eight-class scheme split further by region and by a second
    # height band, but with 155 stations that leaves groups too small to test
    # and multiplies the comparisons for no extra mechanism.
    #
    #   valley floor          enclosed AND below the winter inversion cap:
    #                         cold-air pooling, fog, decoupled from the
    #                         regional field at night
    #   elevated enclosed     the same shape ABOVE the cap: sits above the
    #                         cold pool, couples to the free atmosphere.
    #                         Same TPI as the valley floors — the ONLY
    #                         difference is height, which is what makes the
    #                         pair a controlled test of the inversion process
    #                         against the enclosure geometry.
    #   exposed ridge/summit  free atmosphere, synoptically driven
    #   open / slope          everything else: plateau, gentle terrain
    #
    # The 900 m cut is physical, not a quantile: Switzerland's winter
    # inversion caps near 800-1200 m. Counts are stable across that range
    # (43/45/50 valley-floor stations at 800/900/1000 m).
    tclass = np.where(_tpi10 > 0.75, "exposed ridge/summit",
             np.where(_tpi10 < -0.75,
                      np.where(_h < 900, "valley floor", "elevated enclosed"),
                      "open / slope"))

    out = pd.DataFrame({
        "abbr": stn_ids,
        "name": t["station_name"].values if "station_name" in t else stn_ids,
        "latitude": t["latitude"].astype(float).values,
        "longitude": t["longitude"].astype(float).values,
        "height": t["station_height"].astype(float).values,
        "slope": t["SLOPE_2000M_SIGRATIO1"].astype(float).values,
        # Swiss LV95 metres. Kept as columns, not just consumed above: any
        # pairwise-distance work downstream needs a metric CRS, and degrees
        # would need a cos(latitude) correction to be usable as distance.
        "easting": e,
        "northing": n,
        "dem": _dem,
        "rel_height": _h - _dem,          # sensor vs modelled surface
        "relief_2km": _r2,
        "relief_10km": _r10,
        "tpi_n2": _tpi2,                  # continuous, altitude-free shape
        "tpi_n10": _tpi10,
        "plateau_dist_km": _s,
        "region": region,
        "landform": landform,
        "terrain_class": tclass,
        "nn_dist_km": d2.min(axis=1),
        "n_within_30km": (d2 < 30.0).sum(axis=1),
        "n_within_50km": (d2 < 50.0).sum(axis=1),
    })
    # A failed cache write must not fail the analysis: the frame is already
    # correct in memory, and the cache is only an optimisation.
    try:
        out.to_parquet(cache)
    except Exception as exc:                       # read-only dir, locked file
        print(f"station_table: could not write cache ({exc}); continuing")
    return out


def norm_stats(force: bool = False) -> dict:
    """Per-station training mean/std for the 5 target variables, (N, 5)."""
    cache = os.path.join(CACHE, "norm_stats.npz")
    if os.path.isfile(cache) and not force:
        z = np.load(cache, allow_pickle=True)
        return {"mean": z["mean"], "std": z["std"],
                "var_names": list(z["var_names"])}
    from data.dataset import (load_peakweather, StationMAEDataset,
                              compute_obs_stats)
    from model.embeddings import TARGET_VARIABLE_NAMES
    ds = load_peakweather(root=data_root())
    keep = StationMAEDataset._resolve_keep_indices(ds, EXCLUDE)
    st = compute_obs_stats(ds, train_years=None, per_station=True)
    nv = len(TARGET_VARIABLE_NAMES)
    mean = st["mean"].numpy()[keep][:, :nv]
    std = np.clip(st["std"].numpy()[keep][:, :nv], 1e-6, None)
    np.savez(cache, mean=mean, std=std,
             var_names=np.array(TARGET_VARIABLE_NAMES, dtype=object))
    return {"mean": mean, "std": std, "var_names": list(TARGET_VARIABLE_NAMES)}


def keep_mask(stn: pd.DataFrame, var_names: list) -> np.ndarray:
    """(N, V) bool — False where a station×variable pair is excluded."""
    keep = np.ones((len(stn), len(var_names)), bool)
    abbrs = list(stn["abbr"])
    for s, spec in DROP_SV.items():
        if s not in abbrs:
            continue
        si = abbrs.index(s)
        for v in (var_names if spec == "all" else spec):
            keep[si, var_names.index(v)] = False
    return keep


# ---------------------------------------------------------------------------
# Raw observations over the test window (for baselines / regimes)
# ---------------------------------------------------------------------------
def raw_test_obs(force: bool = False) -> dict:
    """
    Raw PHYSICAL observations restricted to the test years plus 24 h of
    run-up, in dump station order:
        obs   (Tc, N, 5) float32,  mask (Tc, N, 5) bool,  h0 (scalar), nt
    plus a `row(hours)` mapping from hours-since-epoch to row index.
    """
    cache = os.path.join(CACHE, "raw_test_obs.npz")
    if os.path.isfile(cache) and not force:
        z = np.load(cache)
        return {"obs": z["obs"], "mask": z["mask"],
                "h0": float(z["h0"]), "nt": int(z["nt"])}
    from data.dataset import (load_peakweather, StationMAEDataset,
                              build_observations, TEST_YEARS)
    ds = load_peakweather(root=data_root())
    keep = StationMAEDataset._resolve_keep_indices(ds, EXCLUDE)
    obs, msk, ts = build_observations(ds)
    ts = pd.DatetimeIndex(ts)
    sel = np.where(ts.year.isin(TEST_YEARS))[0]
    lo = max(0, int(sel[0]) - CLIM_LAG_STEPS)
    hi = int(sel[-1]) + 1
    o = obs[lo:hi][:, keep, :5].numpy().astype(np.float32)
    m = msk[lo:hi][:, keep, :5].numpy() > 0.5
    # resolution-agnostic (parquet indexes can be datetime64[ns] OR [us])
    step = np.unique(np.diff(ts[lo:hi]) / pd.Timedelta("1s"))
    assert len(step) == 1 and step[0] == 600.0, \
        f"observation grid not regular 10-min: {step} s"
    h0 = (ts[lo] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1h")
    np.savez(cache, obs=o, mask=m, h0=float(h0), nt=len(o))
    return {"obs": o, "mask": m, "h0": float(h0), "nt": len(o)}


def hours_to_row(hours, h0: float) -> np.ndarray:
    return np.rint((np.asarray(hours, np.float64) - h0) * 6.0).astype(np.int64)


# ---------------------------------------------------------------------------
# Aggregation — one streaming pass per dump, cached
# ---------------------------------------------------------------------------
_ESTIMATORS = ("mod", "per", "clim")
_SUBSETS = ("all", "msk", "vis")


def _season_of(hours: np.ndarray) -> np.ndarray:
    ep = datetime.datetime(1970, 1, 1)
    m = np.array([(ep + datetime.timedelta(hours=float(h))).month
                  for h in hours.ravel()])
    s = np.select([np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]),
                   np.isin(m, [6, 7, 8])], [0, 1, 2], 3)
    return s.reshape(hours.shape)


def aggregate_run(run: str, mr: str = "mr0.00", chunk: int = 1000,
                  force: bool = False) -> str:
    """
    Stream one dump and cache summary accumulators. Returns the cache path.

    Accumulators (all float64, all with matching *_cnt):
      {est}_{sub}_sum_phys / _sum_norm / _sumsq_phys / _signed_phys, at
      (K, N, V); est in mod/per/clim; sub in all/msk/vis (msk/vis only when
      the dump has masked stations).
      corr sums c_sp, c_st, c_spp, c_stt, c_spt at (K, N, V) per subset (mod).
      day-block sums day_{est}_sum_phys / day_{est}_cnt at (D, K, V).
      season sums sea_mod_sum_phys / sea_mod_cnt at (4, K, V).
    Baselines are computed from RAW observations (physical units, no
    normalisation); model errors from the dump (normalised) and converted
    with the per-station training std.
    """
    import torch
    tag = f"agg_{run}_{mr}.npz"
    path = os.path.join(CACHE, tag)
    if os.path.isfile(path) and not force:
        return path

    stn = station_table()
    ns = norm_stats()
    VARS = ns["var_names"]
    STD = ns["std"]                                   # (N, 5)
    raw = raw_test_obs()
    OBS, OMSK, H0, NT = raw["obs"], raw["mask"], raw["h0"], raw["nt"]
    KEEP = keep_mask(stn, VARS)

    d = load_dump(run, mr)
    P, T, M = d["preds"], d["targets"], d["masks"]
    TH = d["target_hours"]
    MI = d.get("masked_idx")
    Mw, K, N, _ = P.shape
    NV = len(VARS)
    assert list(d["var_names"]) == VARS, (
        f"variable order mismatch: dump {d['var_names']} vs stats {VARS}")
    grid = d["delta_steps"][0].numpy().astype(int)
    assert grid[0] == 0, "lead 0 must be first for persistence = obs(t0)"

    has_mask = MI is not None and MI.shape[1] > 0
    subs = _SUBSETS if has_mask else ("all",)

    acc: dict = {"grid": grid, "n_windows": np.array(Mw)}
    day0 = None
    D = 800                                            # generous day capacity
    for est in _ESTIMATORS:
        for sub in subs:
            for a in ("sum_phys", "sum_norm", "sumsq_phys", "signed_phys"):
                acc[f"{est}_{sub}_{a}"] = np.zeros((K, N, NV))
            acc[f"{est}_{sub}_cnt"] = np.zeros((K, N, NV))
        acc[f"day_{est}_sum_phys"] = np.zeros((D, K, NV))
        acc[f"day_{est}_cnt"] = np.zeros((D, K, NV))
    for sub in subs:                                   # correlation (model)
        for a in ("c_sp", "c_st", "c_spp", "c_stt", "c_spt"):
            acc[f"{a}_{sub}"] = np.zeros((K, N, NV))
    acc["sea_mod_sum_phys"] = np.zeros((4, K, NV))
    acc["sea_mod_cnt"] = np.zeros((4, K, NV))

    for a0 in range(0, Mw, chunk):
        b0 = min(a0 + chunk, Mw)
        p = P[a0:b0].numpy().astype(np.float64)
        t = T[a0:b0, :, :, :NV].numpy().astype(np.float64)
        m = (M[a0:b0, :, :, :NV].numpy() > 0.5) & KEEP[None, None]
        th = TH[a0:b0].numpy()

        ti = hours_to_row(th, H0)
        ok_t = (ti >= 0) & (ti < NT)
        tg = np.clip(ti, 0, NT - 1)
        truth = OBS[tg].astype(np.float64)             # (c, K, N, 5) physical
        tmask = OMSK[tg] & ok_t[:, :, None, None]
        pers = OBS[tg[:, 0:1]].astype(np.float64)      # obs(t0), broadcasts
        pmask = OMSK[tg[:, 0:1]] & ok_t[:, 0:1, None, None]
        ci = ti - CLIM_LAG_STEPS
        ok_c = (ci >= 0) & (ci < NT)
        cg = np.clip(ci, 0, NT - 1)
        clim = OBS[cg].astype(np.float64)
        cmask = OMSK[cg] & ok_c[:, :, None, None]

        e_norm = p - t                                 # signed, normalised
        e_phys = e_norm * STD[None, None]
        errs = {
            "mod": (e_phys, e_norm, m & tmask),
            "per": (pers - truth, (pers - truth) / STD[None, None],
                    m & tmask & pmask),
            "clim": (clim - truth, (clim - truth) / STD[None, None],
                     m & tmask & cmask),
        }
        sel_m = None
        if has_mask:
            mi = MI[a0:b0].numpy()
            sel_m = np.zeros((b0 - a0, N), bool)
            np.put_along_axis(sel_m, mi, True, axis=1)

        day = np.floor(th[:, 0] / 24.0).astype(np.int64)
        if day0 is None:
            day0 = int(day.min())
        di = day - day0

        sea = _season_of(th)                           # (c, K)
        for est, (ep_, en_, mk_) in errs.items():
            for sub in subs:
                if sub == "all":
                    mm = mk_
                elif sub == "msk":
                    mm = mk_ & sel_m[:, None, :, None]
                else:
                    mm = mk_ & ~sel_m[:, None, :, None]
                acc[f"{est}_{sub}_sum_phys"] += (np.abs(ep_) * mm).sum(0)
                acc[f"{est}_{sub}_sum_norm"] += (np.abs(en_) * mm).sum(0)
                acc[f"{est}_{sub}_sumsq_phys"] += (ep_ ** 2 * mm).sum(0)
                acc[f"{est}_{sub}_signed_phys"] += (ep_ * mm).sum(0)
                acc[f"{est}_{sub}_cnt"] += mm.sum(0)
            np.add.at(acc[f"day_{est}_sum_phys"], di,
                      (np.abs(ep_) * mk_).sum(2))
            np.add.at(acc[f"day_{est}_cnt"], di, mk_.sum(2))

        mmod = errs["mod"][2]
        for sub in subs:
            if sub == "all":
                mm = mmod
            elif sub == "msk":
                mm = mmod & sel_m[:, None, :, None]
            else:
                mm = mmod & ~sel_m[:, None, :, None]
            acc[f"c_sp_{sub}"] += (p * mm).sum(0)
            acc[f"c_st_{sub}"] += (t * mm).sum(0)
            acc[f"c_spp_{sub}"] += (p ** 2 * mm).sum(0)
            acc[f"c_stt_{sub}"] += (t ** 2 * mm).sum(0)
            acc[f"c_spt_{sub}"] += (p * t * mm).sum(0)
        for si in range(4):
            w = (sea == si)[:, :, None, None]
            acc["sea_mod_sum_phys"][si] += (np.abs(e_phys) * (mmod & w)).sum(
                axis=(0, 2))
            acc["sea_mod_cnt"][si] += (mmod & w).sum(axis=(0, 2))
        del p, t, m, truth, tmask, pers, pmask, clim, cmask, e_norm, e_phys

    acc["day0"] = np.array(day0)
    np.savez_compressed(path, **acc)
    return path


def load_agg(run: str, mr: str = "mr0.00") -> dict:
    p = aggregate_run(run, mr)
    return dict(np.load(p, allow_pickle=False))


# ---------------------------------------------------------------------------
# Metrics from accumulators
# ---------------------------------------------------------------------------
_AX = {"K": 0, "N": 1, "V": 2}


def metric(acc: dict, which: str = "mae", est: str = "mod",
           sub: str = "all", unit: str = "phys",
           pool: tuple = ()) -> np.ndarray:
    """
    which: mae | rmse | bias | corr | nmae (normalised mae)
    pool : axes to POOL (sum numerators and counts) before dividing,
           e.g. pool=("N",) -> (K, V) result. Never averages averages.
    """
    axes = tuple(sorted(_AX[a] for a in pool))
    cnt = acc[f"{est}_{sub}_cnt"].sum(axis=axes) if axes else \
        acc[f"{est}_{sub}_cnt"]
    n = np.maximum(cnt, 1)
    if which == "mae":
        key = f"{est}_{sub}_sum_phys" if unit == "phys" else \
            f"{est}_{sub}_sum_norm"
        s = acc[key]
        return (s.sum(axis=axes) if axes else s) / n
    if which == "nmae":
        s = acc[f"{est}_{sub}_sum_norm"]
        return (s.sum(axis=axes) if axes else s) / n
    if which == "rmse":
        s = acc[f"{est}_{sub}_sumsq_phys"]
        return np.sqrt((s.sum(axis=axes) if axes else s) / n)
    if which == "bias":
        s = acc[f"{est}_{sub}_signed_phys"]
        return (s.sum(axis=axes) if axes else s) / n
    if which == "corr":
        assert est == "mod", "correlation only accumulated for the model"
        def g(k):
            a = acc[f"{k}_{sub}"]
            return a.sum(axis=axes) if axes else a
        sp, st, spp, stt, spt = (g(k) for k in
                                 ("c_sp", "c_st", "c_spp", "c_stt", "c_spt"))
        cov = spt - sp * st / n
        vp = spp - sp ** 2 / n
        vt = stt - st ** 2 / n
        den = np.sqrt(np.clip(vp * vt, 1e-30, None))
        r = cov / den
        return np.where(cnt > 10, r, np.nan)
    raise ValueError(which)


def skill(acc: dict, ref: str = "per", pool: tuple = ("N",)) -> np.ndarray:
    """1 - MAE_model / MAE_ref, both pooled the same way, physical units."""
    m = metric(acc, "mae", "mod", pool=pool)
    r = metric(acc, "mae", ref, pool=pool)
    return 1.0 - m / np.where(r > 0, r, np.nan)


# ---------------------------------------------------------------------------
# Shared per-(model, mask ratio, estimator, lead, station, variable) table
# ---------------------------------------------------------------------------
# ONE canonical file, written by whichever notebook runs first and read by the
# rest: notebooks/Test_Results_Exploration.ipynb and this analysis suite both
# use it. Physical units throughout (degC / hPa / % / m s-1).
#
# Schema (column order is part of the contract — both writers must match):
#   model  mask_ratio  estimator  delta_steps  lead  station  variable
#   MAE  RMSE  n  source
#
# estimator: "model" | "persistence" | "climatology"
# Rows with n == 0 are DROPPED, never written as 0.0 — a station-variable pair
# with no observations is missing data, and a 0.0 there reads as a perfect
# forecast (this exact confusion put the five wind-less stations at the top of
# an "easiest stations" ranking).
PER_STATION_LEAD = os.path.join(TAB, "per_station_lead_metrics.csv")
PSL_COLUMNS = ["model", "mask_ratio", "estimator", "delta_steps", "lead",
               "station", "variable", "MAE", "RMSE", "bias", "n", "source"]
_EST_LABEL = {"mod": "model", "per": "persistence", "clim": "climatology"}


def _newest_dump_mtime() -> float:
    return max(os.path.getmtime(p)
               for d in discovered_runs().values() for p in d.values())


def per_station_lead(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Load the shared table, rebuilding only when it is missing, stale (older
    than the newest dump) or force=True.

    Rebuild streams every dump once via aggregate_run(), which is itself
    cached — so a rebuild after the caches exist is fast.
    """
    if (not force and os.path.isfile(PER_STATION_LEAD)
            and os.path.getmtime(PER_STATION_LEAD) > _newest_dump_mtime()):
        df = pd.read_csv(PER_STATION_LEAD)
        missing = [c for c in PSL_COLUMNS if c not in df.columns]
        if not missing:
            if verbose:
                print(f"per_station_lead: loaded {len(df):,} rows from "
                      f"{os.path.relpath(PER_STATION_LEAD, PROJ)}")
            return df
        if verbose:
            print(f"per_station_lead: file lacks {missing} — rebuilding")

    stn = station_table()
    ns = norm_stats()
    var_names = ns["var_names"]
    frames = []
    for run, mrs in discovered_runs().items():
        for mr in sorted(mrs):
            acc = load_agg(run, mr)
            grid = acc["grid"]
            leads = lead_labels(grid)
            for est, tag in _EST_LABEL.items():
                key = f"{est}_all"
                if f"{key}_cnt" not in acc:
                    continue
                n = acc[f"{key}_cnt"]                       # (K, N, V)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mae = np.where(n > 0, acc[f"{key}_sum_phys"]
                                   / np.maximum(n, 1), np.nan)
                    rmse = np.where(n > 0,
                                    np.sqrt(acc[f"{key}_sumsq_phys"]
                                            / np.maximum(n, 1)), np.nan)
                    bias = np.where(n > 0, acc[f"{key}_signed_phys"]
                                    / np.maximum(n, 1), np.nan)
                K, N, V = n.shape
                frames.append(pd.DataFrame({
                    "model": run, "mask_ratio": mr, "estimator": tag,
                    "delta_steps": np.repeat(grid, N * V),
                    "lead": np.repeat(leads, N * V),
                    "station": np.tile(np.repeat(list(stn["abbr"]), V), K),
                    "variable": np.tile(var_names, K * N),
                    "MAE": mae.ravel(), "RMSE": rmse.ravel(), "bias": bias.ravel(),
                    "n": n.ravel().astype(np.int64),
                    "source": "analysis_suite",
                }))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["n"] > 0][PSL_COLUMNS]
    df.to_csv(PER_STATION_LEAD, index=False)
    if verbose:
        print(f"per_station_lead: wrote {len(df):,} rows -> "
              f"{os.path.relpath(PER_STATION_LEAD, PROJ)}")
    return df


def psl_curve(df: pd.DataFrame, model: str, variable: str,
              mask_ratio: str = "mr0.00", estimator: str = "model",
              stations=None, metric: str = "MAE"):
    """
    Pooled metric vs lead time from the shared table.

    Pooling is COUNT-WEIGHTED, never a mean of per-station means: stations
    contribute in proportion to how many valid observations they have, which
    is what every other number in this project does. Returns (leads, values).
    """
    q = df[(df.model == model) & (df.mask_ratio == mask_ratio)
           & (df.estimator == estimator) & (df.variable == variable)]
    if stations is not None:
        q = q[q.station.isin(list(stations))]
    if metric.upper() == "RMSE":                  # pool squares, then sqrt
        num = (q.RMSE ** 2 * q.n).groupby(q.delta_steps).sum()
        den = q.n.groupby(q.delta_steps).sum()
        out = np.sqrt(num / den.clip(lower=1))
    else:
        num = (q.MAE * q.n).groupby(q.delta_steps).sum()
        den = q.n.groupby(q.delta_steps).sum()
        out = num / den.clip(lower=1)
    out = out.sort_index()
    return out.index.to_numpy(), out.to_numpy()


# ---------------------------------------------------------------------------
# Paired block bootstrap over forecast-origin days
# ---------------------------------------------------------------------------
def paired_day_bootstrap(accA: dict, accB: dict, estA: str = "mod",
                         estB: str = "mod", n_boot: int = 2000,
                         seed: int = 0) -> dict:
    """
    CI for (MAE_A - MAE_B) and for skill 1 - A/B, resampling DAYS with
    replacement — consecutive 10-min windows are heavily dependent, days
    much less so. Returns dict of (K, V) arrays: diff, diff_lo, diff_hi,
    skill, skill_lo, skill_hi. Requires both aggregations to cover the same
    day range (they do: same dumps, same windows).
    """
    sA, cA = accA[f"day_{estA}_sum_phys"], accA[f"day_{estA}_cnt"]
    sB, cB = accB[f"day_{estB}_sum_phys"], accB[f"day_{estB}_cnt"]
    used = (cA.sum(axis=(1, 2)) + cB.sum(axis=(1, 2))) > 0
    sA, cA, sB, cB = sA[used], cA[used], sB[used], cB[used]
    D = sA.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=(n_boot, D))
    def mae(s, c, take):
        return s[take].sum(0) / np.maximum(c[take].sum(0), 1)
    mA, mB = mae(sA, cA, slice(None)), mae(sB, cB, slice(None))
    diffs = np.empty((n_boot,) + mA.shape)
    sk = np.empty_like(diffs)
    for i in range(n_boot):
        take = idx[i]
        a, b = mae(sA, cA, take), mae(sB, cB, take)
        diffs[i] = a - b
        sk[i] = 1.0 - a / np.where(b > 0, b, np.nan)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5], axis=0)
    slo, shi = np.nanpercentile(sk, [2.5, 97.5], axis=0)
    return {"diff": mA - mB, "diff_lo": lo, "diff_hi": hi,
            "skill": 1 - mA / np.where(mB > 0, mB, np.nan),
            "skill_lo": slo, "skill_hi": shi, "n_days": D}


# ---------------------------------------------------------------------------
# Per-window error cache (for regimes / residuals / wind, notebook 08)
# ---------------------------------------------------------------------------
def window_errors(run: str, mr: str = "mr0.00", leads: tuple = (1, 4),
                  uv_lead: int = 4, force: bool = False) -> dict:
    """
    Per-window arrays at selected lead indices (default 1 = +30 min,
    4 = +2 h), cached compressed:
        err_norm  (L, M, N, 5) f16   |pred - target| normalised
        signed_mean (L, M, 5)  f32   station-mean signed error (physical)
        valid     (L, M, N, 5) bool
        uv_pred / uv_targ (M, N, 2) f16  RAW-unit u,v at lead `uv_lead`
        sigma     (L, M, N, 5) f16   only if the dump has log_var
        t0_hours  (M,) f64
    """
    import torch
    tag = f"win_{run}_{mr}_L{'-'.join(map(str, leads))}_uv{uv_lead}.npz"
    path = os.path.join(CACHE, tag)
    if os.path.isfile(path) and not force:
        z = np.load(path)
        return dict(z)
    ns = norm_stats()
    STD = ns["std"]
    stn = station_table()
    KEEP = keep_mask(stn, ns["var_names"])
    d = load_dump(run, mr)
    P, T, M = d["preds"], d["targets"], d["masks"]
    Mw, K, N, _ = P.shape
    NV = len(ns["var_names"])
    L = len(leads)
    err = np.zeros((L, Mw, N, NV), np.float16)
    val = np.zeros((L, Mw, N, NV), bool)
    sgn = np.zeros((L, Mw, NV), np.float32)
    has_lv = "log_var" in d
    sig = np.zeros((L, Mw, N, NV), np.float16) if has_lv else None
    ui = [ns["var_names"].index("wind_u"), ns["var_names"].index("wind_v")]
    uvp = np.zeros((Mw, N, 2), np.float16)
    uvt = np.zeros((Mw, N, 2), np.float16)
    CH = 2000
    for a in range(0, Mw, CH):
        b = min(a + CH, Mw)
        p = P[a:b].numpy()
        t = T[a:b, :, :, :NV].numpy()
        m = (M[a:b, :, :, :NV].numpy() > 0.5) & KEEP[None, None]
        for li, k in enumerate(leads):
            e = p[:, k] - t[:, k]
            err[li, a:b] = np.abs(e).astype(np.float16)
            val[li, a:b] = m[:, k]
            ph = e * STD[None] * m[:, k]
            sgn[li, a:b] = ph.sum(1) / np.maximum(m[:, k].sum(1), 1)
            if has_lv:
                lv = d["log_var"][a:b, k].numpy()
                sig[li, a:b] = np.exp(0.5 * lv).astype(np.float16)
        uvp[a:b] = (p[:, uv_lead][:, :, ui] * STD[None, :, ui]).astype(
            np.float16)
        uvt[a:b] = (t[:, uv_lead][:, :, ui] * STD[None, :, ui]).astype(
            np.float16)
    out = {"err_norm": err, "valid": val, "signed_mean": sgn,
           "uv_pred": uvp, "uv_targ": uvt,
           "t0_hours": d["target_hours"][:, 0].numpy().astype(np.float64),
           "leads": np.array(leads), "uv_lead": np.array(uv_lead)}
    if has_lv:
        out["sigma"] = sig
    np.savez_compressed(path, **out)
    return out


# ---------------------------------------------------------------------------
# Wind helpers
# ---------------------------------------------------------------------------
def wind_speed(u, v):
    return np.sqrt(np.asarray(u, np.float64) ** 2
                   + np.asarray(v, np.float64) ** 2)


def wind_dir_deg(u, v):
    """Direction in degrees from u, v. The CONVENTION (math vs meteorological)
    cancels in any pred-vs-target difference, which is all we use it for."""
    return np.degrees(np.arctan2(np.asarray(v, np.float64),
                                 np.asarray(u, np.float64))) % 360.0


def circular_diff_deg(a, b):
    """Smallest absolute angular difference, degrees, in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str) -> str:
    p = os.path.join(TAB, f"{name}.csv")
    df.to_csv(p)
    return p


def save_fig(fig, name: str) -> str:
    p = os.path.join(FIG, f"{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    return p


def lead_labels(grid: np.ndarray) -> list:
    out = []
    for g in grid:
        mins = int(g) * 10
        if mins == 0:
            out.append("t=0")
        elif mins < 60:
            out.append(f"+{mins}min")
        else:
            h, r = divmod(mins, 60)
            out.append(f"+{h}h" if r == 0 else f"+{h}h{r:02d}")
    return out
