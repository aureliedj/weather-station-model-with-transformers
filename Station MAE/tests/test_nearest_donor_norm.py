"""
Verify nearest-donor normalisation for sparse (station, variable) pairs.

The failure this replaces: a station with too few training observations used the
cross-station GLOBAL mean/std. For altitude-driven variables that is badly wrong
across a network spanning ~200-3600 m -- GES pressure has ZERO training
observations, so it was normalised against a mean pressure belonging to no real
station.

Run: python tests/test_nearest_donor_norm.py
"""
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "src"))

from data.dataset import normalise_observations, VARIABLE_NAMES  # noqa: E402

V = len(VARIABLE_NAMES)
PRESSURE = VARIABLE_NAMES.index("pressure") if "pressure" in VARIABLE_NAMES else 2


def build(n_stations=6, T=500, seed=0):
    """
    Six stations laid out so the right answer is unambiguous:

      idx  easting(km)  height(m)   role
       0        0         400       lowland reference
       1        3        2600       HIGH station, very close horizontally
       2       60         420       lowland, far away, similar altitude
       3      120        1500       mid-altitude, far
       4        1         410       lowland, adjacent to 0
       5        2         405       THE SPARSE ONE (no pressure data)

    Station 5 sits at 405 m. Its nearest neighbour by pure horizontal distance
    is station 1 (1 km away) -- but station 1 is 2195 m higher, so its mean
    pressure is ~250 hPa lower. A correct donor search must reject it and pick
    station 4 or 0 instead.
    """
    rng = np.random.default_rng(seed)
    east = np.array([0., 3000., 60000., 120000., 1000., 2000.])
    north = np.zeros(n_stations)
    hgt = np.array([400., 2600., 420., 1500., 410., 405.])

    obs = np.zeros((T, n_stations, V), dtype=np.float32)
    mask = np.ones((T, n_stations, V), dtype=np.float32)
    for i in range(n_stations):
        # barometric-ish: ~12 hPa per 100 m
        p_mean = 1013.0 - 0.12 * hgt[i]
        for v in range(V):
            centre = p_mean if v == PRESSURE else 10.0 - 0.0065 * hgt[i]
            obs[:, i, v] = rng.normal(centre, 2.0, size=T)

    # station 5 has NO pressure observations at all (the GES situation)
    obs[:, 5, PRESSURE] = 0.0
    mask[:, 5, PRESSURE] = 0.0

    # z-scored 15-d spatial vector, matching build_spatial_features() layout:
    # [east, north, sin/cos aspect x2, station_height, dem, TPI, slope x2, deriv x4]
    slope = np.array([2., 30., 3., 18., 2.5, 2.])
    raw = np.zeros((n_stations, 15))
    raw[:, 0], raw[:, 1] = east, north
    raw[:, 6], raw[:, 7] = hgt, hgt
    raw[:, 9], raw[:, 10] = slope, slope
    sd = raw.std(0); sd[sd < 1e-9] = 1.0
    coords = torch.tensor((raw - raw.mean(0)) / sd, dtype=torch.float64)
    ids = ["S0", "S1_HIGH", "S2_FAR", "S3_MID", "S4_NEAR", "S5_SPARSE"]
    return torch.tensor(obs), torch.tensor(mask), coords, ids, hgt


obs, mask, coords, ids, hgt = build()

print("=" * 72)
print("WITH station features -> most-similar-station substitution")
print("=" * 72)
_, stats = normalise_observations(obs, mask, per_station=True,
                                  coords=coords, station_ids=ids,
                                  verbose=True)
donor = stats["donor"]
d = int(donor[5, PRESSURE])
print(f"\nstation 5 pressure donor = {d} ({ids[d] if d >= 0 else 'GLOBAL' })")

assert d >= 0, "should have found a real donor, not the global fallback"
assert d != 1, (f"picked the HIGH station {ids[1]} at {hgt[1]:.0f} m -- "
                f"altitude penalty is not working")
assert hgt[d] < 600, f"donor {ids[d]} is at {hgt[d]:.0f} m, not comparable to 405 m"
print(f"  donor altitude {hgt[d]:.0f} m vs station 5 at {hgt[5]:.0f} m  "
      f"(Δ {hgt[d]-hgt[5]:+.0f} m)  OK")

# the borrowed mean must be physically plausible for a 405 m station
borrowed = float(stats["mean"][5, PRESSURE])
expected = 1013.0 - 0.12 * hgt[5]
print(f"  borrowed mean pressure {borrowed:.1f} hPa  vs physically expected "
      f"{expected:.1f} hPa  (error {abs(borrowed-expected):.1f} hPa)")
assert abs(borrowed - expected) < 15, "borrowed mean is not physically plausible"

# every other pair must be untouched
own = (donor == -1)
assert own.sum() == donor.numel() - 1, "only the one sparse pair should be substituted"
print(f"  {int(own.sum())}/{donor.numel()} pairs kept their own stats  OK")

print("\n" + "=" * 72)
print("WITHOUT coords -> old global behaviour (the bug being replaced)")
print("=" * 72)
_, stats_g = normalise_observations(obs, mask, per_station=True, coords=None)
g_mean = float(stats_g["mean"][5, PRESSURE])
print(f"\nstation 5 pressure mean = {g_mean:.1f} hPa (global, altitude-blind)")
print(f"  physically expected     = {expected:.1f} hPa")
print(f"  error                   = {abs(g_mean-expected):.1f} hPa")
assert int(stats_g["donor"][5, PRESSURE]) == -2, "should be flagged as global fallback"
assert abs(g_mean - expected) > abs(borrowed - expected), \
    "global fallback should be WORSE than the nearest donor -- test is not discriminating"
print(f"\n  nearest-donor error {abs(borrowed-expected):5.1f} hPa")
print(f"  global-fallback err {abs(g_mean-expected):5.1f} hPa"
      f"   -> {abs(g_mean-expected)/max(abs(borrowed-expected),1e-6):.1f}x worse")

print("\n" + "=" * 72)
print("No donor anywhere -> must still fall back to global, not crash")
print("=" * 72)
obs2, mask2 = obs.clone(), mask.clone()
mask2[:, :, PRESSURE] = 0.0            # nobody has pressure
_, stats_n = normalise_observations(obs2, mask2, per_station=True,
                                    coords=coords, station_ids=ids, verbose=False)
assert (stats_n["donor"][:, PRESSURE] == -2).all(), "all should be global fallback"
assert torch.isfinite(stats_n["mean"]).all() and (stats_n["std"] > 0).all()
print("  all pressure pairs -> global fallback, stats finite and std > 0  OK")

print("\nALL NEAREST-DONOR CHECKS PASSED")
