"""
Pins the per-station inverse normalisation in engine/evaluate.py.

Observations are z-scored per (station, variable), so the only correct way back
to physical units is that station's OWN mean/std. evaluate.py used to average
std over stations first; because wind std spans 0.51-7.19 m/s across the 155
stations (14x) against 1.4x for temperature, that shortcut inflated wind error
by ~6-7% and wind-direction MAE from 15.8 to 20.3 degrees (winds > 3 m/s).

These tests lock in:
  1. _row_stat indexes per station, and falls back to the cross-station mean
     ONLY for unknown station indices or (V,)-shaped global stats.
  2. The station index reconstructed from the flattened (B, N, V) arrays lines
     up with the rows (station axis cycles fastest).
  3. Denormalising with per-station stats is an exact inverse; the averaged
     shortcut is not.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.evaluate import _row_stat  # noqa: E402


def test_row_stat_indexes_per_station():
    table = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])   # (3 stations, 2 vars)
    idx = torch.tensor([0, 1, 2, 1, 0])
    got = _row_stat(table, 0, idx)
    assert torch.allclose(got, torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]))
    got_v1 = _row_stat(table, 1, idx)
    assert torch.allclose(got_v1, torch.tensor([10.0, 20.0, 30.0, 20.0, 10.0]))


def test_row_stat_falls_back_only_for_unknown_stations():
    table = torch.tensor([[1.0], [2.0], [3.0]])
    idx = torch.tensor([0, 99, 2])          # 99 is out of range
    got = _row_stat(table, 0, idx)
    # known stations keep their own value; the unknown one gets the column mean
    assert got[0] == pytest.approx(1.0)
    assert got[2] == pytest.approx(3.0)
    assert got[1] == pytest.approx(2.0)     # mean of [1, 2, 3]


def test_row_stat_handles_global_stats():
    """A (V,) table means normalisation was global — every row gets the same value."""
    table = torch.tensor([5.0, 50.0])
    idx = torch.tensor([0, 1, 2])
    assert torch.allclose(_row_stat(table, 1, idx), torch.full((3,), 50.0))


def test_station_index_matches_flattened_layout():
    """
    evaluate_full flattens (B, N, V) -> (B*N, V) and rebuilds the station index
    as arange(N).repeat(B). If that ordering assumption is ever wrong, every
    physical number silently uses another station's std.
    """
    B, N, V = 4, 7, 3
    a = torch.arange(B * N * V, dtype=torch.float32).reshape(B, N, V)
    flat = a.reshape(B * N, V)
    idx = torch.arange(N).repeat(B)
    assert idx.shape[0] == flat.shape[0]
    for r in range(B * N):
        assert torch.equal(flat[r], a[r // N, idx[r]])


def test_per_station_denorm_is_exact_and_averaged_is_not():
    """Round-trip: per-station stats recover the physical values exactly."""
    torch.manual_seed(0)
    N, V = 6, 2
    std = torch.tensor([[0.5, 8.0], [7.0, 8.1], [1.0, 7.9],
                        [3.0, 8.2], [0.8, 7.8], [5.0, 8.0]])   # var 0 spans 14x
    mean = torch.randn(N, V)
    phys = torch.randn(40, N, V) * 3.0
    norm = (phys - mean) / std

    idx = torch.arange(N).repeat(40)
    flat_norm = norm.reshape(-1, V)
    flat_phys = phys.reshape(-1, V)

    for v in range(V):
        s = _row_stat(std, v, idx)
        m = _row_stat(mean, v, idx)
        assert torch.allclose(flat_norm[:, v] * s + m, flat_phys[:, v], atol=1e-5)

    # the averaged shortcut is materially wrong for the high-spread variable
    err_avg = (flat_norm[:, 0] * std[:, 0].mean() + mean[:, 0].mean()
               - flat_phys[:, 0]).abs().mean()
    assert err_avg > 1e-2, "averaged std should visibly disagree on a 14x-spread variable"


def test_wind_direction_needs_per_station_stats():
    """
    Direction is an angle, so an averaged std shears the (u, v) plane and
    rotates it. Verify the per-station reconstruction recovers the true angle
    and the averaged one does not.
    """
    from engine.evaluate import _wind_dir_deg

    N = 4
    std = torch.tensor([[0.5, 5.0], [5.0, 0.5], [2.0, 2.0], [1.0, 4.0]])
    mean = torch.zeros(N, 2)
    u = torch.tensor([[1.0], [1.0], [1.0], [1.0]]).squeeze(-1)
    v = torch.tensor([[1.0], [1.0], [1.0], [1.0]]).squeeze(-1)   # true 45-deg wind
    u_norm = (u - mean[:, 0]) / std[:, 0]
    v_norm = (v - mean[:, 1]) / std[:, 1]

    idx = torch.arange(N)
    u_rec = u_norm * _row_stat(std, 0, idx) + _row_stat(mean, 0, idx)
    v_rec = v_norm * _row_stat(std, 1, idx) + _row_stat(mean, 1, idx)
    assert torch.allclose(_wind_dir_deg(u_rec, v_rec),
                          _wind_dir_deg(u, v), atol=1e-4)

    u_bad = u_norm * std[:, 0].mean()
    v_bad = v_norm * std[:, 1].mean()
    ang_true = _wind_dir_deg(u, v)
    ang_bad = _wind_dir_deg(u_bad, v_bad)
    diff = ((ang_bad - ang_true + 180.0) % 360.0 - 180.0).abs()
    assert diff.max() > 5.0, "averaged std should visibly rotate the wind direction"
