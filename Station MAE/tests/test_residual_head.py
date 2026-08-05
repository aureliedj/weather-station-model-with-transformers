"""
tests/test_residual_head.py — the persistence residual head and the station-state
embedding that makes it usable.

Why this exists
---------------
Measured on v15's own validation metrics, against what a perfect copy of the
station's last observation would score at +30 min:

    variable      persist r   copy bound   v15 actual   ratio
    pressure         0.9995      0.0254       0.0528    2.08x
    temperature      0.9956      0.0744       0.0873    1.17x
    humidity         0.9722      0.1868       0.2086    1.12x
    wind_u           0.8472      0.4240       0.3797    0.90x
    wind_v           0.8355      0.4385       0.3987    0.91x

The model beats a copy on wind and loses to it on pressure, temperature and
humidity — worst exactly where copying is easiest. The decoder query carries no
observations, so reproducing a station's own last value is a retrieval problem
over 1,872 patched encoder tokens. The residual head hands it over directly.

It was tried once and degraded, because the base is bimodal (y(t0) for visible
stations, 0 for masked) while the queries carried no signal saying which regime
they were in. The station_state embedding supplies exactly that one bit.

The tests below pin the two properties that matter: the regime IS observable,
and the masked stations' values do NOT leak.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.mae import StationMAE                                     # noqa: E402
from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES     # noqa: E402

B, W, N, V = 2, 12, 20, NUM_VARIABLES
D = 64


def _model(**kw):
    torch.manual_seed(0)
    return StationMAE(d_model=D, enc_heads=4, enc_layers=1, dec_heads=4,
                      dec_layers=1, dropout=0.0, window_size=W,
                      cross_attention_decoder=True, **kw).eval()


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    x  = torch.randn(B, W, N, V, generator=g)
    xm = (torch.rand(B, W, N, V, generator=g) > 0.1).float()
    return {"x": x * xm, "x_mask": xm,
            "spatial": torch.randn(N, 15, generator=g),
            "x_hours": torch.rand(B, W, generator=g) * 1e3 + 4e5}


# ---------------------------------------------------------------------------
# The station-state embedding
# ---------------------------------------------------------------------------

def test_station_state_exists_and_has_two_states():
    m = _model()
    assert m.decoder.station_state.shape == (2, D), \
        "expected one learned vector for visible and one for masked"


def test_the_two_states_are_distinguishable():
    """
    If both states produced the same query, the head would be blind to which
    regime it is in — the exact defect that made --residual_head degrade.
    """
    m = _model()
    a, b = m.decoder.station_state[0], m.decoder.station_state[1]
    assert (a - b).abs().max() > 1e-6, "visible and masked states are identical"


def test_masking_a_station_changes_its_query():
    """The flag must actually reach the query, not just exist as a parameter."""
    m = _model()
    b = _batch()
    enc, _, _ = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    y_hours = torch.rand(B, 3) * 1e3 + 4e5
    deltas  = torch.tensor([[0, 3, 6]] * B)

    none_masked = torch.zeros(B, N, dtype=torch.bool)
    one_masked  = none_masked.clone(); one_masked[:, 5] = True
    with torch.no_grad():
        p0 = m.decoder(enc, b["spatial"], y_hours, deltas, station_masked=none_masked)
        p1 = m.decoder(enc, b["spatial"], y_hours, deltas, station_masked=one_masked)
    assert (p0[:, :, 5] - p1[:, :, 5]).abs().max() > 1e-6, \
        "flagging a station as masked did not change its prediction"


def test_default_is_all_visible():
    """
    Passing nothing must mean 'every station visible' — what evaluation at
    mask_ratio 0 wants, and what keeps test.py working unchanged.
    """
    m = _model()
    b = _batch()
    enc, _, _ = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    y_hours = torch.rand(B, 2) * 1e3 + 4e5
    deltas  = torch.tensor([[0, 3]] * B)
    with torch.no_grad():
        a = m.decoder(enc, b["spatial"], y_hours, deltas, station_masked=None)
        c = m.decoder(enc, b["spatial"], y_hours, deltas,
                      station_masked=torch.zeros(B, N, dtype=torch.bool))
    assert torch.allclose(a, c)


# ---------------------------------------------------------------------------
# The residual base — and the leak it must not create
# ---------------------------------------------------------------------------

def test_base_is_the_last_observation_for_visible_stations():
    m = _model(residual_head=True, mask_ratio=0.0)
    b = _batch()
    base = m._persistence_base(b["x"], b["x_mask"],
                               torch.empty(B, 0, dtype=torch.long))
    assert base is not None
    expected = (b["x"][:, -1, :, :NUM_TARGET_VARIABLES]
                * b["x_mask"][:, -1, :, :NUM_TARGET_VARIABLES])
    assert torch.allclose(base, expected, atol=1e-6)


def test_masked_stations_get_a_zero_base():
    """
    A masked station's last observation is hidden information. Feeding it
    through the base would bypass the encoder mask — the leak class the old
    input_context pathway had.
    """
    m = _model(residual_head=True)
    b = _batch()
    midx = torch.tensor([[3, 7]] * B)
    base = m._persistence_base(b["x"], b["x_mask"], midx)
    assert base[:, 3].abs().max() == 0.0
    assert base[:, 7].abs().max() == 0.0
    others = [n for n in range(N) if n not in (3, 7)]
    assert base[:, others].abs().max() > 0.0, "visible stations lost their base"


def test_head_disabled_returns_no_base():
    m = _model(residual_head=False)
    b = _batch()
    assert m._persistence_base(b["x"], b["x_mask"], None) is None


def test_station_masked_flag_matches_masked_idx():
    m = _model()
    midx = torch.tensor([[2, 9], [4, 9]])
    flag = m._station_masked(midx, 2, N, torch.device("cpu"))
    assert flag.shape == (2, N) and flag.dtype == torch.bool
    assert flag[0, 2] and flag[0, 9] and flag[1, 4] and flag[1, 9]
    assert flag.sum().item() == 4


@pytest.mark.parametrize("residual", [False, True])
def test_forward_runs_end_to_end(residual):
    """Both regimes must produce finite predictions of the right shape."""
    m = _model(residual_head=residual, mask_ratio=0.5)
    b = _batch()
    y_hours = torch.rand(B, 3) * 1e3 + 4e5
    deltas  = torch.tensor([[0, 3, 6]] * B)
    y  = torch.randn(B, 3, N, V)
    ym = torch.ones(B, 3, N, V)
    with torch.no_grad():
        out = m.forward_multi_delta(b["x"], b["x_mask"], b["spatial"],
                                    b["x_hours"], y, ym, y_hours, deltas)
    loss, preds = out[0], out[1]
    assert torch.isfinite(loss) and preds.shape == (B, 3, N, NUM_TARGET_VARIABLES)
    assert torch.isfinite(preds).all()
