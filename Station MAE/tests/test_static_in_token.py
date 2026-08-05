"""
tests/test_static_in_token.py — v21: static features as slots inside the
variable block, Aurora-style.

The idea
--------
Aurora "incorporate[s] static variables (orography, land-sea mask, and
soil-type mask) by treating them as extra surface-level variables". Terrain then
passes through the SAME per-slot map and the SAME /sqrt(S) divisor as the
weather, so its scale cannot drift away from the observations' — which is what
happened when position and topography had their own MLP branches.

The trap
--------
The grouping matters more than the idea. Aurora has ~3 statics against many
variables; this project has 15 statics against 6 weather variables. Measured
weather share of the token at initialisation:

    one slot per static feature   21 slots    9.6%   <- WORSE than the default
    position + topography          8 slots   25.1%   <- what this implements
    all 15 in one slot             7 slots   28.7%
    v18 separate branches                    18.3%

So the tests below pin the grouping, not just the mechanism.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.embeddings import (                                        # noqa: E402
    VariableProjection, NUM_VARIABLES, SPATIAL_INPUT_DIM, STATIC_SLOTS_DEFAULT,
)
from model.encoder import StationMAEEncoder                           # noqa: E402
from model.token_balance import synthetic_batch                       # noqa: E402

D, V, S = 384, NUM_VARIABLES, SPATIAL_INPUT_DIM


def _vp(**kw):
    torch.manual_seed(0)
    return VariableProjection(num_vars=V, d_model=D, **kw)


def _enc(**kw):
    torch.manual_seed(0)
    return StationMAEEncoder(d_model=D, num_heads=8, num_layers=1,
                             dropout=0.0, **kw).eval()


# ---------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------

def test_default_is_unchanged():
    """v18 behaviour must survive as the default."""
    vp = _vp()
    assert vp.static_slots is None
    assert vp.total_slots == vp.num_slots == V


def test_static_slots_add_to_the_slot_count():
    vp = _vp(static_slots=STATIC_SLOTS_DEFAULT)
    assert vp.total_slots == V + 2, "expected 6 weather + 2 static slots"
    assert len(vp.static_maps) == 2


def test_grouping_is_position_plus_topography():
    """Mirrors the p1/p2 split it replaces: cols 0:2 and 2:15."""
    assert STATIC_SLOTS_DEFAULT == ((0, 2), (2, 15))
    assert sum(hi - lo for lo, hi in STATIC_SLOTS_DEFAULT) == S


def test_static_is_required_when_configured():
    vp = _vp(static_slots=STATIC_SLOTS_DEFAULT)
    x = torch.randn(4, 20, V)
    with pytest.raises(ValueError, match="received static=None"):
        vp(x, torch.ones_like(x))


def test_static_changes_the_output():
    vp = _vp(static_slots=STATIC_SLOTS_DEFAULT)
    x = torch.randn(4, 20, V); m = torch.ones_like(x)
    g = torch.Generator().manual_seed(3)
    a = vp(x, m, static=torch.randn(20, S, generator=g))
    b = vp(x, m, static=torch.randn(20, S, generator=g))
    assert (a - b).abs().max() > 1e-4, "static slots are not reaching the output"


def test_static_is_constant_in_time():
    """
    Statics depend only on the station, so the SAME (N, S) must give the same
    contribution at every timestep — that is what makes it free to compute once.
    """
    vp = _vp(static_slots=STATIC_SLOTS_DEFAULT)
    stat = torch.randn(20, S)
    z = torch.zeros(3, 20, V)
    out = vp(z, torch.zeros_like(z), static=stat)      # all sensors absent
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])


def test_divisor_uses_the_total_slot_count():
    """Adding static slots must widen the divisor, or the branch inflates."""
    g = torch.Generator().manual_seed(9)
    x = torch.randn(64, 40, V, generator=g); m = torch.ones_like(x)
    stat = torch.randn(40, S, generator=g)
    plain  = _vp()(x, m).std().item()
    withst = _vp(static_slots=STATIC_SLOTS_DEFAULT)(x, m, static=stat).std().item()
    assert withst / plain < 1.6, (
        f"branch grew {withst/plain:.2f}x — the divisor is not sqrt(total_slots)")


# ---------------------------------------------------------------------------
# The encoder must not double-count
# ---------------------------------------------------------------------------

def test_encoder_drops_the_separate_branches():
    """
    pos_emb and station_emb still EXIST (the decoder uses its own copies), but
    must not be added to the token — that would double-count the statics and
    restore the imbalance this mode removes.
    """
    enc = _enc(static_in_token=True)
    b = synthetic_batch(seed=0)
    with torch.no_grad():
        tok = enc._build_tokens(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    assert torch.isfinite(tok).all() and tok.shape[-1] == D

    # moving a station's coordinates must still change its token, via the slots
    b2 = {k: v.clone() for k, v in b.items()}
    b2["spatial"][0, :2] += 5.0
    with torch.no_grad():
        tok2 = enc._build_tokens(b2["x"], b2["x_mask"], b2["spatial"], b2["x_hours"])
    assert (tok[:, :, 0] - tok2[:, :, 0]).abs().max() > 1e-4


@pytest.mark.parametrize("mode", ["linear", "mlp", "fourier"])
def test_all_value_modes_accept_static_slots(mode):
    vp = _vp(value_embedding=mode, static_slots=STATIC_SLOTS_DEFAULT)
    x = torch.randn(4, 12, V); m = (torch.rand(4, 12, V) > 0.1).float()
    out = vp(x * m, m, static=torch.randn(12, S))
    assert out.shape == (4, 12, D) and torch.isfinite(out).all()


def test_weather_share_beats_the_separate_branch_default():
    """
    The point of the mode: the weather's share of the token must go UP.
    Compares variance across tokens of the weather-only part.
    """
    torch.manual_seed(0)
    b = synthetic_batch(seed=0)
    x, m, sp = b["x"], b["x_mask"], b["spatial"]
    B, W, N, _ = x.shape
    xf, mf = x.reshape(B * W, N, V), m.reshape(B * W, N, V)

    vp = _vp(value_embedding="mlp", static_slots=STATIC_SLOTS_DEFAULT)
    with torch.no_grad():
        full  = vp(xf, mf, static=sp)
        stat  = vp(torch.zeros_like(xf), torch.zeros_like(mf), static=sp)
    weather_var = (full - stat).reshape(-1, D).var(0).sum().item()
    static_var  = stat.reshape(-1, D).var(0).sum().item()
    assert weather_var > static_var, (
        f"weather {weather_var:.1f} vs static {static_var:.1f} — the 2-slot "
        f"grouping should keep the weather dominant inside the block")
