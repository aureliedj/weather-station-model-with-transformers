"""
tests/test_direct_head.py — v22: the plain forecasting transformer.

Shape
-----
    (station, timestep) tokens  ->  factorised encoder  ->  one vector per
    station  ->  Linear(d_model -> K * V_target)

No masking, no queries, no cross-attention. The encoder and the whole embedding
stack are unchanged from v21.

Why the decoder goes
--------------------
Measured on v15's validation metrics against a perfect copy of the station's own
last observation at +30 min: pressure 2.08x worse, temperature 1.17x, humidity
1.12x — but wind 0.90x, i.e. BETTER than a copy. The model fails where copying
is easiest, which is a retrieval problem: the decoder query holds no
observations, so reproducing a station's own last value means locating it among
1,872 patched encoder tokens.

With a direct 'last' readout that value is one linear layer away.

The prior attempt
-----------------
model/masked_transformer.py used this exact shape and diverged. It predates the
observation-token rebalance and trained at a content fraction of 0.04%, so the
readout was never cleanly implicated — treat it as untested, not known-bad.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.mae import StationMAE                                     # noqa: E402
from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES     # noqa: E402

B, W, N, V, K = 2, 12, 20, NUM_VARIABLES, 5
D = 64


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("mask_ratio", 0.0)
    kw.setdefault("num_horizons", K)
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


def _targets():
    return (torch.randn(B, K, N, V), torch.ones(B, K, N, V),
            torch.rand(B, K) * 1e3 + 4e5,
            torch.arange(0, 3 * K, 3).unsqueeze(0).expand(B, K).contiguous())


# ---------------------------------------------------------------------------
# Shape and wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("readout", ["last", "mean"])
def test_direct_head_produces_all_stations_and_horizons(readout):
    m = _model(direct_head=True, readout=readout)
    b = _batch()
    y, ym, yh, dt = _targets()
    with torch.no_grad():
        loss, preds, *_ = m.forward_multi_delta(
            b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert preds.shape == (B, K, N, NUM_TARGET_VARIABLES)
    assert torch.isfinite(preds).all() and torch.isfinite(loss)


def test_decoder_is_not_used():
    """The point of the mode: no queries, no cross-attention."""
    m = _model(direct_head=True)
    b = _batch()
    y, ym, yh, dt = _targets()
    m.decoder.register_forward_hook(
        lambda *a: pytest.fail("decoder ran despite --direct_head"))
    with torch.no_grad():
        m.forward_multi_delta(b["x"], b["x_mask"], b["spatial"], b["x_hours"],
                              y, ym, yh, dt)


def test_default_still_uses_the_decoder():
    m = _model(direct_head=False, mask_ratio=0.5)
    assert not m.direct_head and not hasattr(m, "direct_proj")


# ---------------------------------------------------------------------------
# The constraint that makes the readout well-defined
# ---------------------------------------------------------------------------

def test_masking_is_refused():
    """
    The readout uses each station's OWN tokens, so a station removed from the
    sequence has nothing to read. Fail loudly at construction rather than
    silently predicting for the wrong stations.
    """
    with pytest.raises(AssertionError, match="mask_ratio"):
        _model(direct_head=True, mask_ratio=0.5)


def test_readout_choice_changes_the_prediction():
    b = _batch()
    y, ym, yh, dt = _targets()
    outs = []
    for r in ("last", "mean"):
        m = _model(direct_head=True, readout=r)
        with torch.no_grad():
            outs.append(m.forward_multi_delta(
                b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)[1])
    assert (outs[0] - outs[1]).abs().max() > 1e-5


# ---------------------------------------------------------------------------
# The property the mode exists for
# ---------------------------------------------------------------------------

def test_a_station_sees_its_own_recent_observation():
    """
    Perturb ONLY the last timestep of one station and its own prediction must
    move. With 'last' this is a short path through the residual stream — the
    structural advantage the query decoder lacks.
    """
    m = _model(direct_head=True, readout="last")
    b = _batch()
    y, ym, yh, dt = _targets()
    with torch.no_grad():
        p0 = m.forward_multi_delta(b["x"], b["x_mask"], b["spatial"],
                                   b["x_hours"], y, ym, yh, dt)[1]
    b2 = {k: v.clone() for k, v in b.items()}
    b2["x"][:, -1, 3, :] += 5.0                     # station 3, last step only
    with torch.no_grad():
        p1 = m.forward_multi_delta(b2["x"], b2["x_mask"], b2["spatial"],
                                   b2["x_hours"], y, ym, yh, dt)[1]
    moved = (p0[:, :, 3] - p1[:, :, 3]).abs().max()
    assert moved > 1e-4, f"station 3's own prediction barely moved ({moved:.2e})"


def test_perturbing_one_station_does_not_move_the_others():
    """
    The test above is necessary but NOT sufficient, and on its own it missed a
    real bug: `_mask_stations` used to permute the station axis even at
    mask_ratio 0, so row j of the encoder output was some other station. Under
    a permutation station 3's prediction still MOVES when station 3 is
    perturbed — it has simply been routed elsewhere — so the assertion above
    passed while the head was reading the wrong rows.

    Locality is the property that actually pins the mapping: with a per-station
    readout and one encoder whose spatial attention is the only cross-station
    path, perturbing station 3 must leave the OTHER stations far less affected
    than station 3 itself.
    """
    m = _model(direct_head=True, readout="last")
    b = _batch()
    y, ym, yh, dt = _targets()
    with torch.no_grad():
        p0 = m.forward_multi_delta(b["x"], b["x_mask"], b["spatial"],
                                   b["x_hours"], y, ym, yh, dt)[1]
    b2 = {k: v.clone() for k, v in b.items()}
    b2["x"][:, -1, 3, :] += 5.0
    with torch.no_grad():
        p1 = m.forward_multi_delta(b2["x"], b2["x_mask"], b2["spatial"],
                                   b2["x_hours"], y, ym, yh, dt)[1]

    delta = (p0 - p1).abs().amax(dim=(0, 1, 3))          # (N,) per station
    others = torch.tensor([n for n in range(N) if n != 3])
    assert delta[3] > delta[others].max(), (
        f"station 3 moved {delta[3]:.3e} but station "
        f"{others[delta[others].argmax()].item()} moved "
        f"{delta[others].max():.3e} — the readout is not reading station 3's "
        f"own tokens")


def test_gradient_reaches_every_temporal_slot_with_mean():
    """
    'last' gives direct gradient only to the final temporal token; 'mean' gives
    it to all of them. This is the lever if 'last' proves unstable — the
    concern raised (though never cleanly demonstrated) by the earlier attempt.
    """
    m = _model(direct_head=True, readout="mean").train()
    b = _batch()
    y, ym, yh, dt = _targets()
    b["x"].requires_grad_(True)
    loss = m.forward_multi_delta(b["x"], b["x_mask"], b["spatial"],
                                 b["x_hours"], y, ym, yh, dt)[0]
    loss.backward()
    per_step = b["x"].grad.abs().flatten(2).sum(-1)          # (B, W)
    assert (per_step > 0).all(), "some timesteps received no gradient at all"
