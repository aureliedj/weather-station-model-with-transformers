"""
tests/test_query_anchor.py — v23: the decoder query starts from the station's
own encoder token instead of a shared mask_token.

The problem it replaces
-----------------------
The Δ-query carried no observations, so reproducing a station's own recent
state was a RETRIEVAL problem: cross-attention had to locate that station among
1,872 encoder tokens, learned from scratch, starting from a near-uniform
softmax over 1,872 keys.

Measured on v15's test dump against a perfect copy of the station's own last
observation, the failure is ordered exactly by how copyable a variable is:

    pressure     11.2x the persistence bound
    temperature   1.36x
    humidity      1.23x
    wind          0.94x   <- BETTER than a copy

It failed hardest where copying was the whole answer. That is a lookup failure,
not a modelling failure.

v20 short-circuited it with a persistence residual (y(t0) added outside the
attention path). This replaces that with the structural fix: the station index
is KNOWN when the query is built, so the token is fetched by gather. Attention
was being used to do a soft, learned lookup for something `gather` does exactly.

Two properties the residual did not have:
  * it hands over the full learned representation, not one scalar per variable
  * it applies equally at every lead time, where a single y(t0) base broadcast
    over 13 horizons fades as delta grows

NOT the removed input_context pathway
-------------------------------------
That one built a separate key/value set of last-step embeddings, zeros for
masked stations, and cross-attended to it — so at mask 0.5 half the keys were
zero vectors softmax still had to spend mass on. Here there is no extra
attention and no extra key set, and a masked station receives a LEARNED vector.

The tests below pin the leak guard first, then the mechanism.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.mae import StationMAE                                     # noqa: E402
from model.embeddings import NUM_VARIABLES                           # noqa: E402

B, W, N, V = 2, 12, 20, NUM_VARIABLES
D = 64


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("mask_ratio", 0.5)
    return StationMAE(d_model=D, enc_heads=4, enc_layers=1, dec_heads=4,
                      dec_layers=1, dropout=0.0, window_size=W,
                      cross_attention_decoder=True, **kw).eval()


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, W, N, V, generator=g)
    xm = (torch.rand(B, W, N, V, generator=g) > 0.1).float()
    return {"x": x * xm, "x_mask": xm,
            "spatial": torch.randn(N, 15, generator=g),
            "x_hours": torch.rand(B, W, generator=g) * 1e3 + 4e5}


def _targets(K=3):
    return (torch.randn(B, K, N, V), torch.ones(B, K, N, V),
            torch.rand(B, K) * 1e3 + 4e5,
            torch.arange(0, 3 * K, 3).unsqueeze(0).expand(B, K).contiguous())


# ---------------------------------------------------------------------------
# The leak guard — first, because it is the property that matters most
# ---------------------------------------------------------------------------

def test_masked_stations_receive_the_mask_token_not_an_encoder_token():
    """
    A masked station has no token in the encoder sequence. Its anchor row must
    be exactly mask_token — if anything station-specific leaked in, the whole
    masking objective would be void.
    """
    m = _model(query_anchor=True)
    b = _batch()
    enc, midx, vidx = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    anchor = m._query_anchor(enc, vidx, B, N)
    mt = m.decoder.mask_token.view(-1)
    for bi in range(B):
        for n in midx[bi].tolist():
            assert torch.equal(anchor[bi, n], mt), \
                f"masked station {n} did not get mask_token"


def test_changing_a_masked_stations_values_does_not_change_its_anchor():
    """
    Stronger than the row check: perturb a hidden station's observations and its
    own anchor must be bit-identical. Catches any indirect path.
    """
    m = _model(query_anchor=True)
    b = _batch()
    torch.manual_seed(7)
    enc, midx, vidx = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    a0 = m._query_anchor(enc, vidx, B, N)

    hidden = midx[0, 0].item()
    b2 = {k: v.clone() for k, v in b.items()}
    b2["x"][:, :, hidden, :] += 10.0
    torch.manual_seed(7)                      # same mask draw
    enc2, midx2, vidx2 = m.encoder(b2["x"], b2["x_mask"], b2["spatial"], b2["x_hours"])
    assert torch.equal(midx, midx2), "mask draw differed; test is invalid"
    a1 = m._query_anchor(enc2, vidx2, B, N)
    assert torch.equal(a0[0, hidden], a1[0, hidden])


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------

def test_visible_stations_get_their_own_final_encoder_token():
    m = _model(query_anchor=True)
    b = _batch()
    enc, midx, vidx = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    anchor = m._query_anchor(enc, vidx, B, N)

    n_vis = vidx.shape[1]
    T = enc.shape[1] // n_vis
    own = enc.view(B, T, n_vis, D)[:, -1]              # (B, N_vis, D)
    for bi in range(B):
        for j, n in enumerate(vidx[bi].tolist()):
            assert torch.equal(anchor[bi, n], own[bi, j]), \
                f"station {n} anchored to the wrong row"


def test_anchor_rows_are_distinct_per_station():
    """If every row were the same the anchor would carry no station identity."""
    m = _model(query_anchor=True)
    b = _batch()
    enc, _, vidx = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    a = m._query_anchor(enc, vidx, B, N)
    v0, v1 = vidx[0, 0].item(), vidx[0, 1].item()
    assert not torch.allclose(a[0, v0], a[0, v1])


def test_anchor_changes_the_prediction():
    b = _batch()
    y, ym, yh, dt = _targets()
    outs = []
    for flag in (False, True):
        m = _model(query_anchor=flag)
        torch.manual_seed(3)
        with torch.no_grad():
            outs.append(m.forward_multi_delta(
                b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)[1])
    assert (outs[0] - outs[1]).abs().max() > 1e-5


def test_default_is_off():
    m = _model()
    assert m.query_anchor is False
    b = _batch(); y, ym, yh, dt = _targets()
    with torch.no_grad():
        loss, preds, *_ = m.forward_multi_delta(
            b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert torch.isfinite(loss) and torch.isfinite(preds).all()


@pytest.mark.parametrize("mask_ratio", [0.0, 0.5])
def test_runs_end_to_end(mask_ratio):
    m = _model(query_anchor=True, mask_ratio=mask_ratio)
    b = _batch(); y, ym, yh, dt = _targets()
    with torch.no_grad():
        loss, preds, *_ = m.forward_multi_delta(
            b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert torch.isfinite(loss) and torch.isfinite(preds).all()


def test_anchor_is_normalised_so_the_regimes_differ_in_content_not_scale():
    """
    Encoder output is LayerNorm'd; mask_token is trunc_normal(std=0.02). Without
    anchor_norm the visible queries would be ~50x the masked ones and the two
    regimes would be distinguished by magnitude rather than by what they carry.
    """
    m = _model(query_anchor=True)
    b = _batch()
    enc, midx, vidx = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    a = m.decoder.anchor_norm(m._query_anchor(enc, vidx, B, N))
    vis = a[0, vidx[0]].std()
    msk = a[0, midx[0]].std()
    assert 0.2 < (msk / vis).item() < 5.0, \
        f"visible {vis:.3f} vs masked {msk:.3f} — anchor_norm is not equalising scale"


# ---------------------------------------------------------------------------
# Composition with the modes it replaces / is compared against
# ---------------------------------------------------------------------------

def test_anchor_and_residual_are_independent():
    """
    The residual is what the anchor replaces, but nothing forbids both. Verify
    they compose so a controlled A/B is possible.
    """
    m = _model(query_anchor=True, residual_head=True)
    b = _batch(); y, ym, yh, dt = _targets()
    with torch.no_grad():
        loss, preds, *_ = m.forward_multi_delta(
            b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert torch.isfinite(loss) and torch.isfinite(preds).all()


def test_direct_head_ignores_the_anchor():
    """v22 has no decoder, so the anchor is inert there — it must not crash."""
    m = _model(query_anchor=True, direct_head=True, mask_ratio=0.0, num_horizons=3)
    b = _batch(); y, ym, yh, dt = _targets()
    with torch.no_grad():
        loss, preds, *_ = m.forward_multi_delta(
            b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert torch.isfinite(loss) and torch.isfinite(preds).all()


def test_cfg_roundtrip_restores_the_flag():
    cfg = {"d_model": D, "enc_heads": 4, "enc_layers": 1, "dec_heads": 4,
           "dec_layers": 1, "mask_ratio": 0.5, "window": W,
           "cross_attn_decoder": True, "query_anchor": True}
    assert StationMAE.from_cfg(cfg).query_anchor is True
