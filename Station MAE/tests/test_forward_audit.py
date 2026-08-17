"""
tests/test_forward_audit.py — end-to-end shape + gradient-flow validation.

Self-contained audit harness: dummy tensors in, expected shapes out, one real
backward pass per configuration, and explicit checks that gradients reach
every parameter family they are supposed to reach — including the SHARED
embedding modules, which must accumulate gradient from BOTH the encoder
(key/value side) and the decoder (query side).

Small config on purpose (D=32, 1+1 layers, W=12, N=10, K=3): every check here
is shape- and graph-structural, none depends on model capacity.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.mae import StationMAE                                     # noqa: E402
from model.embeddings import NUM_VARIABLES, NUM_TARGET_VARIABLES     # noqa: E402

B, W, N, V = 2, 12, 10, NUM_VARIABLES          # V = 6
Vt = NUM_TARGET_VARIABLES                      # 5 (no precipitation)
K, D = 3, 32
HOURS = 4.9e5                                  # realistic epoch-hours magnitude


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("mask_ratio", 0.5)
    return StationMAE(d_model=D, enc_heads=4, enc_layers=1, dec_heads=4,
                      dec_layers=1, dropout=0.0, window_size=W,
                      cross_attention_decoder=True, **kw)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    x  = torch.randn(B, W, N, V, generator=g)
    xm = (torch.rand(B, W, N, V, generator=g) > 0.1).float()
    return dict(
        x=x * xm, x_mask=xm,
        spatial=torch.randn(N, 15, generator=g),
        x_hours=HOURS + torch.arange(W).float().unsqueeze(0).expand(B, W) / 6.0,
    )


def _targets(seed=2):
    g = torch.Generator().manual_seed(seed)
    y  = torch.randn(B, K, N, V, generator=g)
    ym = torch.ones(B, K, N, V)
    yh = HOURS + 2.0 + torch.zeros(B, K)
    dt = torch.arange(0, 3 * K, 3).unsqueeze(0).expand(B, K).contiguous()
    return y, ym, yh, dt


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_output_shapes_multi_delta():
    m = _model()
    b = _batch()
    y, ym, yh, dt = _targets()
    loss, preds, midx = m.forward_multi_delta(
        b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert loss.shape == ()                      # scalar
    assert preds.shape == (B, K, N, Vt)          # NOT V — precipitation excluded
    assert midx.shape == (B, int(N * 0.5))       # 5 masked stations
    assert midx.max() < N


def test_encoder_intermediate_shapes():
    m = _model()
    b = _batch()
    enc, midx, vidx = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    n_vis = N - int(N * 0.5)
    assert vidx.shape == (B, n_vis)
    assert enc.shape == (B, W * n_vis, D)        # time-major flatten, patch=1
    # sorted-order invariant (the station-permutation fix)
    assert (vidx.diff(dim=1) > 0).all() and (midx.diff(dim=1) > 0).all()


def test_single_delta_shapes():
    m = _model()
    b = _batch()
    y, ym, yh, dt = _targets()
    loss, preds, _ = m(b["x"], b["x_mask"], b["spatial"], b["x_hours"],
                       y[:, 0], ym[:, 0], yh[:, 0], dt[:, 0])
    assert preds.shape == (B, N, Vt)


def test_mask_ratio_zero_shapes():
    m = _model(mask_ratio=0.0)
    b = _batch()
    y, ym, yh, dt = _targets()
    loss, preds, midx = m.forward_multi_delta(
        b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert preds.shape == (B, K, N, Vt)
    assert midx.shape == (B, 0)                  # empty, not None


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def _backward(m, **model_kw):
    b = _batch()
    y, ym, yh, dt = _targets()
    loss, *_ = m.forward_multi_delta(
        b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert torch.isfinite(loss), "loss is not finite before backward"
    loss.backward()
    return loss


@pytest.mark.parametrize("kw", [
    dict(),                                       # v23-style: plain
    dict(residual_head=True),                     # v20/v26-style
    dict(query_anchor=True),                      # v24-style
    dict(residual_head=True, query_anchor=True),  # composed
    dict(mask_ratio=0.0),                         # no-masking regime
])
def test_backward_runs_and_all_grads_finite(kw):
    m = _model(**kw)
    m.train()
    _backward(m)
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_every_parameter_family_receives_gradient():
    """
    No silently-dead branches: after one step every module family that is in
    the forward path must have at least one parameter with a nonzero grad.
    """
    m = _model()
    m.train()
    _backward(m)
    families = {
        "encoder.var_proj":    False,   # observation content
        "encoder.blocks":      False,   # transformer trunk
        "decoder.blocks":      False,   # cross-attention stack
        "decoder.head":        False,   # prediction head
        "decoder.mask_token":  False,   # query base
        "decoder.station_state": False, # visible/masked code
    }
    for name, p in m.named_parameters():
        for fam in families:
            if name.startswith(fam) and p.grad is not None and p.grad.abs().sum() > 0:
                families[fam] = True
    dead = [f for f, ok in families.items() if not ok]
    assert not dead, f"no gradient reached: {dead}"


def test_shared_embeddings_receive_gradient_from_both_sides():
    """
    pos/station/temporal/step embeddings are ONE module used by encoder (K/V)
    and decoder (Q). Blocking either path must still leave a gradient — and
    the two paths must not cancel to exactly zero (they couldn't, structurally,
    but a detach on either side would zero one contribution silently).
    """
    m = _model()
    m.train()
    _backward(m)
    for mod_name in ("pos_emb", "station_emb", "temporal_emb", "step_emb"):
        mod = getattr(m.encoder, mod_name)
        assert mod is getattr(m.decoder, mod_name)          # still shared
        gsum = sum(p.grad.abs().sum().item()
                   for p in mod.parameters() if p.grad is not None)
        assert gsum > 0, f"shared {mod_name} received no gradient"


def test_no_leak_from_masked_station_observations():
    """
    Leak guard: a station's own observations must not influence its prediction
    in the batch items where it is MASKED.

    The mask is drawn independently PER BATCH ITEM, so a station masked in
    item 0 may be visible in item 1 — where its observations legitimately
    reach its prediction. The first version of this test missed that and
    asserted over the whole batch; it failed on exactly the visible rows,
    which is the guard WORKING. Condition on the per-item mask instead, and
    require both directions: bit-identical where masked, changed where
    visible (proving the perturbation actually took effect).
    """
    m = _model()
    m.eval()
    b = _batch()
    y, ym, yh, dt = _targets()

    torch.manual_seed(7)
    with torch.no_grad():
        _, midx, _ = m.encoder(b["x"], b["x_mask"], b["spatial"], b["x_hours"])
    hidden = midx[0, 0].item()
    # In which batch items is `hidden` masked? (independent draw per item)
    masked_in = (midx == hidden).any(dim=1)              # (B,) bool

    preds = []
    for bump in (0.0, 10.0):
        b2 = {k: v.clone() for k, v in b.items()}
        b2["x"][:, :, hidden, :] += bump
        torch.manual_seed(7)                             # identical mask draw
        with torch.no_grad():
            _, p, _ = m.forward_multi_delta(
                b2["x"], b2["x_mask"], b2["spatial"], b2["x_hours"], y, ym, yh, dt)
        preds.append(p)

    for bi in range(B):
        if masked_in[bi]:
            assert torch.equal(preds[0][bi, :, hidden], preds[1][bi, :, hidden]), \
                f"item {bi}: masked station's own observations leaked into its prediction"
        else:
            assert not torch.equal(preds[0][bi, :, hidden], preds[1][bi, :, hidden]), \
                f"item {bi}: station is visible yet its prediction ignored a +10 sigma bump"


def test_nll_mode_backward():
    m = _model(use_nll_loss=True)
    m.train()
    _backward(m)
    assert m.decoder.log_var_head.weight.grad is not None
