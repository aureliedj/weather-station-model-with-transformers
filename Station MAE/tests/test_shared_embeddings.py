"""
tests/test_shared_embeddings.py — pos_emb / station_emb / temporal_emb must be
the SAME instance on the encoder and decoder, not two independently-trained
copies of the same idea.

The defect this guards
-----------------------
Before this fix, StationMAEEncoder and StationMAEDecoder each built their own
PositionalEmbedding / StationEmbedding / TemporalEmbedding in __init__. Same
architecture, same input (a station's coordinates / topography / a target
timestamp), but two separate parameter sets. Nothing forced the encoder's
embedding of "station n's position" to equal the decoder's embedding of
"station n's position" — cross-attention had to LEARN an approximate
alignment between them rather than starting from an exact one.

Only StepIndexEmbedding was already shared (passed in from StationMAE), with
this exact reasoning documented next to it: "Two independently-constructed
instances would only share their fixed Fourier frequencies, not their trained
MLP weights ... Sharing the instance makes them provably identical instead."
The same argument applies to pos_emb/station_emb/temporal_emb and is now
applied the same way.

Deliberately NOT shared: VariableProjection (var_proj). The encoder embeds an
observed VALUE; the decoder has no observation to share a value-embedding
with, so there is nothing on the decoder side for it to be identical to.

No torch execution beyond construction and one forward pass; no checkpoint or
dataset needed.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.mae import StationMAE                                     # noqa: E402
from model.encoder import StationMAEEncoder                          # noqa: E402
from model.decoder import StationMAEDecoder                          # noqa: E402
from model.embeddings import NUM_VARIABLES                           # noqa: E402

B, W, N, V = 2, 12, 10, NUM_VARIABLES
D = 32


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("mask_ratio", 0.5)
    return StationMAE(d_model=D, enc_heads=4, enc_layers=1, dec_heads=4,
                      dec_layers=1, dropout=0.0, window_size=W,
                      cross_attention_decoder=True, **kw).eval()


# ---------------------------------------------------------------------------
# The sharing itself
# ---------------------------------------------------------------------------

def test_pos_emb_is_the_same_instance_on_both_sides():
    m = _model()
    assert m.encoder.pos_emb is m.decoder.pos_emb


def test_station_emb_is_the_same_instance_on_both_sides():
    m = _model()
    assert m.encoder.station_emb is m.decoder.station_emb


def test_temporal_emb_is_the_same_instance_on_both_sides():
    m = _model()
    assert m.encoder.temporal_emb is m.decoder.temporal_emb


def test_step_emb_is_still_the_same_instance_on_both_sides():
    """Pins the pre-existing sharing (StepIndexEmbedding) alongside the new one."""
    m = _model()
    assert m.encoder.step_emb is m.decoder.step_emb


def test_sharing_means_shared_weights_not_just_matching_shapes():
    """
    The point of sharing is that a gradient step on one side moves the other.
    Verify identity implies exactly that: mutating a parameter through the
    encoder's reference is visible through the decoder's reference.
    """
    m = _model()
    with torch.no_grad():
        m.encoder.pos_emb.proj[0].weight.add_(1.0)
    assert torch.equal(m.encoder.pos_emb.proj[0].weight, m.decoder.pos_emb.proj[0].weight)


def test_var_proj_has_no_decoder_side_counterpart():
    """
    The explicit exclusion: an observed VALUE has nothing to share with on the
    decoder side, so var_proj must stay encoder-only.
    """
    m = _model()
    assert hasattr(m.encoder, "var_proj")
    assert not hasattr(m.decoder, "var_proj")


# ---------------------------------------------------------------------------
# Standalone construction still works (tests, notebooks, ad-hoc scripts)
# ---------------------------------------------------------------------------

def test_encoder_builds_standalone_without_shared_embeddings():
    enc = StationMAEEncoder(d_model=D, num_heads=4, num_layers=1, mask_ratio=0.5)
    assert enc.pos_emb is not None and enc.station_emb is not None and enc.temporal_emb is not None


def test_decoder_builds_standalone_without_shared_embeddings():
    dec = StationMAEDecoder(d_model=D, num_heads=4, num_layers=1, window_size=W)
    assert dec.pos_emb is not None and dec.station_emb is not None and dec.temporal_emb is not None


def test_two_standalone_encoders_do_not_share_embeddings():
    """Without an explicit shared instance passed in, each build gets its own."""
    enc1 = StationMAEEncoder(d_model=D, num_heads=4, num_layers=1, mask_ratio=0.5)
    enc2 = StationMAEEncoder(d_model=D, num_heads=4, num_layers=1, mask_ratio=0.5)
    assert enc1.pos_emb is not enc2.pos_emb


# ---------------------------------------------------------------------------
# End-to-end: the wiring doesn't break a forward pass
# ---------------------------------------------------------------------------

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


def test_forward_runs_end_to_end_with_shared_embeddings():
    m = _model()
    b = _batch()
    y, ym, yh, dt = _targets()
    with torch.no_grad():
        loss, preds, *_ = m.forward_multi_delta(
            b["x"], b["x_mask"], b["spatial"], b["x_hours"], y, ym, yh, dt)
    assert torch.isfinite(loss) and torch.isfinite(preds).all()


def test_cfg_roundtrip_still_shares_embeddings():
    """from_cfg goes through the same __init__ path — sharing must survive it."""
    cfg = {"d_model": D, "enc_heads": 4, "enc_layers": 1, "dec_heads": 4,
           "dec_layers": 1, "mask_ratio": 0.5, "window": W,
           "cross_attn_decoder": True}
    m = StationMAE.from_cfg(cfg)
    assert m.encoder.pos_emb is m.decoder.pos_emb
    assert m.encoder.station_emb is m.decoder.station_emb
    assert m.encoder.temporal_emb is m.decoder.temporal_emb
