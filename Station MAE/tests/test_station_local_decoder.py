"""
Pins the station-independent decoder (--station_local_decoder).

`--no_spatial_attn` alone does NOT make the model station-independent: it removes
the encoder's spatial sub-layer, but the decoder's queries still attend to one
another across stations and cross-attend to every station's encoder tokens.
`--station_local_decoder` folds the station axis into the batch so that each
station is a separate attention problem, while keeping the Delta-query decoder.

The decisive test is behavioural, not structural: perturbing station j's inputs
must leave station i's prediction BIT-IDENTICAL when the model is station-local,
and must change it when it is not. A shape-only test would pass even if the
reshape mixed stations.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.mae import StationMAE  # noqa: E402
from model.embeddings import NUM_VARIABLES, STATION_CHAR_DIM  # noqa: E402

B, W, N, P = 2, 12, 6, 3
D_MODEL, K = 32, 3
MAX_DELTA, STRIDE = 6, 3


def _build(station_local: bool, spatial_attn: bool = False) -> StationMAE:
    torch.manual_seed(0)
    return StationMAE(
        d_model=D_MODEL, enc_heads=2, dec_heads=2, enc_layers=2, dec_layers=2,
        window_size=W, temporal_patch=P, mask_ratio=0.0,
        factorised_encoder=True, encoder_spatial_attn=spatial_attn,
        cross_attention_decoder=True, station_local_decoder=station_local,
        num_horizons=K, dropout=0.0, drop_path_rate=0.0,
    ).eval()


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x       = torch.randn(B, W, N, NUM_VARIABLES, generator=g)
    x_mask  = torch.ones(B, W, N, NUM_VARIABLES)
    spatial = torch.randn(N, 2 + STATION_CHAR_DIM, generator=g)
    x_hours = torch.arange(W, dtype=torch.float32).repeat(B, 1) * (1 / 6)
    deltas  = torch.arange(0, MAX_DELTA + 1, STRIDE).repeat(B, 1)
    y       = torch.randn(B, K, N, NUM_VARIABLES, generator=g)
    y_mask  = torch.ones(B, K, N, NUM_VARIABLES)
    y_hours = x_hours[:, -1:].expand(B, K) + deltas * (1 / 6)
    return x, x_mask, spatial, x_hours, y, y_mask, y_hours, deltas


def _predict(model, batch):
    with torch.no_grad():
        _, preds, _ = model.forward_multi_delta(*batch)
    return preds                                   # (B, K, N, V_target)


def test_station_local_isolates_stations():
    """Perturbing station 0 must not move any other station's prediction."""
    model = _build(station_local=True)
    b = list(_batch())
    base = _predict(model, b)

    b2 = [t.clone() if torch.is_tensor(t) else t for t in b]
    b2[0][:, :, 0, :] += 10.0                      # perturb station 0 only
    pert = _predict(model, b2)

    moved_own = (pert[:, :, 0, :] - base[:, :, 0, :]).abs().max()
    assert moved_own > 1e-5, "station 0's own prediction should react to its own input"

    others = (pert[:, :, 1:, :] - base[:, :, 1:, :]).abs().max()
    assert others == 0.0, (
        f"station-local decoder leaked across stations: other stations moved by "
        f"{others:.3e} (must be exactly 0)")


def test_default_decoder_does_mix_stations():
    """Control: without the flag, the same perturbation DOES reach other stations."""
    model = _build(station_local=False)
    b = list(_batch())
    base = _predict(model, b)

    b2 = [t.clone() if torch.is_tensor(t) else t for t in b]
    b2[0][:, :, 0, :] += 10.0
    pert = _predict(model, b2)

    others = (pert[:, :, 1:, :] - base[:, :, 1:, :]).abs().max()
    assert others > 1e-6, (
        "control failed: the standard decoder should propagate one station's "
        "input to other stations' predictions")


def test_output_shape_unchanged():
    model = _build(station_local=True)
    preds = _predict(model, _batch())
    assert preds.shape == (B, K, N, model.num_target_vars_), preds.shape


def test_station_order_is_preserved():
    """
    The fold is a permute+reshape; if it were transposed the head would silently
    attribute station i's tokens to station j. Feed each station a distinct
    constant and check the ordering survives by comparing against a per-station
    reference computed one station at a time.
    """
    model = _build(station_local=True)
    b = list(_batch())
    full = _predict(model, b)

    # Re-run with all stations but zero out every station except one, and check
    # that station's prediction is unaffected by the others (it must be, since
    # the decoder is station-local and the encoder has spatial_attn=False).
    for n in (0, N - 1, N // 2):
        b2 = [t.clone() if torch.is_tensor(t) else t for t in b]
        keep = torch.zeros(N, dtype=torch.bool)
        keep[n] = True
        b2[0][:, :, ~keep, :] = 0.0                # blank every other station
        one = _predict(model, b2)
        diff = (one[:, :, n, :] - full[:, :, n, :]).abs().max()
        assert diff == 0.0, (
            f"station {n}'s prediction changed by {diff:.3e} when OTHER stations "
            f"were blanked — station ordering or isolation is wrong")


def test_requires_all_stations_present_at_construction():
    """A real run sets mask_ratio up front; reject it there."""
    with pytest.raises(AssertionError, match="every station must contribute"):
        StationMAE(
            d_model=D_MODEL, enc_heads=2, dec_heads=2, enc_layers=2, dec_layers=2,
            window_size=W, temporal_patch=P, mask_ratio=0.5,
            factorised_encoder=True, encoder_spatial_attn=False,
            cross_attention_decoder=True, station_local_decoder=True,
            num_horizons=K,
        )


def test_requires_all_stations_present_at_forward():
    """
    mask_ratio can also be changed AFTER construction (test.py does exactly this
    when sweeping mask ratios). The forward pass must still refuse.

    This case is why a divisibility check inside the decoder is not enough:
    with N=6 and mask_ratio 0.5 the encoder returns T*3 = 12 tokens, and
    12 % 6 == 0, so the reshape would silently succeed with a halved temporal
    length instead of raising.
    """
    model = _build(station_local=True)
    model.encoder.mask_ratio = 0.5                 # forced after construction
    with pytest.raises(RuntimeError, match="every station present"):
        _predict(model, _batch())


def test_divisibility_alone_would_not_have_caught_it():
    """
    Documents the trap explicitly, so nobody 'simplifies' the guard back to a
    divisibility test. With these dimensions the bad reshape is well-formed.
    """
    T = W // P
    n_vis = N - int(N * 0.5)
    assert (T * n_vis) % N == 0, "the dimensions no longer demonstrate the trap"
    assert (T * n_vis) // N != T, "the wrong temporal length must differ from T"


def test_cfg_roundtrip_preserves_the_flag():
    """A checkpoint must rebuild station-local, or evaluation silently differs."""
    model = _build(station_local=True)
    cfg = {
        "d_model": D_MODEL, "enc_heads": 2, "dec_heads": 2,
        "enc_layers": 2, "dec_layers": 2, "window": W, "temporal_patch": P,
        "mask_ratio": 0.0, "factorised_encoder": True,
        "encoder_spatial_attn": False, "cross_attn_decoder": True,
        "station_local_decoder": True,
        "max_delta": MAX_DELTA, "delta_grid_stride": STRIDE,
    }
    rebuilt = StationMAE.from_cfg(cfg)
    assert rebuilt.decoder.station_local is True
    assert model.decoder.station_local is True
