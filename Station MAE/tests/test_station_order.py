"""
tests/test_station_order.py — the station axis keeps canonical ORDER while the
mask stays random.

The defect
----------
``_mask_stations`` built both groups by slicing ``argsort(rand)``::

    shuffle_idx     = torch.argsort(torch.rand(B, N), dim=1)
    visible_indices = shuffle_idx[:, num_masked:]
    masked_indices  = shuffle_idx[:, :num_masked]

so the sequence handed to the encoder was PERMUTED — including at
``mask_ratio 0``, where ``num_masked`` is 0 and nothing is dropped at all.
Nothing un-permuted it: ``forward`` returns ``visible_idx`` and every caller
ignored it.

Why it went unnoticed for so long: with a query decoder it genuinely does not
matter. The encoder output is only a key/value set, attention over the station
axis is permutation-equivariant, and station identity travels inside each token
rather than in its position. Spatial attention here is FULL over N — the
windowing options apply to the temporal axis — so there is no positional
structure on the station axis to disturb.

It becomes fatal the moment anything reads the axis positionally. The v22
direct head does::

    h = encoded.view(B, T, n_stations, d)
    s = h[:, -1]                    # row j assumed to be station j

Under a permutation, row j is station ``visible_indices[b, j]``, redrawn every
step, so station j's target would be matched against another station's tokens.
The only solution consistent across draws is a station-independent one, i.e.
mean collapse — for a reason unrelated to the architecture being tested.

The fix sorts each group. The mask stays random (which stations are hidden is
redrawn every forward pass); only the ORDER becomes deterministic. At
mask_ratio 0 ``visible_indices`` is exactly ``arange(N)``; above 0 it is a
monotone subsequence.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.encoder import StationMAEEncoder                          # noqa: E402

B, W, N, D = 3, 8, 20, 64


def _enc(mask_ratio):
    torch.manual_seed(0)
    return StationMAEEncoder(d_model=D, num_heads=4, num_layers=1,
                             dropout=0.0, mask_ratio=mask_ratio).eval()


def _tokens(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, W, N, D, generator=g)


# ---------------------------------------------------------------------------
# Order — the property that was broken
# ---------------------------------------------------------------------------

def test_order_is_identity_at_mask_ratio_zero():
    """
    The case that silently ruined the direct head: nothing is masked, so the
    sequence must be the stations in their own order.
    """
    _, masked, visible = _enc(0.0)._mask_stations(_tokens())
    assert masked.shape[1] == 0
    expected = torch.arange(N).expand(B, N)
    assert torch.equal(visible, expected), (
        f"expected arange({N}) per batch item, got {visible[0][:8].tolist()}…")


@pytest.mark.parametrize("mr", [0.25, 0.5, 0.75])
def test_both_groups_are_ascending_when_masking(mr):
    _, masked, visible = _enc(mr)._mask_stations(_tokens())
    for name, idx in (("visible", visible), ("masked", masked)):
        if idx.shape[1] < 2:
            continue
        assert (idx[:, 1:] > idx[:, :-1]).all(), \
            f"{name} indices are not strictly ascending"


def test_tokens_are_gathered_in_the_reported_order():
    """
    The returned indices must actually describe the returned tokens — the
    invariant every positional readout depends on.
    """
    tok = _tokens(3)
    vis_tok, _, visible = _enc(0.5)._mask_stations(tok)
    for b in range(B):
        assert torch.equal(vis_tok[b], tok[b][:, visible[b], :])


# ---------------------------------------------------------------------------
# Randomness — the property that must SURVIVE the fix
# ---------------------------------------------------------------------------

def test_which_stations_are_masked_is_still_random():
    """
    Sorting must not turn the mask into a fixed prefix. Two passes over the
    same input should hide different stations.
    """
    enc, tok = _enc(0.5), _tokens()
    sets = {tuple(enc._mask_stations(tok)[1][0].tolist()) for _ in range(12)}
    assert len(sets) > 1, "the masked set never changed across 12 passes"


def test_the_mask_is_not_a_contiguous_block():
    """
    A sorted set of random draws is scattered, not the first k stations —
    which would make masking trivially predictable from position.
    """
    enc, tok = _enc(0.5), _tokens()
    masked = enc._mask_stations(tok)[1][0]
    n_masked = masked.shape[0]
    assert not torch.equal(masked, torch.arange(n_masked)), \
        "masked stations are exactly the first k — the draw is not random"


def test_batch_items_get_different_masks():
    enc, tok = _enc(0.5), _tokens()
    masked = enc._mask_stations(tok)[1]
    rows = {tuple(masked[b].tolist()) for b in range(B)}
    assert len(rows) > 1, "every batch item was masked identically"


# ---------------------------------------------------------------------------
# The partition is still a partition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mr", [0.0, 0.5, 0.75])
def test_masked_and_visible_partition_all_stations(mr):
    _, masked, visible = _enc(mr)._mask_stations(_tokens())
    assert masked.shape[1] + visible.shape[1] == N
    for b in range(B):
        both = torch.cat([masked[b], visible[b]]).sort().values
        assert torch.equal(both, torch.arange(N)), \
            "masked and visible do not tile the station set exactly once"
