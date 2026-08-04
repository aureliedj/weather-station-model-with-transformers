"""
tests/test_token_balance.py — guards the observation/metadata token invariant.

    A weather token should change about as much when the weather changes
    as when its metadata changes.

Everything here runs at INITIALISATION on synthetic data with a fixed seed, so
it needs no checkpoint, no dataset and no GPU, and it is deterministic.

Why this exists
---------------
The encoder token is the sum of five branches and only one of them sees an
observation.  Before the rebalance, that branch arrived at 0.07% of the token's
variance: permuting every observation in the window moved the token by 3%,
while permuting the station metadata moved it by 64%.  The model trained, and
nothing in the loss, the validation metrics or the sanity checks reported a
problem — the defect was upstream of all of them.  These assertions would have
caught it at the first forward pass.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.embeddings import VariableProjection, NUM_VARIABLES          # noqa: E402
from model.encoder import StationMAEEncoder                             # noqa: E402
from model.token_balance import (                                       # noqa: E402
    token_balance, synthetic_batch, format_report,
    CONTENT_FRACTION_FLOOR, WEATHER_PERM_FLOOR,
)

D_MODEL = 384
V = NUM_VARIABLES


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    return StationMAEEncoder(d_model=D_MODEL, num_heads=8, num_layers=1,
                             dropout=0.0).eval()


@pytest.fixture(scope="module")
def balance(encoder):
    return token_balance(encoder, **synthetic_batch(seed=0))


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def test_content_fraction_above_floor(balance):
    """The data branch must carry a real share of the token's variance."""
    cf = balance["content_fraction"]
    assert cf >= CONTENT_FRACTION_FLOOR, (
        f"content fraction {cf*100:.3f}% < {CONTENT_FRACTION_FLOOR*100:.2f}% — "
        f"the token is essentially pure metadata.\n{format_report(balance)}")


def test_weather_permutation_above_floor(balance):
    """Replacing every observation must visibly move the token."""
    w = balance["permutation"]["weather"]
    assert w >= WEATHER_PERM_FLOOR, (
        f"permuting all observations moved the token by only {w:.4f} "
        f"(floor {WEATHER_PERM_FLOOR}).\n{format_report(balance)}")


def test_weather_comparable_to_metadata(balance):
    """The invariant itself: weather and metadata should be the same order."""
    r = balance["weather_over_metadata"]
    assert 0.3 <= r <= 3.0, (
        f"weather/metadata permutation ratio {r:.3f} outside [0.3, 3.0] — the "
        f"two are not the same order of magnitude.\n{format_report(balance)}")


def test_observation_branch_is_not_mostly_constant(balance):
    """var_type_embedding used to make ~93% of this branch a fixed offset."""
    const, weather = balance["constant_rms"], balance["weather_rms"]
    assert const < weather, (
        f"the observation branch is {const:.4f} constant vs {weather:.4f} "
        f"data — most of it carries no information.")


# ---------------------------------------------------------------------------
# The specific defects that were fixed, pinned so they cannot return
# ---------------------------------------------------------------------------

def test_var_weights_init_scale():
    """uniform(-1,1) (std 0.577), not xavier on the storage shape (std 0.072)."""
    torch.manual_seed(0)
    vp = VariableProjection(num_vars=V, d_model=D_MODEL)
    s = vp.var_weights.std().item()
    xavier = math.sqrt(6.0 / (V + D_MODEL)) / math.sqrt(3)
    assert s > 4 * xavier, (
        f"var_weights std {s:.4f} is near the old xavier value {xavier:.4f}")
    assert 0.4 < s < 0.75, f"var_weights std {s:.4f} outside the expected ~0.577"


def test_divisor_is_sqrt_v():
    """Scale must be independent of the variable count; /V is not."""
    outs = []
    for nv in (3, 6, 12):
        torch.manual_seed(0)
        vp = VariableProjection(num_vars=nv, d_model=D_MODEL)
        g = torch.Generator().manual_seed(1)
        x = torch.randn(64, 32, nv, generator=g)
        outs.append(vp(x, torch.ones_like(x)).std().item())
    assert max(outs) / min(outs) < 1.25, (
        f"observation-branch scale varies with the variable count: {outs} — "
        f"the divisor is not sqrt(V)")


def test_var_type_embedding_removed():
    vp = VariableProjection(num_vars=V, d_model=D_MODEL)
    assert not hasattr(vp, "var_type_embedding"), (
        "var_type_embedding is redundant (it absorbs into var_biases and "
        "var_absent_embedding) and should not have come back")


def test_absent_differs_from_zero_reading_at_init():
    """
    'Sensor missing' and 'sensor reads its station mean' must not be the same
    vector at step 0.  x = 0 is the modal value under per-station
    normalisation, so this collision sat on the most common reading.
    """
    torch.manual_seed(0)
    vp = VariableProjection(num_vars=V, d_model=D_MODEL)
    zero = torch.zeros(1, 1, V)
    reads_zero = vp(zero, torch.ones_like(zero))
    absent     = vp(zero, torch.zeros_like(zero))
    assert (reads_zero - absent).abs().max() > 1e-6, (
        "absent and zero-reading produce an identical embedding at init")


# ---------------------------------------------------------------------------
# Migration: old checkpoints must keep their exact behaviour
# ---------------------------------------------------------------------------

def test_old_checkpoint_migrates_exactly():
    """
    Folding var_type_embedding into the biases and rescaling by 1/sqrt(V) is an
    algebraic identity, so a pre-rebalance checkpoint must load bit-for-bit.
    """
    torch.manual_seed(0)
    vp = VariableProjection(num_vars=V, d_model=D_MODEL)
    g = torch.Generator().manual_seed(2)
    w = torch.randn(V, D_MODEL, generator=g) * 0.28
    b = torch.randn(V, D_MODEL, generator=g) * 0.013
    t = torch.randn(V, D_MODEL, generator=g)          # the N(0,1) offender
    a = torch.randn(V, D_MODEL, generator=g) * 0.025

    x = torch.randn(8, 40, V, generator=g)
    m = (torch.rand(8, 40, V, generator=g) > 0.1).float()

    # what the OLD module computed
    proj  = x.unsqueeze(-1) * w + b + t
    mixed = proj * m.unsqueeze(-1) + (a + t) * (1 - m.unsqueeze(-1))
    expected = mixed.sum(-2) / float(V)

    vp.load_state_dict({"var_weights": w, "var_biases": b,
                        "var_type_embedding.weight": t,
                        "var_absent_embedding": a})
    got = vp(x, m)
    assert torch.allclose(got, expected, atol=1e-5), (
        f"migration is not exact: max diff {(got-expected).abs().max():.3e}")


def test_new_checkpoint_roundtrips(encoder):
    """A checkpoint saved by the new code must load without migration."""
    torch.manual_seed(1)
    vp = VariableProjection(num_vars=V, d_model=D_MODEL)
    sd = vp.state_dict()
    assert "var_type_embedding.weight" not in sd
    vp2 = VariableProjection(num_vars=V, d_model=D_MODEL)
    vp2.load_state_dict(sd)
    x = torch.randn(4, 9, V)
    m = (torch.rand(4, 9, V) > 0.1).float()
    assert torch.allclose(vp(x, m), vp2(x, m))


# ---------------------------------------------------------------------------
# The decomposition must not drift from the encoder
# ---------------------------------------------------------------------------

def test_components_match_encoder_build_tokens(encoder):
    """token_components() raises if it ever disagrees with _build_tokens()."""
    b = synthetic_batch(seed=3)
    token_balance(encoder, **b)      # the assertion lives inside


# ---------------------------------------------------------------------------
# v18 — Fourier (PLR) value embedding and the shared wind encoder
# ---------------------------------------------------------------------------

def _vp(**kw):
    torch.manual_seed(0)
    return VariableProjection(num_vars=V, d_model=D_MODEL, **kw)


def test_default_is_still_v17():
    """The v17 path must remain the default and keep its exact parameter set."""
    vp = _vp()
    assert vp.value_embedding == "linear" and vp.wind_pair is None
    assert hasattr(vp, "var_weights") and not hasattr(vp, "slot_maps")


@pytest.mark.parametrize("mode", ["mlp", "fourier"])
def test_v18_modes_are_nonlinear_and_beat_rank_one(mode):
    """Both v18 modes must be nonlinear in the value; v17 is exactly linear."""
    vp = _vp(value_embedding=mode)
    x  = torch.randn(64, 16, V); m = torch.ones_like(x)
    assert (vp(2 * x, m) - 2 * vp(x, m)).abs().mean() > 1e-3
    lin = _vp()
    assert (lin(2 * x, m) - 2 * lin(x, m)).abs().max() < 1e-4, "v17 should be linear"


def _signal_and_constant(vp, x, m):
    """
    Split the branch into its token-independent and data-dependent parts.
    The constant is the branch evaluated at x = 0 with every sensor present.
    """
    z = torch.zeros(1, 1, V)
    const = vp(z, torch.ones_like(z))
    out   = vp(x, m)
    return (out - const), const


@pytest.mark.parametrize("mode", ["mlp", "fourier"])
def test_v18_signal_scale_matches_v17(mode):
    """
    Match on the SIGNAL, not on RMS.

    An earlier gain matched RMS exactly while 91% of that RMS was an uncentred
    GELU constant — magnitude with no information — and content fraction fell
    from 18.9% to 6.5%. RMS cannot see that; variance across tokens can.
    """
    g = torch.Generator().manual_seed(5)
    x = torch.randn(256, 64, V, generator=g); m = torch.ones_like(x)
    s_v18, _ = _signal_and_constant(_vp(value_embedding=mode), x, m)
    s_v17, _ = _signal_and_constant(_vp(), x, m)
    r = s_v18.std().item() / s_v17.std().item()
    assert 0.6 < r < 1.7, f"{mode} SIGNAL std ratio {r:.2f} outside [0.6, 1.7]"


@pytest.mark.parametrize("mode", ["linear", "mlp", "fourier"])
def test_branch_carries_no_large_constant(mode):
    """
    No mode may reintroduce a token-independent offset. This is the defect
    var_type_embedding had, and that the uncentred GELU MLP recreated.
    """
    g = torch.Generator().manual_seed(6)
    x = torch.randn(128, 32, V, generator=g); m = torch.ones_like(x)
    sig, const = _signal_and_constant(_vp(value_embedding=mode), x, m)
    ratio = const.pow(2).mean().sqrt().item() / sig.std().item()
    assert ratio < 0.25, (
        f"{mode}: constant part is {ratio:.2f}x the signal — magnitude "
        f"without information")


def test_mlp_thresholds_land_inside_the_value_range():
    """
    Each hidden unit bends at x = -b/a. PyTorch's default init makes that
    Cauchy-distributed, wasting a sixth of the basis outside +-3 sigma.
    """
    vp = _vp(value_embedding="mlp")
    lin1 = vp.slot_maps[0][0]
    a = lin1.weight.squeeze(-1); b = lin1.bias
    assert torch.allclose(a.abs(), torch.ones_like(a)), "|a| should be exactly 1"
    th = (-b / a)
    assert (th.abs() < 3.0 + 1e-4).all(), "thresholds outside +-3 sigma"


def test_fourier_is_nonlinear_in_the_value():
    """The whole point: e(2x) must not equal 2·e(x)."""
    vp = _vp(value_embedding="fourier")
    x = torch.randn(64, 16, V)
    m = torch.ones_like(x)
    lin_err = (vp(2 * x, m) - 2 * vp(x, m)).abs().mean()
    assert lin_err > 1e-3, "fourier value embedding is behaving linearly"


def test_fourier_separates_values_one_sigma_apart():
    """Different value RANGES must get different directions, not just scales."""
    vp = _vp(value_embedding="fourier")
    a = torch.zeros(1, 1, V)
    e0 = vp(a, torch.ones_like(a)).squeeze()
    e1 = vp(a + 1.0, torch.ones_like(a)).squeeze()
    cos = torch.nn.functional.cosine_similarity(e0, e1, dim=0).item()
    assert cos < 0.99, f"codes 1 sigma apart have cosine similarity {cos:.4f}"


def test_wind_pair_forms_one_slot():
    vp = _vp(wind_pair=(3, 4))
    assert vp.num_slots == V - 1
    assert (3, 4) in vp.slots
    assert vp.var_absent_embedding.shape[0] == V - 1


def _wind_nonseparability(vp):
    """
    For any additively separable map e(u,v) = f(u) + h(v) the cross-swap
    identity  e(u1,v1) + e(u2,v2) == e(u1,v2) + e(u2,v1)  holds EXACTLY.
    A joint map breaks it. This — not asymmetry in (u,v) — is the property
    the shared wind encoder buys: independent per-variable maps are already
    asymmetric simply because W_u != W_v, which tests nothing.
    """
    def e(u, v):
        x = torch.zeros(1, 1, V); x[..., 3], x[..., 4] = u, v
        return vp(x, torch.ones_like(x))
    lhs = e(1.0, 2.0) + e(-3.0, 0.5)
    rhs = e(1.0, 0.5) + e(-3.0, 2.0)
    return (lhs - rhs).abs().max().item()


def test_wind_pair_is_not_additively_separable():
    assert _wind_nonseparability(_vp(value_embedding="fourier", wind_pair=(3, 4))) > 1e-3


def test_without_pairing_wind_is_exactly_separable():
    """The control: this is what the shared encoder is fixing."""
    assert _wind_nonseparability(_vp(value_embedding="fourier")) < 1e-5


def test_wind_slot_absent_only_when_both_absent():
    """A slot counts as present only if every member is present."""
    vp = _vp(wind_pair=(3, 4))
    x = torch.zeros(1, 1, V)
    full = torch.ones(1, 1, V)
    half = full.clone(); half[..., 4] = 0.0          # v missing, u present
    assert not torch.allclose(vp(x, full), vp(x, half))


def test_absent_code_is_comparable_to_a_present_contribution():
    """
    Regression on my own under-scaling: absent was init at 0.02 while a present
    variable contributes ~0.53, making 'missing' 26x weaker than any reading.
    """
    vp = _vp()
    assert vp.var_absent_embedding.std().item() > 0.2, (
        f"absent code std {vp.var_absent_embedding.std():.3f} is far below a "
        f"typical present contribution (~0.5)")


def test_old_checkpoint_refused_by_v18_layout():
    """A pre-rebalance checkpoint has no algebraic rewrite into the v18 layout."""
    vp = _vp(value_embedding="fourier")
    sd = {"var_weights": torch.randn(V, D_MODEL),
          "var_biases": torch.zeros(V, D_MODEL),
          "var_type_embedding.weight": torch.randn(V, D_MODEL),
          "var_absent_embedding": torch.zeros(V, D_MODEL)}
    with pytest.raises(RuntimeError, match="predates the observation-token rebalance"):
        vp.load_state_dict(sd, strict=False)
