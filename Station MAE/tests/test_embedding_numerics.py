"""
tests/test_embedding_numerics.py — the Fourier banks must actually resolve the
wavelengths they advertise.

The defect this guards
----------------------
TemporalEmbedding encodes time as HOURS SINCE 1970 (~4.9e5 in 2026) and spans
wavelengths from lambda_min = 1/6 h (10 min) to lambda_max = 1 year. The angle
handed to cos/sin at the short end is

    2*pi * 4.9e5 / (1/6) ~= 1.85e7 radians

float32 has a 24-bit mantissa, so the absolute rounding error there is
~1.85e7 * 6e-8 ~= 2.2 radians. Measured phase error against float64 at a real
2026 timestamp, before the fix:

    lambda = 10 min   ->  126 deg   (cos off by 0.64 on a [-1,1] range)
    lambda = 21 min   ->   61 deg
    lambda = 43 min   ->   30 deg
    lambda = 12.9 h   ->  1.6 deg   (usable from here down)

i.e. the features meant to resolve the dataset's native 10-minute step were
the least trustworthy in the bank, and roughly a third of the wavelengths were
degraded or pure noise. The fix computes the expansion in float64 and casts the
bounded cos/sin outputs back.

The other four Fourier banks do NOT have this problem — their inputs are small
(lead time <= 6 h, step index <= 107, normalised coordinates ~3 sigma, values
~26 sigma) so the largest angle is ~700 rad and the float32 error ~1e-4 rad.
Those are pinned here too, so that raising a lambda range or switching an input
to an absolute scale cannot silently reintroduce the same defect elsewhere.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.embeddings import (                                       # noqa: E402
    TemporalEmbedding,
    DeltaTimeEmbedding,
    StepIndexEmbedding,
    PositionalEmbedding,
    TEMPORAL_WAVELENGTHS_H,
    TEMPORAL_FOURIER_DIM,
)

# Hours since 1970 for a timestamp inside the dataset's range (2026-08).
HOURS_2026 = 490896.0
D = 32


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------

def test_temporal_fourier_matches_float64_reference():
    """
    Every feature must match an exact float64 computation. Before the fix the
    10-minute wavelength was off by ~0.64 on a [-1, 1] range.
    """
    m = TemporalEmbedding(d_model=D)
    hours = torch.tensor([HOURS_2026], dtype=torch.float32)
    got = m._fourier(hours)[0].double()               # (32,)

    lam = m.lambdas.double()                          # (16,)
    ang = 2.0 * math.pi * torch.tensor(HOURS_2026, dtype=torch.float64) / lam
    want = torch.cat([ang.cos(), ang.sin()], dim=-1)  # (32,) — NO trailing [0]:
    # the first version indexed this down to a SCALAR (cos at lambda=24, which
    # is 1.0 because 490896 = 24 * 20454 exactly) and broadcast it against all
    # 32 features — the "err = 2.0" failure was |sin(lambda=192) - 1| = 2, a
    # silent-broadcasting bug in the TEST, not a precision bug in the module.
    assert want.shape == got.shape

    err = (got - want).abs().max().item()
    assert err < 1e-5, (
        f"temporal Fourier features are off by {err:.4f} from the exact value; "
        f"the expansion is being computed in reduced precision")


def test_the_shortest_wavelength_is_accurate():
    """
    Targeted regression on the SHORTEST wavelength, which is always the one
    with the largest angle and therefore the first to lose precision. Pin it
    so a refactor that reorders or drops the float64 cast fails loudly here.
    """
    m = TemporalEmbedding(d_model=D)
    hours = torch.tensor([HOURS_2026], dtype=torch.float32)
    j = int(m.lambdas.argmin())
    got = float(m._fourier(hours)[0, j])          # cos at the smallest lambda

    lam = float(m.lambdas[j].double())
    want = math.cos(2.0 * math.pi * HOURS_2026 / lam)
    assert abs(got - want) < 1e-5, (
        f"cos at lambda_min={lam:.4f}h: got {got:.6f}, exact {want:.6f}")


def test_nearby_timestamps_produce_different_features():
    """
    The functional consequence: two timestamps 10 minutes apart must be
    distinguishable at the short wavelengths. Under float32 the difference was
    dominated by rounding noise rather than by the 10-minute offset.
    """
    m = TemporalEmbedding(d_model=D)
    a = m._fourier(torch.tensor([HOURS_2026], dtype=torch.float32))
    b = m._fourier(torch.tensor([HOURS_2026 + 1.0 / 6.0], dtype=torch.float32))
    assert not torch.allclose(a, b, atol=1e-6)


def test_output_dtype_follows_the_input():
    """float64 internally, but the module must not leak float64 downstream."""
    m = TemporalEmbedding(d_model=D)
    h32 = torch.tensor([HOURS_2026], dtype=torch.float32)
    assert m._fourier(h32).dtype == torch.float32
    assert m(h32).dtype == torch.float32


def test_forward_still_works_on_batched_shapes():
    m = TemporalEmbedding(d_model=D)
    for shape in [(4,), (4, 12), (2, 3)]:
        h = torch.full(shape, HOURS_2026, dtype=torch.float32)
        assert m(h).shape == shape + (D,)


# ---------------------------------------------------------------------------
# The other banks: small inputs, so float32 is sufficient — pin that it stays so
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,module,x_max", [
    ("delta",    DeltaTimeEmbedding(d_model=D), 36.0),    # steps -> 6 h
    ("step",     StepIndexEmbedding(d_model=D), 107.0),   # W-1+max_delta
])
def test_small_input_banks_are_within_float32_tolerance(name, module, x_max):
    """
    These are fine today only because their inputs are small. If someone raises
    lambda_max, switches to an absolute epoch, or widens the horizon, the same
    catastrophic-cancellation failure appears here — this test is the tripwire.
    """
    lam = module.lambdas.double()
    max_angle = float(2.0 * math.pi * x_max / lam.min())
    f32_err = max_angle * torch.finfo(torch.float32).eps
    assert f32_err < 0.05, (
        f"{name} bank now reaches {max_angle:.2e} rad, giving a float32 phase "
        f"error of {f32_err:.3f} rad. Compute this bank in float64 too, the way "
        f"TemporalEmbedding does.")


def test_position_bank_is_within_float32_tolerance():
    m = PositionalEmbedding(d_model=D)
    x_max = 5.0                       # normalised coordinates, generous bound
    max_angle = float(2.0 * math.pi * x_max / m.lambdas.double().min())
    assert max_angle * torch.finfo(torch.float32).eps < 0.05


def test_temporal_and_step_banks_are_not_accidentally_the_same_scale():
    """
    Sanity on the design: absolute time and within-window step index are
    different coordinate systems. If they ever coincide, one of them is
    misconfigured.
    """
    t = TemporalEmbedding(d_model=D)
    s = StepIndexEmbedding(d_model=D)
    assert not torch.allclose(t.lambdas.float(), s.lambdas.float())


# ---------------------------------------------------------------------------
# Wavelength PLACEMENT — a separate defect from precision
# ---------------------------------------------------------------------------
#
# The two are independent. Precision is "the features you have are noisy";
# placement is "the feature you need was never in the set". Computing a basis
# more accurately does not add a 24 h component to it — before this fix the
# diurnal fit was R^2 = 0.0081 in float32 and 0.0054 in float64.

def _fit_r2(lambdas, period_h, days=60):
    """Least-squares R^2 of a pure `period_h` cycle from the cos/sin basis."""
    t = np.arange(0, 24 * days, 1 / 6)
    F = np.concatenate([np.cos(2 * np.pi * t[:, None] / lambdas),
                        np.sin(2 * np.pi * t[:, None] / lambdas)], axis=1)
    y = np.cos(2 * np.pi * t / period_h)
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    return 1.0 - ((F @ coef - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


@pytest.mark.parametrize("name,period", [
    ("diurnal",      24.0),
    ("semi-diurnal", 12.0),      # also the atmospheric pressure tide
    ("ter-diurnal",   8.0),
    ("annual",     8766.0),
    ("semi-annual",4383.0),
])
def test_physically_meaningful_cycles_are_representable(name, period):
    """
    The regression that matters. A log-spaced sweep from 1/6 h to 1 year put
    wavelengths at 12.889 h and 26.605 h — incommensurate with 24 h, so the
    DOMINANT periodicity in surface weather could not be expressed at all
    (R^2 = 0.005) while the annual cycle happened to land exactly (R^2 = 1.0).
    """
    lam = np.asarray(TEMPORAL_WAVELENGTHS_H, dtype=np.float64)
    r2 = _fit_r2(lam, period)
    assert r2 > 0.99, (
        f"the {name} cycle ({period} h) is not representable from the temporal "
        f"basis (R^2={r2:.4f}). Check TEMPORAL_WAVELENGTHS_H still contains it "
        f"or a set of its harmonics.")


def test_wavelength_count_matches_the_fourier_dim():
    assert 2 * len(TEMPORAL_WAVELENGTHS_H) == TEMPORAL_FOURIER_DIM
    m = TemporalEmbedding(d_model=D)
    assert m.lambdas.numel() == TEMPORAL_FOURIER_DIM // 2
    assert m._fourier(torch.tensor([HOURS_2026])).shape[-1] == TEMPORAL_FOURIER_DIM


def test_no_sub_4h_absolute_wavelengths():
    """
    Absolute sub-hourly phase ("22 min past the hour, measured from 1970")
    carries no meteorological signal, and it is where the float32 precision
    failure was worst. Relative position in the window is StepIndexEmbedding's
    job; lead time is DeltaTimeEmbedding's.
    """
    assert min(TEMPORAL_WAVELENGTHS_H) >= 4.0


def test_a_custom_wavelength_set_is_accepted_and_validated():
    m = TemporalEmbedding(d_model=D, fourier_dim=4, wavelengths=(24.0, 12.0))
    assert m.lambdas.numel() == 2
    with pytest.raises(AssertionError, match="wavelengths"):
        TemporalEmbedding(d_model=D, fourier_dim=32, wavelengths=(24.0,))


# ---------------------------------------------------------------------------
# Provably-constant features: cos(2*pi*k) = 1 for integer k
# ---------------------------------------------------------------------------

def test_step_index_has_no_constant_feature():
    """
    lambda = 1 step was dead: the input is an INTEGER index, so every token
    got cos = 1, sin = 0. Two of 32 features were a fixed (1, 0) forever.
    """
    m = StepIndexEmbedding(d_model=D)
    k = torch.arange(0, 108)
    f = m._fourier(k)                       # (108, fourier_dim)
    var = f.var(dim=0)
    assert (var > 1e-8).all(), (
        f"{int((var <= 1e-8).sum())} constant feature(s) in StepIndexEmbedding; "
        f"lambda_min must exceed the integer-grid Nyquist of 2 steps")


def test_delta_time_has_no_constant_feature_on_the_fixed_grid():
    """
    lambda = 1/6 h was dead: fixed_grid horizons are multiples of 3 steps
    (0.5 h), so every lead time was an exact integer number of wavelengths.
    """
    m = DeltaTimeEmbedding(d_model=D)
    delta_steps = torch.arange(0, 39, 3)    # 0, 3, ..., 36 -> 0 .. 6 h
    hours = delta_steps.float() * m.step_size_h
    ang = 2.0 * math.pi * hours.unsqueeze(-1) / m.lambdas
    f = torch.cat([ang.cos(), ang.sin()], dim=-1)
    var = f.var(dim=0)
    assert (var > 1e-8).all(), (
        f"{int((var <= 1e-8).sum())} constant feature(s) in DeltaTimeEmbedding; "
        f"lambda_min must exceed 2 x (delta_grid_stride x 10 min)")
