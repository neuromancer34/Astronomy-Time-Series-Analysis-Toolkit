"""Tests for astrotime.phasefold."""

import numpy as np
import pytest

from astrotime.lightcurve import LightCurve
from astrotime.phasefold import phase_fold


def test_phase_values_are_in_zero_to_one():
    time = np.linspace(0, 10, 50)
    lc = LightCurve(time=time, flux=np.ones(50), flux_err=np.zeros(50))

    folded = phase_fold(lc, period=3.0)

    assert (folded.time >= 0).all()
    assert (folded.time < 1).all()


def test_folding_aligns_points_one_period_apart():
    # Two points exactly one period apart should land at the same phase.
    # With t0 defaulting to time.min() (0.5 here), the first point's
    # phase is exactly 0.0, not 0.125 -- the earlier value assumed the
    # old fixed t0=0.0 default.
    period = 4.0
    time = np.array([0.5, 0.5 + period, 0.5 + 2 * period])
    flux = np.array([10.0, 20.0, 30.0])
    lc = LightCurve(time=time, flux=flux, flux_err=np.zeros(3))

    folded = phase_fold(lc, period=period)

    np.testing.assert_allclose(folded.time, [0.0, 0.0, 0.0])


def test_folded_lightcurve_is_sorted_by_phase():
    time = np.array([0.9, 0.1, 0.5]) * 4.0  # period = 4.0, phases 0.9/0.1/0.5
    lc = LightCurve(time=time, flux=np.array([1.0, 2.0, 3.0]), flux_err=np.zeros(3))

    folded = phase_fold(lc, period=4.0)

    assert list(folded.time) == sorted(folded.time)
    # flux should be reordered to match, not just time
    np.testing.assert_allclose(folded.flux, [2.0, 3.0, 1.0])


def test_folding_sets_time_unit_to_phase():
    lc = LightCurve(time=[0, 1, 2], flux=[1.0, 2.0, 3.0], flux_err=[0, 0, 0])
    folded = phase_fold(lc, period=1.0)
    assert folded.time_unit == "phase"


def test_original_lightcurve_is_unmodified():
    original_time = np.array([0.0, 1.0, 2.0])
    lc = LightCurve(time=original_time, flux=[1.0, 2.0, 3.0], flux_err=[0, 0, 0])

    phase_fold(lc, period=1.5)

    np.testing.assert_allclose(lc.time, [0.0, 1.0, 2.0])


def test_nonpositive_period_raises():
    lc = LightCurve(time=[0, 1, 2], flux=[1.0, 2.0, 3.0], flux_err=[0, 0, 0])
    with pytest.raises(ValueError, match="positive"):
        phase_fold(lc, period=0.0)


def test_default_t0_is_shift_invariant():
    # The whole point of defaulting t0 to lc.time.min(): the same
    # periodic signal, observed starting at an arbitrary later time,
    # should fold to the identical shape. period=1.0 with a +5.3 shift
    # (NOT a whole number of periods) is the case that actually tests
    # this -- a fixed t0 like the old default (0.0) fails it.
    time = np.array([10.0, 10.5, 11.0, 11.5])
    flux = np.array([1.0, 2.0, 3.0, 4.0])

    lc_a = LightCurve(time=time, flux=flux, flux_err=np.zeros(4))
    folded_a = phase_fold(lc_a, period=1.0)

    lc_b = LightCurve(time=time + 5.3, flux=flux, flux_err=np.zeros(4))
    folded_b = phase_fold(lc_b, period=1.0)

    np.testing.assert_allclose(folded_a.time, folded_b.time)
    np.testing.assert_allclose(folded_a.flux, folded_b.flux)


def test_explicit_t0_overrides_default():
    # Passing t0 explicitly should still work for cases that need
    # phase 0 aligned to a known physical event (e.g. a transit
    # center), not just "first observation."
    time = np.array([0.0, 0.25, 0.5, 0.75])
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    lc = LightCurve(time=time, flux=flux, flux_err=np.zeros(4))

    folded_default = phase_fold(lc, period=1.0)  # t0 defaults to 0.0 here
    folded_explicit = phase_fold(lc, period=1.0, t0=0.25)

    # With these evenly-spaced points, the *set* of phase values is the
    # same either way (0, 0.25, 0.5, 0.75) -- what changes is WHICH
    # original point lands at phase 0. With t0=0.0, that's the point
    # with flux=1.0 (the original t=0.0 point). With t0=0.25, the point
    # at t=0.25 (flux=2.0) becomes phase 0 instead.
    flux_at_phase_zero_default = folded_default.flux[folded_default.time == 0.0][0]
    flux_at_phase_zero_explicit = folded_explicit.flux[folded_explicit.time == 0.0][0]

    assert flux_at_phase_zero_default == pytest.approx(1.0)
    assert flux_at_phase_zero_explicit == pytest.approx(2.0)
