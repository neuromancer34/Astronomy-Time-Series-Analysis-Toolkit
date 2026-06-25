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
    period = 4.0
    time = np.array([0.5, 0.5 + period, 0.5 + 2 * period])
    flux = np.array([10.0, 20.0, 30.0])
    lc = LightCurve(time=time, flux=flux, flux_err=np.zeros(3))

    folded = phase_fold(lc, period=period)

    np.testing.assert_allclose(folded.time, [0.125, 0.125, 0.125])


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
