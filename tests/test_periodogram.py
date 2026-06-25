"""Tests for astrotime.periodogram.

The core test strategy: generate a sine wave with a *known* period,
run compute_periodogram on it, and assert the recovered best_period
is close to the true one. This is only possible because we control
the input -- it's the main argument for synthetic data over a
real downloaded light curve in the test suite.
"""

import numpy as np
import pytest

from astrotime.lightcurve import LightCurve
from astrotime.periodogram import compute_periodogram


def _make_sinusoidal_lightcurve(true_period, n_points=300, noise_level=0.05, seed=42):
    rng = np.random.default_rng(seed)
    # Irregular sampling on purpose -- this is exactly the case
    # Lomb-Scargle exists for, and it's a stronger test than evenly
    # spaced points would be.
    time = np.sort(rng.uniform(0, 20 * true_period, n_points))
    flux = np.sin(2 * np.pi * time / true_period) + rng.normal(0, noise_level, n_points)
    flux_err = np.full(n_points, noise_level)
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


def test_recovers_known_period():
    true_period = 3.3
    lc = _make_sinusoidal_lightcurve(true_period)

    result = compute_periodogram(lc)

    assert result.best_period == pytest.approx(true_period, rel=0.05)


def test_recovers_known_period_without_flux_err():
    # flux_err of all zeros should not crash the weighted fit --
    # compute_periodogram should fall back to an unweighted one.
    true_period = 2.0
    lc = _make_sinusoidal_lightcurve(true_period)
    lc_no_err = LightCurve(time=lc.time, flux=lc.flux, flux_err=np.zeros(len(lc)))

    result = compute_periodogram(lc_no_err)

    assert result.best_period == pytest.approx(true_period, rel=0.05)


def test_result_shapes_match():
    lc = _make_sinusoidal_lightcurve(1.5)
    result = compute_periodogram(lc, n_periods=500)

    assert result.frequency.shape == (500,)
    assert result.power.shape == (500,)


def test_invalid_period_bounds_raise():
    lc = _make_sinusoidal_lightcurve(1.5)
    with pytest.raises(ValueError, match="min_period"):
        compute_periodogram(lc, min_period=5.0, max_period=1.0)


def test_best_power_is_max_of_power_array():
    lc = _make_sinusoidal_lightcurve(2.5)
    result = compute_periodogram(lc)
    assert result.best_power == pytest.approx(np.max(result.power))
