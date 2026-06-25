"""Tests for astrotime.autocorrelation.

Same strategy as test_periodogram.py: synthetic signal with a known
period, assert the recovered period is close to the truth.
"""

import numpy as np
import pytest

from astrotime.lightcurve import LightCurve
from astrotime.autocorrelation import compute_autocorrelation


def _make_periodic_lightcurve(true_period, n_points=400, noise_level=0.02, seed=7):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 15 * true_period, n_points)  # evenly sampled, by design
    flux = np.sin(2 * np.pi * time / true_period) + rng.normal(0, noise_level, n_points)
    return LightCurve(time=time, flux=flux, flux_err=np.full(n_points, noise_level))


def test_recovers_known_period():
    true_period = 2.7
    lc = _make_periodic_lightcurve(true_period)

    result = compute_autocorrelation(lc)

    assert result.best_period == pytest.approx(true_period, rel=0.1)


def test_recovers_known_period_dense_long_baseline():
    # Regression test: dense sampling (500 pts) over a long baseline
    # (50 units) with low noise. This combination previously fooled
    # the peak-finder into locking onto a near-zero-lag artifact
    # instead of the true period -- catching that required testing
    # with realistic data volumes, not just the lighter default in
    # _make_periodic_lightcurve.
    true_period = 4.2
    rng = np.random.default_rng(1)
    time = np.sort(rng.uniform(0, 50, 500))
    flux = np.sin(2 * np.pi * time / true_period) + rng.normal(0, 0.08, 500)
    lc = LightCurve(time=time, flux=flux, flux_err=np.full(500, 0.08))

    result = compute_autocorrelation(lc)

    assert result.best_period == pytest.approx(true_period, rel=0.1)


def test_acf_at_zero_lag_is_one():
    lc = _make_periodic_lightcurve(1.0)
    result = compute_autocorrelation(lc)
    assert result.acf[0] == pytest.approx(1.0)


def test_lag_and_acf_shapes_match():
    lc = _make_periodic_lightcurve(1.0)
    result = compute_autocorrelation(lc)
    assert result.lag.shape == result.acf.shape


def test_lag_respects_max_lag():
    lc = _make_periodic_lightcurve(1.0)
    result = compute_autocorrelation(lc, max_lag=3.0)
    assert result.lag.max() <= 3.0


def test_too_small_max_lag_raises():
    lc = _make_periodic_lightcurve(1.0)
    with pytest.raises(ValueError, match="max_lag"):
        compute_autocorrelation(lc, max_lag=1e-6)
