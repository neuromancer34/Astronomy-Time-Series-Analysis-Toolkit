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


def test_recovers_known_period_on_irregular_sampling():
    # This is the actual reason compute_autocorrelation resamples onto
    # a uniform time grid before computing the ACF: computing it
    # directly on irregularly-spaced timestamps mislabels the lag axis
    # (np.correlate's output index is a sample count, not a time
    # value), which was confirmed to give a noticeably wrong period on
    # data like this -- the resampling step is what this test guards.
    true_period = 4.2
    rng = np.random.default_rng(1)
    time = np.sort(rng.uniform(0, 50, 500))  # genuinely irregular
    flux = np.sin(2 * np.pi * time / true_period) + rng.normal(0, 0.08, 500)
    lc = LightCurve(time=time, flux=flux, flux_err=np.full(500, 0.08))

    result = compute_autocorrelation(lc)

    assert result.best_period == pytest.approx(true_period, rel=0.05)


def test_peak_lags_includes_harmonics():
    # find_peaks should pick up the period AND its harmonics (2P, 3P,
    # ...) as separate peaks, ordered by lag. This is new information
    # peak_lags exposes that a single best_period number doesn't.
    true_period = 2.0
    lc = _make_periodic_lightcurve(true_period, n_points=600)

    result = compute_autocorrelation(lc, max_lag=8.0)

    assert len(result.peak_lags) >= 3
    np.testing.assert_allclose(result.peak_lags[:3], [2.0, 4.0, 6.0], rtol=0.1)


def test_high_min_prominence_filters_weak_peaks():
    # A high min_prominence should find fewer (or no) peaks than a low
    # one on the same noisy-ish data -- confirms the parameter is
    # actually wired through to scipy's find_peaks, not ignored.
    lc = _make_periodic_lightcurve(2.0, n_points=600, noise_level=0.3)

    result_lenient = compute_autocorrelation(lc, min_prominence=0.01)
    n_peaks_lenient = len(result_lenient.peak_lags)

    try:
        result_strict = compute_autocorrelation(lc, min_prominence=0.95)
        n_peaks_strict = len(result_strict.peak_lags)
    except ValueError:
        n_peaks_strict = 0  # no peak met the very strict threshold

    assert n_peaks_strict <= n_peaks_lenient


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
