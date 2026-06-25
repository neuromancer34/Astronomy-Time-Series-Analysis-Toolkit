"""Tests for astrotime.lightcurve.LightCurve."""

import numpy as np
import pytest

from astrotime.lightcurve import LightCurve


def test_construction_succeeds_with_matching_shapes():
    lc = LightCurve(time=[0, 1, 2], flux=[1.0, 2.0, 3.0], flux_err=[0.1, 0.1, 0.1])
    assert len(lc) == 3
    assert isinstance(lc.time, np.ndarray)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        LightCurve(time=[0, 1, 2], flux=[1.0, 2.0], flux_err=[0.1, 0.1, 0.1])


def test_too_few_points_raise():
    with pytest.raises(ValueError, match="at least 2 points"):
        LightCurve(time=[0], flux=[1.0], flux_err=[0.1])


def test_nan_in_flux_raises():
    with pytest.raises(ValueError, match="NaN"):
        LightCurve(time=[0, 1, 2], flux=[1.0, np.nan, 3.0], flux_err=[0.1, 0.1, 0.1])


def test_is_frozen():
    lc = LightCurve(time=[0, 1, 2], flux=[1.0, 2.0, 3.0], flux_err=[0.1, 0.1, 0.1])
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        lc.flux = np.array([9.0, 9.0, 9.0])


def test_mean_and_std_flux():
    lc = LightCurve(time=[0, 1, 2, 3], flux=[1.0, 2.0, 3.0, 4.0], flux_err=[0, 0, 0, 0])
    assert lc.mean_flux() == pytest.approx(2.5)
    assert lc.std_flux() == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))


def test_median_flux():
    lc = LightCurve(time=[0, 1, 2, 3], flux=[1.0, 2.0, 3.0, 100.0], flux_err=[0, 0, 0, 0])
    # median should be robust to the outlier (100.0) in a way mean isn't
    assert lc.median_flux() == pytest.approx(2.5)
    assert lc.mean_flux() != pytest.approx(2.5)  # confirms mean *is* pulled by the outlier


def test_rms_flux_matches_formula():
    flux = [1.0, 2.0, 3.0, 4.0]
    lc = LightCurve(time=[0, 1, 2, 3], flux=flux, flux_err=[0, 0, 0, 0])
    expected = np.sqrt(np.mean((np.array(flux) - np.mean(flux)) ** 2))
    assert lc.rms_flux() == pytest.approx(expected)


def test_rms_flux_currently_equals_std_flux():
    # Documents the relationship described in rms_flux's docstring:
    # they coincide today (both population-style, ddof=0) but are
    # computed independently and are allowed to diverge later.
    lc = LightCurve(time=[0, 1, 2, 3], flux=[2.0, 4.0, 4.0, 6.0], flux_err=[0, 0, 0, 0])
    assert lc.rms_flux() == pytest.approx(lc.std_flux())


def test_amplitude_is_peak_to_peak():
    lc = LightCurve(time=[0, 1, 2], flux=[5.0, 1.0, 9.0], flux_err=[0, 0, 0])
    assert lc.amplitude() == pytest.approx(8.0)  # 9 - 1


def test_duration_is_time_span():
    lc = LightCurve(time=[2.0, 5.0, 10.0], flux=[1.0, 2.0, 3.0], flux_err=[0, 0, 0])
    assert lc.duration() == pytest.approx(8.0)  # 10 - 2


def test_summary_contains_all_stats():
    lc = LightCurve(time=[0, 1, 2], flux=[1.0, 2.0, 3.0], flux_err=[0, 0, 0])
    summary = lc.summary()
    assert set(summary.keys()) == {
        "n_points",
        "mean_flux",
        "median_flux",
        "std_flux",
        "rms_flux",
        "amplitude",
        "duration",
    }
