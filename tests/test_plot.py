"""Tests for astrotime.plotting.

Plotting code is hard to test meaningfully (we're not asserting on
pixel output), so these tests check the more achievable, still
useful things: the functions run without error, return an Axes, and
the right Axes when one is passed in. We force the "Agg" backend so
these tests don't try to open a GUI window in CI or anywhere else
without a display.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from astrotime.autocorrelation import compute_autocorrelation
from astrotime.lightcurve import LightCurve
from astrotime.periodogram import compute_periodogram
from astrotime.phasefold import phase_fold
from astrotime.plotting import (
    plot_autocorrelation,
    plot_lightcurve,
    plot_periodogram,
    plot_phase_curve,
)


def _make_lightcurve():
    import numpy as np

    time = np.linspace(0, 20, 200)
    flux = np.sin(time) + 0.1 * np.random.default_rng(0).normal(size=200)
    return LightCurve(time=time, flux=flux, flux_err=np.full(200, 0.1))


def test_plot_lightcurve_returns_axes():
    lc = _make_lightcurve()
    ax = plot_lightcurve(lc)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_lightcurve_uses_provided_axes():
    lc = _make_lightcurve()
    _, ax = plt.subplots()
    returned = plot_lightcurve(lc, ax=ax)
    assert returned is ax
    plt.close(ax.figure)


def test_plot_periodogram_returns_axes():
    lc = _make_lightcurve()
    result = compute_periodogram(lc)
    ax = plot_periodogram(result)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_autocorrelation_returns_axes():
    lc = _make_lightcurve()
    result = compute_autocorrelation(lc)
    ax = plot_autocorrelation(result)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_phase_curve_returns_axes():
    lc = _make_lightcurve()
    folded = phase_fold(lc, period=2.5)
    ax = plot_phase_curve(folded)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_phase_curve_rejects_unfolded_lightcurve():
    # plot_phase_curve should refuse a raw LightCurve (time_unit "days",
    # not phase values in [0, 1)) rather than silently plotting nonsense.
    lc = _make_lightcurve()
    with pytest.raises(ValueError, match="phase-folded"):
        plot_phase_curve(lc)


def test_plot_phase_curve_repeat_cycle_doubles_points():
    lc = _make_lightcurve()
    folded = phase_fold(lc, period=2.5)
    n = len(folded)

    ax_repeated = plot_phase_curve(folded, repeat_cycle=True)
    # errorbar's first container holds the actual data line; its xdata
    # length tells us how many points were actually plotted.
    n_plotted_repeated = len(ax_repeated.containers[0][0].get_xdata())
    plt.close(ax_repeated.figure)

    ax_single = plot_phase_curve(folded, repeat_cycle=False)
    n_plotted_single = len(ax_single.containers[0][0].get_xdata())
    plt.close(ax_single.figure)

    assert n_plotted_single == n
    assert n_plotted_repeated == 2 * n
