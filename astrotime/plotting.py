"""Visualize light curves and the outputs of the analysis modules.

A joint effort across the team (see CONTRIBUTORS.md). Every function
here takes a matplotlib Axes as an optional argument and returns the
Axes it drew on, rather than calling plt.show() itself. This is a
deliberate convention, not an accident: it lets you compose multiple
plots into one figure (e.g. raw + folded side by side) and it keeps
this module usable in a script, a notebook, or a test, without
forcing a GUI window to pop up in any of them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from astrotime.autocorrelation import AutocorrelationResult
from astrotime.lightcurve import LightCurve
from astrotime.periodogram import PeriodogramResult


def plot_lightcurve(lc: LightCurve, ax: Axes | None = None, **kwargs) -> Axes:
    """Plot flux vs. time (or phase, if lc has been phase-folded).

    Parameters
    ----------
    lc : LightCurve
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    **kwargs
        Passed through to ax.errorbar (e.g. color="C1", alpha=0.5).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    kwargs.setdefault("fmt", "o")
    kwargs.setdefault("markersize", 3)
    kwargs.setdefault("alpha", 0.7)

    ax.errorbar(lc.time, lc.flux, yerr=lc.flux_err, **kwargs)
    ax.set_xlabel(lc.time_unit.capitalize())
    ax.set_ylabel(f"Flux ({lc.flux_unit})")
    return ax


def plot_phase_curve(lc: LightCurve, ax: Axes | None = None, repeat_cycle: bool = True, **kwargs) -> Axes:
    """Plot a phase-folded light curve.

    This is plot_lightcurve()'s sibling for folded data specifically.
    It does two things a generic flux-vs-time plot doesn't:

    1. It checks that `lc` actually looks phase-folded (time_unit is
       "phase" and values lie in [0, 1)) and raises a clear error if
       not -- catching the easy mistake of passing in a raw,
       un-folded LightCurve by accident.
    2. By default it plots phase 0-to-1 *twice* (0 to 2, with the
       second cycle being a repeat of the first). This is a standard
       convention in phase-curve plots: it makes the periodic shape
       visible right at the wraparound point (phase ~0 / ~1), which
       would otherwise look like two disconnected edges.

    Parameters
    ----------
    lc : LightCurve
        Must be phase-folded (e.g. the output of phasefold.phase_fold).
    ax : matplotlib.axes.Axes, optional
    repeat_cycle : bool
        If True (default), plot two repeated cycles (phase 0-2) for
        wraparound clarity. If False, plot a single cycle (phase 0-1).
    **kwargs
        Passed through to ax.errorbar.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if lc.time_unit != "phase" or lc.time.min() < 0 or lc.time.max() >= 1:
        raise ValueError(
            "plot_phase_curve expects a phase-folded LightCurve (time_unit "
            "'phase', values in [0, 1)). Did you mean to call phase_fold() "
            "first, or use plot_lightcurve() for a non-folded light curve?"
        )

    if ax is None:
        _, ax = plt.subplots()

    kwargs.setdefault("fmt", "o")
    kwargs.setdefault("markersize", 3)
    kwargs.setdefault("alpha", 0.7)
    kwargs.setdefault("color", "C0")

    if repeat_cycle:
        phase = np.concatenate([lc.time, lc.time + 1.0])
        flux = np.concatenate([lc.flux, lc.flux])
        flux_err = np.concatenate([lc.flux_err, lc.flux_err])
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    else:
        phase, flux, flux_err = lc.time, lc.flux, lc.flux_err

    ax.errorbar(phase, flux, yerr=flux_err, **kwargs)
    ax.set_xlabel("Phase")
    ax.set_ylabel(f"Flux ({lc.flux_unit})")
    return ax


def plot_periodogram(result: PeriodogramResult, ax: Axes | None = None, **kwargs) -> Axes:
    """Plot Lomb-Scargle power vs. period, with the best period marked.

    Parameters
    ----------
    result : PeriodogramResult
        Output of periodogram.compute_periodogram.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed through to ax.plot for the power curve.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    periods = 1.0 / result.frequency
    ax.plot(periods, result.power, **kwargs)
    ax.axvline(
        result.best_period,
        color="C1",
        linestyle="--",
        label=f"best period = {result.best_period:.4g}",
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("Lomb-Scargle power")
    ax.legend()
    return ax


def plot_autocorrelation(
    result: AutocorrelationResult, ax: Axes | None = None, **kwargs
) -> Axes:
    """Plot the autocorrelation function vs. lag, marking all detected peaks.

    Parameters
    ----------
    result : AutocorrelationResult
        Output of autocorrelation.compute_autocorrelation.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed through to ax.plot for the ACF curve.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Every detected peak in result.peak_lags is marked with a faint
    dotted line, not just best_period (highlighted in solid orange).
    Seeing peaks line up at P, 2P, 3P, ... is a good visual check that
    a period estimate is real and not a noise artifact -- a single
    isolated peak with no harmonics is more suspect than one with a
    clean harmonic series behind it.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(result.lag, result.acf, **kwargs)

    for peak_lag in result.peak_lags:
        if peak_lag == result.best_period:
            continue
        ax.axvline(peak_lag, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.axvline(
        result.best_period,
        color="C1",
        linestyle="--",
        label=f"best period = {result.best_period:.4g}",
    )
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    return ax
