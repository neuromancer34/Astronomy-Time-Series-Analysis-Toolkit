"""Estimate periods via autocorrelation -- a second, independent check.

The idea: shift the light curve against a copy of itself by every
possible time lag, and measure how well they line up at each shift.
If the signal repeats with period P, the shifted copy will line up
well again at lag = P (and 2P, 3P, ...), producing a peak in the
autocorrelation function (ACF) at that lag.

Why have this alongside the Lomb-Scargle periodogram?
-------------------------------------------------------
They make different assumptions and fail in different ways.
Lomb-Scargle assumes a roughly sinusoidal signal and handles
irregular sampling natively. Autocorrelation makes no assumption
about the shape of the periodic signal (it works fine on sharp
eclipses or spiky flares, where a sinusoid-based fit can struggle)
but it does assume roughly even time sampling, so this module
resamples the light curve onto a uniform time grid first.

Getting the same answer from both methods is a good sign your period
estimate is real and not an artifact of one method's assumptions.
Getting different answers is a useful red flag worth investigating,
not a bug to silently average away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astrotime.lightcurve import LightCurve


@dataclass(frozen=True)
class AutocorrelationResult:
    """The output of an autocorrelation-based period search.

    Attributes
    ----------
    lag : np.ndarray
        Time lags the ACF was evaluated at, in the same units as the
        original light curve's time.
    acf : np.ndarray
        Autocorrelation value at each lag, normalized so acf[0] == 1.
    best_period : float
        The lag of the first real periodic peak in the ACF (the first
        local maximum reached after the ACF has dipped down from its
        trivial peak at lag=0) -- the toolkit's best guess at the
        period, from this method.
    """

    lag: np.ndarray
    acf: np.ndarray
    best_period: float


def compute_autocorrelation(
    lc: LightCurve,
    max_lag: float | None = None,
    n_samples: int | None = None,
) -> AutocorrelationResult:
    """Compute the autocorrelation function of a LightCurve.

    Parameters
    ----------
    lc : LightCurve
    max_lag : float, optional
        Largest lag to consider, in the same units as lc.time.
        Defaults to half the light curve's duration, for the same
        reason compute_periodogram defaults max_period the same way:
        you need to see at least two cycles to trust a period that long.
    n_samples : int, optional
        Number of points to resample the light curve onto a uniform
        time grid before computing the ACF (see module docstring for
        why this resampling is necessary). Defaults to the original
        number of points in lc.

    Returns
    -------
    AutocorrelationResult
    """
    if max_lag is None:
        max_lag = lc.duration() / 2
    if n_samples is None:
        n_samples = len(lc)

    # Resample onto an evenly-spaced time grid via linear interpolation.
    # This is the cost of using autocorrelation on irregular data --
    # interpolation is a real assumption, not a free lunch, which is
    # part of why this module exists alongside (not instead of) the
    # Lomb-Scargle periodogram.
    uniform_time = np.linspace(lc.time.min(), lc.time.max(), n_samples)
    uniform_flux = np.interp(uniform_time, lc.time, lc.flux)

    # Mean-subtract so the ACF measures shape similarity, not just
    # both copies sharing a nonzero average brightness.
    flux_centered = uniform_flux - uniform_flux.mean()

    # Full autocorrelation via convolution with a reversed copy of itself.
    # mode="full" gives lags from -(n-1) to +(n-1); we only need the
    # non-negative half since the ACF is symmetric.
    acf_full = np.correlate(flux_centered, flux_centered, mode="full")
    mid = len(acf_full) // 2
    acf = acf_full[mid:]
    acf = acf / acf[0]  # normalize so acf[0] == 1

    dt = uniform_time[1] - uniform_time[0]
    lag = np.arange(len(acf)) * dt

    in_range = lag <= max_lag
    lag = lag[in_range]
    acf = acf[in_range]

    # Find the first real peak after lag=0, by locating where the ACF
    # stops decreasing and starts increasing again (a local minimum)
    # and then taking the next local maximum after that point. This is
    # more robust than a fixed min_lag cutoff: a smooth signal's ACF
    # decreases monotonically away from lag=0 before turning back up
    # toward its true periodic peak, and that turning point is what
    # actually separates "still lag=0's shoulder" from "the real signal" --
    # not any particular distance in lag, which depends on the sampling
    # density and would need re-tuning for every dataset.
    diffs = np.diff(acf)
    # Index of the first point where the ACF goes from decreasing to
    # increasing, i.e. diffs flips from negative to non-negative.
    turning_points = np.where((diffs[:-1] < 0) & (diffs[1:] >= 0))[0]

    if len(turning_points) == 0:
        raise ValueError(
            "Could not find a periodic peak in the autocorrelation function "
            "within max_lag; the signal may not be periodic, or max_lag may "
            "be too small to reach the first full cycle."
        )

    # +1 to land on the local minimum itself (turning_points indexes diffs,
    # which is offset by one from acf); the peak we want is the max of acf
    # from that minimum onward, up to the next local minimum (or max_lag).
    search_start = turning_points[0] + 1
    search_lag = lag[search_start:]
    search_acf = acf[search_start:]

    best_period = float(search_lag[int(np.argmax(search_acf))])

    return AutocorrelationResult(lag=lag, acf=acf, best_period=best_period)
