"""Estimate periods via autocorrelation -- a second, independent check.

Shifts the light curve against a copy of itself by every lag and
measures how well they line up; a periodic signal produces a peak in
the autocorrelation function (ACF) at lag = period. Unlike Lomb-Scargle,
this makes no assumption about signal shape (works on sharp eclipses,
not just sinusoids) but does assume even time sampling, so the light
curve is resampled onto a uniform grid first -- see README design notes
for why skipping that step gives wrong answers on real, gappy data.

Based on Pulkit's original autocorrelation.py; this version adds
resampling onto a uniform grid for correctness on irregularly-sampled
data. See CONTRIBUTORS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

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
    peak_lags : np.ndarray
        Lags of every detected peak in the ACF (excluding the trivial
        peak at lag=0), ordered by lag, not by prominence. Useful for
        spotting harmonics (e.g. peaks near P, 2P, 3P all showing up)
        or comparing multiple candidate periods rather than trusting
        a single number.
    best_period : float
        The lag of the most prominent detected peak -- the toolkit's
        best single guess at the period, from this method.
    """

    lag: np.ndarray
    acf: np.ndarray
    peak_lags: np.ndarray
    best_period: float


def compute_autocorrelation(
    lc: LightCurve,
    max_lag: float | None = None,
    n_samples: int | None = None,
    min_prominence: float = 0.1,
    min_peak_spacing: float | None = None,
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
    min_prominence : float
        Minimum prominence (in normalized ACF units, where acf[0]==1)
        a peak must have to be considered real rather than noise.
        Passed directly to scipy.signal.find_peaks. 0.1 is a
        reasonable starting point for moderately noisy data; raise it
        if noise is producing spurious peaks, lower it for very clean
        signals with genuinely weak periodicity.
    min_peak_spacing : float, optional
        Minimum lag (in the same units as lc.time) between two
        detected peaks. Prevents two points on the shoulder of the
        same real peak from being counted as separate peaks. Defaults
        to roughly the resampled time step, the smallest spacing that
        means anything given the grid resolution.

    Returns
    -------
    AutocorrelationResult

    Raises
    ------
    ValueError
        If no peak meeting min_prominence is found within max_lag --
        this usually means the signal isn't strongly periodic, or
        min_prominence is set too high for this data.
    """
    if max_lag is None:
        max_lag = lc.duration() / 2
    if n_samples is None:
        n_samples = len(lc)

    # Resample onto a uniform time grid -- see module docstring for why
    # this is necessary for irregularly-sampled data.
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

    if min_peak_spacing is None:
        min_peak_spacing = dt
    distance_in_samples = max(1, int(round(min_peak_spacing / dt)))

    # find_peaks operates on lag[1:]/acf[1:] -- excluding lag=0, which
    # is always the trivial global maximum and isn't a real "peak" in
    # the sense we're searching for.
    peak_idx, properties = find_peaks(
        acf[1:], prominence=min_prominence, distance=distance_in_samples
    )
    peak_idx = peak_idx + 1  # shift back since we sliced off index 0

    if len(peak_idx) == 0:
        raise ValueError(
            "No periodic peak found in the autocorrelation function within "
            "max_lag at the given min_prominence. The signal may not be "
            "strongly periodic, or try lowering min_prominence."
        )

    peak_lags = lag[peak_idx]
    # Most prominent peak, not necessarily the first one in lag order --
    # a harmonic at 2P can sometimes be more prominent than the true
    # period P itself, but for most well-behaved signals the strongest
    # peak is the real period.
    best_idx_within_peaks = int(np.argmax(properties["prominences"]))
    best_period = float(peak_lags[best_idx_within_peaks])

    return AutocorrelationResult(
        lag=lag, acf=acf, peak_lags=peak_lags, best_period=best_period
    )
