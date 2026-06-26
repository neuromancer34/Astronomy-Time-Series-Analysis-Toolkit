"""Estimate periods in unevenly-sampled time series via Lomb-Scargle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import LombScargle

from astrotime.lightcurve import LightCurve


@dataclass(frozen=True)
class PeriodogramResult:
    """The output of a Lomb-Scargle period search.

    Attributes
    ----------
    frequency : np.ndarray
        Frequencies the periodogram was evaluated at.
    power : np.ndarray
        Lomb-Scargle power at each frequency. Higher = more likely to
        be the true period of a periodic signal.
    best_period : float
        The period (1 / frequency) at maximum power -- the toolkit's
        single best guess at the dominant periodicity.
    best_power : float
        The power value at best_period, useful as a rough significance
        check: a low peak power means "no convincing periodicity found,"
        not "here is a confident answer."
    """

    frequency: np.ndarray
    power: np.ndarray
    best_period: float
    best_power: float


def compute_periodogram(
    lc: LightCurve,
    min_period: float | None = None,
    max_period: float | None = None,
    samples_per_peak: int = 10,
) -> PeriodogramResult:
    """Run a Lomb-Scargle periodogram on a LightCurve.

    Parameters
    ----------
    lc : LightCurve
    min_period, max_period : float, optional
        Bounds on the periods to search, in the same units as
        lc.time. If not given, defaults are chosen from the data
        itself: min_period defaults to roughly twice the median time
        spacing (you can't reliably detect a period shorter than your
        sampling cadence -- this is the Nyquist-like limit for
        irregular sampling), and max_period defaults to half the
        total time baseline (you need to see at least two cycles to
        call something periodic with any confidence).
    samples_per_peak : int
        Frequency-grid points per periodogram peak, passed to
        astropy's LombScargle.autopower. This adapts grid density to
        peak width (narrower at short periods), which matters more
        than it sounds — see the README's design notes for why a
        plain evenly-spaced grid can miss a true short period.

    Returns
    -------
    PeriodogramResult
    """
    if min_period is None:
        median_dt = float(np.median(np.diff(np.sort(lc.time))))
        min_period = 2 * median_dt
    if max_period is None:
        max_period = lc.duration() / 2

    if min_period >= max_period:
        raise ValueError(
            f"min_period ({min_period}) must be less than max_period "
            f"({max_period}). This usually means the light curve's time "
            "baseline is too short for the default bounds -- pass "
            "explicit min_period/max_period."
        )

    # astropy expects flux_err of zero to be treated as "no weighting" --
    # passing actual zeros would cause a division-by-zero inside its
    # weighted fit, so we only pass dy when we have real uncertainties.
    has_errors = np.any(lc.flux_err > 0)
    dy = lc.flux_err if has_errors else None

    ls = LombScargle(lc.time, lc.flux, dy)
    frequency, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=samples_per_peak,
    )

    periods = 1.0 / frequency

    best_idx = int(np.argmax(power))
    best_period = float(periods[best_idx])
    best_power = float(power[best_idx])

    return PeriodogramResult(
        frequency=frequency,
        power=power,
        best_period=best_period,
        best_power=best_power,
    )
