"""Phase-fold a light curve on a known or estimated period.

Based on Sheetal's original phasefold.py. Phase-folding takes every
observation's timestamp and replaces it with where in the cycle (0 to
1) that observation falls, given a period. If the period is right,
scattered-looking data collapses into a clean repeating shape -- this
is the classic "does this period actually explain the data" sanity
check, and it's usually the next thing you'd do right after
compute_periodogram gives you a candidate period. See CONTRIBUTORS.md.
"""

from __future__ import annotations

import numpy as np

from astrotime.lightcurve import LightCurve


def phase_fold(lc: LightCurve, period: float, t0: float | None = None) -> LightCurve:
    """Fold a LightCurve's time axis onto phase [0, 1) for a given period.

    Parameters
    ----------
    lc : LightCurve
    period : float
        The period to fold on, in the same units as lc.time. Typically
        this is the best_period from a PeriodogramResult.
    t0 : float, optional
        Reference epoch; phase 0 corresponds to t0. Defaults to
        lc.time.min() rather than a fixed 0.0, which makes the folded
        shape invariant to the light curve's absolute start time (see
        README design notes). Pass explicitly to align phase 0 with a
        known physical event instead.

    Returns
    -------
    LightCurve
        A new LightCurve whose `time` field now holds phase values in
        [0, 1) instead of the original timestamps, sorted by phase.
        flux and flux_err are reordered to match. The original `lc` is
        not modified.

    Notes
    -----
    The returned LightCurve's time_unit is set to "phase" so that
    plotting.py (or you, inspecting it later) can tell at a glance
    that this isn't a normal time axis anymore.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    if t0 is None:
        t0 = float(lc.time.min())

    # (t - t0) / period gives "how many cycles since t0", as a float.
    # The fractional part (via np.mod) is the phase within the current
    # cycle, e.g. 2.3 cycles in means phase 0.3.
    phase = np.mod((lc.time - t0) / period, 1.0)

    order = np.argsort(phase)

    return LightCurve(
        time=phase[order],
        flux=lc.flux[order],
        flux_err=lc.flux_err[order],
        time_unit="phase",
        flux_unit=lc.flux_unit,
    )
