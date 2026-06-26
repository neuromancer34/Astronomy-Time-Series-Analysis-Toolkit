"""astrotime: a small toolkit for loading and analyzing astronomical light curves.

Public API
----------
The names below are the package's supported surface -- everything a
user should need is importable directly from `astrotime`, e.g.:

    from astrotime import load_csv, compute_periodogram, phase_fold

Internals (helper functions, private constants) may change between
versions without notice. These names are the ones we're committing to.
"""

from astrotime.autocorrelation import AutocorrelationResult, compute_autocorrelation
from astrotime.lightcurve import LightCurve
from astrotime.loader import load_csv
from astrotime.periodogram import PeriodogramResult, compute_periodogram
from astrotime.phasefold import phase_fold
from astrotime.plotting import (
    plot_autocorrelation,
    plot_lightcurve,
    plot_periodogram,
    plot_phase_curve,
)

__all__ = [
    "LightCurve",
    "load_csv",
    "compute_periodogram",
    "PeriodogramResult",
    "phase_fold",
    "compute_autocorrelation",
    "AutocorrelationResult",
    "plot_lightcurve",
    "plot_periodogram",
    "plot_autocorrelation",
    "plot_phase_curve",
]

__version__ = "0.1.0"
