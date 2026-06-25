"""
Core data structure for the astrotime package.

Every other module (loader, statistics, periodogram, phasefold,
autocorrelation, plotting) takes a LightCurve as its main input.
Defining it once here means we don't pass around three loose arrays
(time, flux, flux_err) through every function signature in the package.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LightCurve:
    """
    Represents a single astronomical light curve.

    Parameters
    ----------
    time : np.ndarray
        Observation times, typically in days (e.g. MJD or BJD).
        Must be the same length as flux.
    flux : np.ndarray
        Brightness measurement at each time. Often normalized so the
        star's median brightness is ~1.0.
    flux_err : np.ndarray, optional
        Measurement uncertainty on each flux value. If not provided,
        defaults to an array of zeros (meaning "uncertainty unknown",
        not "no uncertainty" -- some methods will treat zeros specially).

    Notes
    -----
    This class does not sort or clean the data on its own. Use
    loader.load_lightcurve() to get a LightCurve from a file; that
    function is responsible for sorting by time and dropping NaNs.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None

    def __post_init__(self):
        # Convert plain lists to numpy arrays so every downstream
        # function can rely on numpy behavior (vectorized math, etc.)
        self.time = np.asarray(self.time, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)

        if self.flux_err is None:
            self.flux_err = np.zeros_like(self.flux)
        else:
            self.flux_err = np.asarray(self.flux_err, dtype=float)

        # Fail loudly and early if the arrays don't line up, rather
        # than letting a confusing error surface later inside, say,
        # the periodogram code.
        if not (len(self.time) == len(self.flux) == len(self.flux_err)):
            raise ValueError(
                f"time, flux, and flux_err must be the same length. "  # pyright: ignore[reportImplicitStringConcatenation]
                f"Got {len(self.time)}, {len(self.flux)}, {len(self.flux_err)}."
            )

    def __len__(self):
        return len(self.time)

    def __repr__(self):
        return (
            f"LightCurve(n_points={len(self)}, "
            f"time_range=({self.time.min():.3f}, {self.time.max():.3f}))"
        )
