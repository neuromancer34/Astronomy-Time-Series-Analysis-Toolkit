"""Core data container for astronomical light curves.

A light curve is just three aligned arrays: time, flux, and (usually)
flux uncertainty. Every other module in this package (periodogram,
phasefold, autocorrelation, plotting) takes a LightCurve as input and,
where relevant, returns a new LightCurve rather than mutating the one
you handed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LightCurve:
    """An immutable container for time-series brightness measurements.

    Parameters
    ----------
    time : np.ndarray
        Observation timestamps. Units are up to the caller (days, BJD,
        etc.) but should be consistent within one LightCurve.
    flux : np.ndarray
        Brightness measurement at each timestamp. Same length as time.
    flux_err : np.ndarray, optional
        Per-point uncertainty on flux. Same length as time. If you
        don't have real uncertainties, pass an array of zeros rather
        than None -- this keeps every downstream function able to
        assume flux_err exists and is the right shape.
    time_unit : str
        Free-text label for what `time` is measured in, e.g. "days".
        Not used in any calculation -- purely for your own sanity and
        for axis labels in plotting.py.
    flux_unit : str
        Free-text label for what `flux` is measured in, e.g.
        "relative" or "mag".

    Why frozen?
    -----------
    LightCurve instances are immutable. Operations like phase-folding
    don't modify self -- they return a brand new LightCurve. This
    means a variable holding a raw light curve can never accidentally
    become a folded one out from under you.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    time_unit: str = "days"
    flux_unit: str = "relative"

    def __post_init__(self) -> None:
        # np.asarray makes this forgiving of plain lists/tuples as input,
        # while still giving every downstream function real ndarrays.
        # object.__setattr__ is required because the dataclass is frozen.
        object.__setattr__(self, "time", np.asarray(self.time, dtype=float))
        object.__setattr__(self, "flux", np.asarray(self.flux, dtype=float))
        object.__setattr__(self, "flux_err", np.asarray(self.flux_err, dtype=float))

        if not (self.time.shape == self.flux.shape == self.flux_err.shape):
            raise ValueError(
                "time, flux, and flux_err must all be the same shape; got "
                f"{self.time.shape}, {self.flux.shape}, {self.flux_err.shape}"
            )

        if self.time.ndim != 1:
            raise ValueError(f"LightCurve arrays must be 1-D; got ndim={self.time.ndim}")

        if len(self.time) < 2:
            raise ValueError("LightCurve needs at least 2 points to be meaningful")

        if np.isnan(self.time).any() or np.isnan(self.flux).any():
            raise ValueError(
                "time and flux cannot contain NaNs. Drop or interpolate them "
                "before constructing a LightCurve -- silent NaNs will corrupt "
                "downstream calculations like the Lomb-Scargle fit."
            )

    def __len__(self) -> int:
        return len(self.time)

    # --- basic statistics -------------------------------------------------
    # These are deliberately simple. The goal of v1 is correct, readable
    # basics -- not, e.g., outlier-robust or sigma-clipped statistics.

    def mean_flux(self) -> float:
        """Arithmetic mean of the flux values."""
        return float(np.mean(self.flux))

    def median_flux(self) -> float:
        """Median of the flux values.

        Unlike the mean, the median is robust to outliers -- a single
        anomalously bright or faint point (a cosmic ray hit, a bad
        calibration frame) can shift the mean noticeably but barely
        moves the median. Useful as a sanity check: if mean_flux()
        and median_flux() disagree a lot, that's a sign your light
        curve has outliers worth a closer look before you trust other
        statistics computed from it.
        """
        return float(np.median(self.flux))

    def std_flux(self) -> float:
        """Standard deviation of the flux values (population, ddof=0)."""
        return float(np.std(self.flux))

    def rms_flux(self) -> float:
        """Root-mean-square (RMS) variability of the flux values.

        Defined as sqrt(mean((flux - mean_flux)^2)) -- this is
        mathematically the population standard deviation (ddof=0), so
        right now rms_flux() and std_flux() will always return the
        same number. They're kept as separate methods on purpose:
        "RMS variability" and "standard deviation" are different
        concepts that happen to coincide for this exact formula, and
        if std_flux() is ever changed (e.g. to ddof=1, the sample
        standard deviation, which is a very common edit) the two
        should NOT change together. RMS variability is specifically
        defined here as population-style (1/N), regardless of what
        std_flux() does.
        """
        return float(np.sqrt(np.mean((self.flux - self.mean_flux()) ** 2)))

    def amplitude(self) -> float:
        """Peak-to-peak flux range: max(flux) - min(flux).

        A quick-and-dirty proxy for "how much does this thing vary."
        Sensitive to single outliers -- fine for a first look, not a
        substitute for a proper variability statistic later.
        """
        return float(np.max(self.flux) - np.min(self.flux))

    def duration(self) -> float:
        """Total time baseline: max(time) - min(time).

        Matters for period-finding: you generally can't trust a
        Lomb-Scargle period estimate longer than roughly half the
        baseline, since you haven't observed even two full cycles.
        """
        return float(np.max(self.time) - np.min(self.time))

    def summary(self) -> dict[str, float]:
        """Convenience bundle of the stats above, e.g. for printing."""
        return {
            "n_points": len(self),
            "mean_flux": self.mean_flux(),
            "median_flux": self.median_flux(),
            "std_flux": self.std_flux(),
            "rms_flux": self.rms_flux(),
            "amplitude": self.amplitude(),
            "duration": self.duration(),
        }
