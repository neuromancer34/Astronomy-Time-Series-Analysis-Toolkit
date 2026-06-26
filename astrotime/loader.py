"""Load light curves from disk into LightCurve objects.

A joint effort across the team (see CONTRIBUTORS.md). v1 supports CSV
only, since that's what most beginner-friendly astronomical datasets
(including TESS exports via lightkurve) end up as once you've done
any preprocessing. Add more formats (FITS, etc.) here later -- this
module is the single place that should know how to turn "a file on
disk" into a LightCurve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from astrotime.lightcurve import LightCurve


def load_csv(
    path: str | Path,
    time_col: str = "time",
    flux_col: str = "flux",
    flux_err_col: str | None = "flux_err",
) -> LightCurve:
    """Load a light curve from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file. Must have a header row.
    time_col : str
        Name of the column containing timestamps.
    flux_col : str
        Name of the column containing flux measurements.
    flux_err_col : str or None
        Name of the column containing flux uncertainties. If None, or
        if the column isn't present in the file, flux_err is filled
        with zeros rather than left missing -- see LightCurve's
        docstring for why every LightCurve always has a flux_err
        array, even when there's no real uncertainty data.

    Returns
    -------
    LightCurve

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    KeyError
        If `time_col` or `flux_col` is not a column in the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    df = pd.read_csv(path)

    missing = [c for c in (time_col, flux_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    time = df[time_col].to_numpy(dtype=float)
    flux = df[flux_col].to_numpy(dtype=float)

    if flux_err_col is not None and flux_err_col in df.columns:
        flux_err = df[flux_err_col].to_numpy(dtype=float)
    else:
        flux_err = np.zeros_like(flux)

    # Drop any rows where time or flux is NaN, rather than letting
    # LightCurve raise on construction -- a few dropped rows from a
    # real dataset is normal; failing the whole load isn't useful.
    valid = ~(np.isnan(time) | np.isnan(flux))
    if not valid.all():
        n_dropped = (~valid).sum()
        print(f"load_csv: dropping {n_dropped} row(s) with NaN time/flux")

    return LightCurve(time=time[valid], flux=flux[valid], flux_err=flux_err[valid])
