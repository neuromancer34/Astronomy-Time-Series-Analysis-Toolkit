"""Tests for astrotime.loader.

Uses pytest's tmp_path fixture to write a throwaway CSV per test,
rather than a checked-in fixture file -- keeps the test self-contained
and the input visible right next to the assertion that checks it.
"""

import numpy as np
import pytest

from astrotime.loader import load_csv


def _write_csv(path, time, flux, flux_err=None):
    lines = ["time,flux,flux_err"] if flux_err is not None else ["time,flux"]
    rows = zip(time, flux, flux_err) if flux_err is not None else zip(time, flux)
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines))


def test_load_csv_with_flux_err(tmp_path):
    csv_path = tmp_path / "lc.csv"
    _write_csv(csv_path, time=[0, 1, 2], flux=[1.0, 2.0, 3.0], flux_err=[0.1, 0.1, 0.1])

    lc = load_csv(csv_path)

    assert len(lc) == 3
    np.testing.assert_allclose(lc.time, [0, 1, 2])
    np.testing.assert_allclose(lc.flux, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(lc.flux_err, [0.1, 0.1, 0.1])


def test_load_csv_without_flux_err_defaults_to_zeros(tmp_path):
    csv_path = tmp_path / "lc.csv"
    _write_csv(csv_path, time=[0, 1, 2], flux=[1.0, 2.0, 3.0])

    lc = load_csv(csv_path)

    np.testing.assert_allclose(lc.flux_err, [0.0, 0.0, 0.0])


def test_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "does_not_exist.csv")


def test_load_csv_missing_column_raises(tmp_path):
    csv_path = tmp_path / "lc.csv"
    csv_path.write_text("timestamp,brightness\n0,1.0\n1,2.0\n")

    with pytest.raises(KeyError, match="time"):
        load_csv(csv_path)


def test_load_csv_drops_nan_rows(tmp_path):
    csv_path = tmp_path / "lc.csv"
    csv_path.write_text("time,flux\n0,1.0\n1,\n2,3.0\n")

    lc = load_csv(csv_path)

    assert len(lc) == 2
    np.testing.assert_allclose(lc.time, [0, 2])


def test_load_csv_with_custom_column_names(tmp_path):
    csv_path = tmp_path / "lc.csv"
    csv_path.write_text("t,brightness\n0,1.0\n1,2.0\n2,3.0\n")

    lc = load_csv(csv_path, time_col="t", flux_col="brightness", flux_err_col=None)

    np.testing.assert_allclose(lc.flux, [1.0, 2.0, 3.0])
