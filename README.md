# astrotime

[![tests](https://github.com/neuromancer34/Astronomy-Time-Series-Analysis-Toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/neuromancer34/Astronomy-Time-Series-Analysis-Toolkit/actions/workflows/test.yml)

A small Python toolkit for loading, visualizing, and analyzing astronomical
light curves. Built as a Code/Astro workshop project.

## Status

This is a first version. It covers the core workflow end to end:

| Capability | Module |
|---|---|
| Load a light curve from CSV | `loader.py` |
| Store time/flux/flux_err + basic statistics (mean, median, std, RMS, amplitude) | `lightcurve.py` |
| Estimate the period via Lomb-Scargle | `periodogram.py` |
| Estimate the period via autocorrelation | `autocorrelation.py` |
| Phase-fold on a period | `phasefold.py` |
| Visualize raw and phase-folded light curves, periodograms, and ACFs | `plotting.py` |

Not yet implemented: FITS file support, multi-band light curves, transit
fitting. These are reasonable next steps once the basics above are solid.

## Installation

```bash
git clone https://github.com/neuromancer34/Astronomy-Time-Series-Analysis-Toolkit.git
cd Astronomy-Time-Series-Analysis-Toolkit
uv sync
```

## Quickstart

```python
from astrotime import load_csv, compute_periodogram, phase_fold, plot_phase_curve

lc = load_csv("my_lightcurve.csv")  # expects columns: time, flux, flux_err
print(lc.summary())  # mean, median, std, RMS, amplitude, duration

result = compute_periodogram(lc)
print(f"Best period: {result.best_period}")

folded = phase_fold(lc, period=result.best_period)
plot_phase_curve(folded)
```

See `examples/quickstart.ipynb` for a full walkthrough with real TESS data,
and `examples/download_tess_data.py` for fetching that data.

## Design notes

- `LightCurve` is an immutable dataclass. Operations like `phase_fold` return
  a *new* `LightCurve` rather than modifying the one you pass in.
- Every light curve always has a `flux_err` array, even if it's all zeros —
  this keeps every function able to assume it exists, without special-casing
  "no error data" everywhere.
- The Lomb-Scargle periodogram (`periodogram.py`) and the autocorrelation
  function (`autocorrelation.py`) are two independent ways to estimate a
  period. They make different assumptions, so agreement between them is a
  good sign; disagreement is worth investigating, not averaging away.

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT — see `LICENSE`.
