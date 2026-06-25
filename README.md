# astrotime

A Python toolkit for loading, analyzing, and phase-folding astronomical
light curves. Built as a team project for the
[Code/Astro](https://github.com/semaphoreP/codeastro) workshop.

## What it does (or will do)

Given a light curve (a star's brightness measured over time), `astrotime`
helps you:

- Load light curve data from common file formats
- Compute basic descriptive statistics (mean brightness, variability, etc.)
- Estimate periodic signals using Lomb-Scargle periodograms
- Cross-check periods using autocorrelation
- Phase-fold data onto a candidate period to visually confirm it
- Plot light curves, periodograms, and phase-folded data


## Installation

Clone the repo, then install in editable mode from the repo root (the same
folder as `pyproject.toml`):

```bash
git clone https://github.com/Sheetal7040/Astronomy-Time-Series-Analysis-Toolkit.git
cd Astronomy-Time-Series-Analysis-Toolkit
pip install -e ".[dev]"
```

The `[dev]` extra also installs `pytest`, needed to run the test suite.
We recommend doing this inside a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # on Linux/macOS
pip install -e ".[dev]"
```

Editable mode (`-e`) means changes to the source code take effect
immediately, without needing to reinstall.

## Quickstart

```python
from astrotime.loader import load_lightcurve_csv
from astrotime.statistics import mean_flux, std_flux, amplitude, time_baseline

# Load a light curve from a CSV with "time" and "flux" columns
lc = load_lightcurve_csv("data/lightcurve_dataset.csv")

print(lc)
print("Mean flux:", mean_flux(lc))
print("Flux std dev:", std_flux(lc))
print("Peak-to-peak amplitude:", amplitude(lc))
print("Time baseline (days):", time_baseline(lc))
```

If your CSV has a flux uncertainty column too:

```python
lc = load_lightcurve_csv(
    "data/lightcurve_dataset.csv",
    flux_err_col="flux_err",
)
```

## Running tests

```bash
pytest tests/
```

## Project structure

```
astrotime/
├── astrotime/          # package source
│   ├── lightcurve.py    # LightCurve data class
│   ├── loader.py         # file -> LightCurve
│   ├── statistics.py     # descriptive statistics
│   ├── periodogram.py    # Lomb-Scargle period estimation (planned)
│   ├── autocorrelation.py # autocorrelation period estimation (planned)
│   ├── phasefold.py      # phase-folding (planned)
│   └── plotting.py       # visualization (planned)
├── tests/               # automated tests (pytest)
├── examples/            # example/demo scripts, incl. TESS data download
├── pyproject.toml       # package metadata + dependencies
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).

## Contributors

Built by the astrotime team at Code/Astro.
