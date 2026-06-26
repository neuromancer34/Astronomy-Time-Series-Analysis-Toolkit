# astrotime

[![tests](https://github.com/neuromancer34/Astronomy-Time-Series-Analysis-Toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/neuromancer34/Astronomy-Time-Series-Analysis-Toolkit/actions/workflows/test.yml)
[![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small Python toolkit for loading, visualizing, and analyzing astronomical
light curves.
Built as a [Code/Astro 2026](https://semaphorep.github.io/codeastro/) workshop project.

| Capability | Module |
|---|---|
| Load a light curve from CSV | `loader.py` |
| Store time/flux/uncertainty + basic statistics (mean, median, std, RMS, amplitude) | `lightcurve.py` |
| Estimate period via Lomb-Scargle | `periodogram.py` |
| Estimate period via autocorrelation (independent cross-check) | `autocorrelation.py` |
| Phase-fold on a known period | `phasefold.py` |
| Visualize raw/folded light curves, periodograms, and ACFs | `plotting.py` |

**Not yet implemented:** FITS support, multi-band light curves.

## Installation

Requires **Python 3.10+**. This project uses [`uv`](https://docs.astral.sh/uv/), which manages the
Python environment and dependencies for you — no manual virtual environment needed.

**Install uv**, if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```
Restart your terminal afterward, then confirm with `uv --version`.

**Clone and install:**
```bash
git clone https://github.com/neuromancer34/Astronomy-Time-Series-Analysis-Toolkit.git
cd Astronomy-Time-Series-Analysis-Toolkit
uv sync
```
That's it. Run any command with `uv run <command>` (e.g. `uv run python`,
`uv run pytest`) and it executes inside the correct environment automatically.

<details>
<summary>Prefer plain pip?</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
pip install -e ".[dev]"
```
Then drop `uv run` from any command below.
</details>

## Quickstart

```python
from astrotime import load_csv, compute_periodogram, phase_fold, plot_phase_curve

lc = load_csv("my_lightcurve.csv")  # expects columns: time, flux, flux_err
print(lc.summary())

result = compute_periodogram(lc)
folded = phase_fold(lc, period=result.best_period)
plot_phase_curve(folded)
```

Run with `uv run python your_file.py`. For a full walkthrough with real TESS
data, open `examples/quickstart.ipynb` (`uv run jupyter notebook examples/quickstart.ipynb`);
to fetch that data first, see `examples/download_tess_data.py`.

## Project structure

```
astrotime/
├── .github/workflows/test.yml   # CI
├── astrotime/                   # the package
│   ├── __init__.py              #   public API
│   ├── lightcurve.py            #   LightCurve container + statistics
│   ├── loader.py                #   CSV loading
│   ├── periodogram.py           #   Lomb-Scargle period estimation
│   ├── phasefold.py             #   phase-folding
│   ├── autocorrelation.py       #   period estimation via ACF
│   └── plotting.py              #   visualization
├── tests/                       # one file per module above
├── examples/
│   ├── quickstart.ipynb
│   └── download_tess_data.py
├── pyproject.toml
├── CONTRIBUTORS.md
├── LICENSE
└── README.md
```

## Troubleshooting

- **`uv: command not found`** — restart your terminal after installing.
- **`load_csv` raises `KeyError`** — it expects columns named `time`,
  `flux`, `flux_err`. Pass alternatives: `load_csv(path, time_col="MJD")`.
- **Anything else** — open an issue with the exact command and full error.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

MIT — see [LICENSE](LICENSE).
