# Astronomy Time-Series Analysis Toolkit
**Package name:** `astrotime`

A small Python toolkit for loading, visualizing, and analyzing astronomical
light curves. Built as a Code/Astro workshop project.

## Status

This is a first version. It covers the core workflow end to end:

| Capability | Module |
|---|---|
| Load a light curve from CSV | `loader.py` |
| Plot raw light curves | `plotting.py` |
| Store time/flux/flux_err + basic statistics (mean, median, std, RMS, amplitude) | `statistics.py` |
| Estimate the period via Lomb-Scargle + plotting of the Lomb-Scargle Periodogram | `periodogram.py` |
| Estimate the period via autocorrelation + plotting of autocorrelation function peaks | `autocorrelation.py` |
| Phase-fold on a period + plotting of Phase-fold | `phasefold.py` |


Not yet implemented: FITS file support, multi-band light curves, transit
fitting.


## Repository Structure

```text
Astronomy-Time-Series-Analysis-Toolkit/
├── astrotime/
│   ├── loader.py
│   ├── plotting.py
│   ├── statistics.py
│   ├── periodogram.py
│   ├── autocorrelation.py
│   └── phasefold.py
│
├── tests/
│   ├── test_loader.py
│   ├── test_plot.py
│   ├── test_statistics.py
│   ├── test_periodogram.py
│   ├── test_autocorrelation.py
│   └── test_phasefold.py
│
├── examples/
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```


## Installation

```bash
git clone https://github.com/Sheetal7040/Astronomy-Time-Series-Analysis-Toolkit.git
cd Astronomy-Time-Series-Analysis-Toolkit
pip install -e .
```

## Quickstart

The toolkit provides a modular workflow for astronomical time-series analysis:

1. Load a light-curve dataset from a CSV file.
2. Visualize the raw light curve.
3. Estimate the period using Lomb-Scargle periodogram.
4. Validate the recovered period using phase folding.
5. Perform additional analysis using the statistics and Autocorrelation modules.

See `examples/quickstart.ipynb` for a full walkthrough with real TESS data,
and `examples/download_tess_data.py` for fetching that data.

## Design Notes

- The toolkit follows a modular architecture, with each analysis task implemented as an independent Python module.
- The main workflow consists of loading a light curve, visualization, period estimation using the Lomb–Scargle periodogram, and phase folding.
- The Statistics and Autocorrelation modules operate independently on the input light-curve dataset and provide complementary analyses.
- The project is packaged using `pyproject.toml`, making it easy to install, test, and extend.
- Unit tests are provided for each module to help verify correctness and maintain code quality.

  
## Development

```bash
pip install -e .
pip install -e ".[dev]" 
pytest
```

## Future Work

- Support additional input formats such as FITS files.
- Extend the toolkit to handle multi-band light curves.
- Add more time-series analysis techniques and period-search algorithms.
- Improve documentation with more examples and tutorials.
- Publish the package on PyPI for easy installation.
  
  
## Acknowledgements
This project was developed as part of the Code/Astro 2026 Workshop by :
-Kartik Gupta
-Pulkit Sheoran
-Sheetal
