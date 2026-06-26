# Contributors

This package was built collaboratively during the Code/Astro 2026
workshop. This file credits where ideas and implementations originated,
independent of which fork or branch a given line of code currently lives
on.

## Team

- **Sheetal** — original project proposal and scope. Wrote the original
  `periodogram.py` (Lomb-Scargle period search) and `phasefold.py`
  (phase-folding), plus the TESS data acquisition approach in
  `download_tess_data.py` via Lightkurve.
- **Pulkit** — wrote `autocorrelation.py`, including the
  scipy-based peak-detection approach (`scipy.signal.find_peaks` with
  prominence and minimum-spacing constraints).
- **Kartik** — wrote `lightcurve.py` (the `LightCurve` data model
  and basic statistics), plus the test suite. 
  Adapted Sheetal's and Pulkit's originals above into this
  package's shared `LightCurve`-based interface.
- **`loader.py` and `plotting.py`** were a joint effort across the team,
  with each person's version contributing to the final implementations
  here.

