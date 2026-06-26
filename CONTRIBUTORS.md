# Contributors

This package was built collaboratively during the Code/Astro 2026
workshop. This file credits where ideas and implementations originated,
independent of which fork or branch a given line of code currently lives
on.

## Team

- **Sheetal** — original project proposal and scope. Wrote the original
  `periodogram.py` (Lomb-Scargle period search) and `phasefold.py`
  (phase-folding), plus the TESS data acquisition approach in
  `download_tess_data.py` via Lightkurve. Several design choices from
  the original `periodogram.py`/`phasefold.py` are kept directly in
  this package's versions of the same modules — notably the `t0`
  default in `phasefold.py` (folding on the light curve's own first
  observation rather than a fixed `0.0`, which makes the folded shape
  invariant to absolute time origin) and use of astropy's
  `autopower`/`samples_per_peak` in `periodogram.py` for adaptive
  frequency-grid resolution.
- **Pulkit** — wrote `autocorrelation.py`, including the
  scipy-based peak-detection approach (`scipy.signal.find_peaks` with
  prominence and minimum-spacing constraints). This package's version
  builds on that approach directly, exposed via `min_prominence` and
  `min_peak_spacing`.
- **Kartik** — wrote `lightcurve.py` (the `LightCurve` data model
  and basic statistics) in full, plus the test suite, CI setup, and
  packaging. Adapted Sheetal's and Pulkit's originals above into this
  package's shared `LightCurve`-based interface.
- **`loader.py` and `plotting.py`** were a joint effort across the team,
  with each person's version contributing to the final implementations
  here.

## Notes on adapted code

Where an idea or approach originated with a teammate but the
implementation here differs (different function signature, added
validation, different data structures), the relevant module's docstring
says so explicitly with a short note on what changed and why. The goal
is for credit to track *ideas*, not just exact lines of code, since the
team worked across separate forks rather than one shared branch.
