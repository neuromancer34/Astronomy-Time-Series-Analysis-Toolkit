from astrotime.loader import load_lightcurve
from astrotime.plotting import plot_lightcurve
lc = load_lightcurve("lightcurve_dataset.csv")
plot_lightcurve(lc)