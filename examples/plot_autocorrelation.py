from astrotime.loader import load_lightcurve
from astrotime.plotting import plot_lightcurve
from astrotime.autocorrelation import plot_acf,compute_acf, find_acf_peaks, estimate_timescale

lc = load_lightcurve("lightcurve_dataset.csv")
time = lc["time"].to_numpy()
flux = lc["flux"].to_numpy()
lags, acf, dt = compute_acf(time, flux)
peaks, properties, dt = find_acf_peaks(lags, acf, dt)
timescale = estimate_timescale(lags, peaks, properties)
plot_acf(lags, acf, peaks)