import numpy as np
from astrotime.loader import load_lightcurve
from astrotime.plotting import plot_lightcurve
from astrotime.periodogram import compute_lombscargle, plot_periodogram, find_best_period
from astrotime.phasefold import phase_fold, plot_fold

lc = load_lightcurve("lightcurve_dataset.csv")
plot_lightcurve(lc)
time =lc["time"].to_numpy()
flux =lc["flux"].to_numpy()
flux = flux/np.nanmedian(flux)
periods, power = compute_lombscargle(time,flux,min_period=0.2,max_period=1.0)
best_period,best_power = find_best_period(periods,power)
print("Recovered Best Period =",best_period)
plot_periodogram(periods, power, best_period=best_period)
phase, folded_flux = phase_fold(time, flux, best_period, t0=None)
plot_fold(phase,folded_flux, show=True)