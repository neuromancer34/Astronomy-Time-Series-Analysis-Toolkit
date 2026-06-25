from astrotime.loader import load_lightcurve

lc = load_lightcurve("lightcurve_dataset.csv")
print(lc.head())