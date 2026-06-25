
from astrotime.loader import load_lightcurve
from astrotime.periodogram import compute_lombscargle, find_best_period, plot_periodogram

def test_periodogram():
    lc = load_lightcurve('lightcurve_dataset.csv')
    time =lc["time"].to_numpy()
    flux =lc["flux"].to_numpy()
    periods, power = compute_lombscargle(time,flux,min_period=1,max_period=20)
    best_period,best_power = find_best_period(periods,power)
    print("Recovered Best Period =",best_period)
    plot_periodogram(periods, power, best_period=best_period)
    assert len(periods) > 0
    assert len(power) > 0
    assert best_period > 0