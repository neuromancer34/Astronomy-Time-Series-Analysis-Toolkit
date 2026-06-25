import numpy as np
import matplotlib.pyplot as plt    
from astropy.timeseries import LombScargle

def compute_lombscargle(time, flux, min_period=None, max_period=None, samples_per_peak=10):
    """
    Input Parameters :
    time : array-like observation times.
    flux : array-like flux values coresspomding to the observation times.
    min_period : float, Minimum period to search (optional).
    max_period : float, Maximum period to search (optional).
    samples_per_peak : int, Sampling density of the frequency grid (optional).

    Returns : 
    periods : np.ndarray Period grid.
    power : np.ndarray Lomb-Scargle power at each period.
    """
    time = np.asarray(time)
    flux = np.asarray(flux)
    # Removing the Nan values from the array 
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask] 
    flux = flux[mask]

    if len(time)< 3:
        raise ValueError("Not enough valid data pounts to compute periodogram")

    flux = flux - np.mean(flux)

    baseline = time.max()-time.min()
    if baseline <= 0:
        raise ValueError("Time array vmust span a non zero baseline")
    

    if min_period is None:
        min_period = 0.05
    if max_period is None:
        max_period = baseline / 2

    min_frequency = 1.0 / max_period
    max_frequency = 1.0/ min_period

    frequency, power = LombScargle(time,flux).autopower(minimum_frequency= min_frequency, 
    maximum_frequency = max_frequency, samples_per_peak = samples_per_peak)

    periods = 1.0/frequency
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]
    power = power[sort_idx]
    return periods, power


def find_best_period(periods, power):
    """
    Input Parameters:
    periods: Period grid
    power: LombScargle power values

    Returns:
    best_period : float, Period with maximum Power
    best_power : float, Maximum LombScargle power.
    """

    periods = np.asarray(periods)
    power = np.asarray(power)

    if len(periods) == 0 or len(power) == 0 :
        raise ValueError("Periods and Power arrays must not be empty")

    idx = np.argmax(power)
    return periods[idx], power[idx]


def plot_periodogram(periods, power, best_period=None):
    """
    Input Parameters:
    periods : array like Period grid.
    power : array like LombScargle power values
    best_period : Best period to highlighton the plot (optional)
    
    Returns :
    fig : Figure
    """
    plt.figure(figsize=(10,5))
    plt.plot(periods,power,lw=1.5,label="Lomb-Scargle Periodogram")

    if best_period is not None:
        plt.axvline(best_period,linestyle=":",color='r', label=f"Best Period = {best_period:.4f} d")
        

    plt.xlabel("Period(days)")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram")
    plt.legend()
    plt.grid()
    plt.show()


