# This function calculates the autocorrelation of a given light curve.

# Loading required libraries.
import csv
import numpy as np
from scipy.signal import find_peaks


# Loading the data file.
def load_light_curve(filename):
    flux = []
    time = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)

        next(reader) # skip the header

        for row in reader:
            if len(row) == 0 or row[0].startswith('#'):
                continue
            flux.append(float(row[1]))
            time.append(float(row[0]))

    return np.array(flux), np.array(time)


def compute_acf(time, flux):
    """
    Computes the autocorrelation function (ACF) of a given light curve.
    
    Args:
        time (numpy.ndarray): An array of time values corresponding to the light curve.
        flux (numpy.ndarray): An array of flux values corresponding to the light curve.

    Returns:
        lags (numpy.ndarray): An array of lag values corresponding to the ACF.
        acf (numpy.ndarray): The computed autocorrelation function values.
    """
    # Processing the loaded flux data.
    flux_centered = flux - np.mean(flux)
    
    n = len(flux_centered)
    acf = np.correlate(flux_centered, flux_centered, mode='full')

    acf = acf[n-1:]
    acf /= acf[0]  # Normalize the autocorrelation function

    dt = np.median(np.diff(time))
    lags = np.arange(0, len(acf)) * dt

    return np.array(lags), np.array(acf), np.array(dt)

def find_acf_peaks(lags, acf, dt):
    """
    Identifies the peaks in the autocorrelation function (ACF).
    
    Args:
        lags (numpy.ndarray): An array of lag values corresponding to the ACF.
        acf (numpy.ndarray): The computed autocorrelation function values.
        dt (float): The median time difference between consecutive time points in the light curve.

    Returns:
        peaks (list): A list of tuples containing the lag and ACF value of each identified peak.
        properties (dict): A dictionary containing properties of the identified peaks, such as prominence and width.
        dt (float): The median time difference between consecutive time points in the light curve.
    """
    minimum_distance_days = 0.2
    distance_in_indices = int(minimum_distance_days / dt)

    distance_in_indices = max(1, distance_in_indices)
    
    peaks = []
    peaks, properties = find_peaks(acf, prominence=0.1, distance=distance_in_indices)  # Find peaks which are atleast 0.1 prominent and are spaced out by at least 10 indices.

    return peaks, properties, dt

def estimate_timescale(lags, peaks, properties, min_period_days=0):
    """
    Estimates the characteristic timescale of the light curve based on the identified peaks in the ACF.
    
    Args:
        lags (numpy.ndarray): An array of lag values corresponding to the ACF.
        peaks (list): A list of indices corresponding to the identified peaks in the ACF.
        properties (dict): A dictionary containing properties of the identified peaks, such as prominence and width.
        min_period_days (float): The minimum period in days to consider for the timescale estimate.

    Returns:
        timescale (float): The estimated characteristic timescale of the light curve.
    """ 
    if len(peaks) > 0:
        peak_lags = lags[peaks]
        
        valid_indices = np.where(peak_lags >= min_period_days)[0]
        
        if len(valid_indices) == 0:
            return None
        
        peaks = peaks[valid_indices]
        prominences = properties['prominences'][valid_indices]

        best_peak_index = np.argmax(prominences)  # Find the index of the peak with the highest prominence
        best_peak = peaks[best_peak_index]  # Get the index of the best peak

        timescale = lags[best_peak]  # The best peak corresponds to the characteristic timescale
    else:
        timescale = None  # No peaks found

    return timescale

def plot_acf(lags, acf, peaks):
    """
    Plots the autocorrelation function (ACF) along with the identified peaks.
    
    Args:
        lags (numpy.ndarray): An array of lag values corresponding to the ACF.
        acf (numpy.ndarray): The computed autocorrelation function values.
        peaks (list): A list of indices corresponding to the identified peaks in the ACF.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(lags, acf, label='Autocorrelation Function', color='blue')
    plt.plot(lags[peaks], acf[peaks], 'ro', label='Identified Peaks')
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    flux, time = load_light_curve('/Users/pulkit/Desktop/Work/code_astro/project/Astronomy-Time-Series-Analysis-Toolkit/lightcurve_dataset.csv') 
    
    lags, acf, dt = compute_acf(time, flux)
    
    peaks, properties, dt = find_acf_peaks(lags, acf, dt)
    
    timescale = estimate_timescale(lags, peaks, properties, min_period_days=1.0)
    
    if timescale:
        print(f"Estimated timescale (period): {timescale:.4f}")
    else:
        print("No significant peaks found.")
        
    plot_acf(lags, acf, peaks)