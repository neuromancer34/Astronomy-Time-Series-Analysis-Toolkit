# This is a function to test if the autocorrelation function (ACF) is working correctly.

import numpy as np
from astrotime import compute_acf, find_acf_peaks, estimate_timescale

def test_known_sine_wave():
    """
    Test the autocorrelation function (ACF) on a known sine wave with a period of exactly 5.0 days. 
    The test checks if the estimated timescale from the ACF matches the known period of the sine wave.
    """
    time = np.linspace(0, 100, 1000)
    
    flux = np.sin(2 * np.pi * time / 5.0)
    
    lags, acf, dt = compute_acf(time, flux)
    peaks, properties, dt = find_acf_peaks(lags, acf, dt)
    timescale = estimate_timescale(lags, peaks, properties)
    
    assert np.isclose(timescale, 5.0, atol=0.1)