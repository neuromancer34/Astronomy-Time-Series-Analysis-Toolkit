import numpy as np
import matplotlib.pyplot as plt

def phase_fold(time, flux, period, t0=None):
    """
    Input Parameter :
    time : Observation times (array)
    flux : flux values corresponding to observational times (array)
    period : Period to fold the light curve on (float)
    t0 : Referemnce epoch

    Return:
    phase : phase values between 0 and 1
    """
    time = np.asarray(time)
    flux = np.asarray(flux)

    if len(time) != len(flux):
        raise ValueError("time and flux must have same length")

    if period <= 0 :
        raise ValueError("period must be positive")

    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    if t0 is None:
        t0 = time.min()

    phase = ((time - t0) % period / period)
    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    folded_flux = flux[sort_idx]

    return phase, folded_flux

def plot_fold(phase, folded_flux, show=True):
    plt.figure(figsize=(10,5))
    plt.scatter(phase, folded_flux, s=10)
    plt.xlabel("Phase")
    plt.ylabel("Flux")
    plt.title("Phase-folded Light Curve")
    plt.tight_layout()
    plt.grid()
    plt.show()
