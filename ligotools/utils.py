# ligotools/utils.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.io import wavfile as _wavfile
from scipy.signal import get_window, welch

__all__ = ["whiten", "reqshift", "write_wavfile", "plot_psd"]

# ---------------------------------------------------------------------
# Whitening (matches your notebook: takes an interpolator callable)
# ---------------------------------------------------------------------
def whiten(strain: np.ndarray, interp_psd, dt: float) -> np.ndarray:
    """
    Whiten a real-valued time series using a provided one-sided PSD interpolator.

    Parameters
    ----------
    strain : np.ndarray
        Real-valued time series.
    interp_psd : callable
        A function f(freqs) -> PSD(freqs), e.g. scipy.interpolate.interp1d(...)
        built from (freqs, data_psd) as in the notebook.
    dt : float
        Sample spacing (seconds).

    Returns
    -------
    np.ndarray
        Whitened time series (same length as input).
    """
    x = np.asarray(strain, dtype=float)
    Nt = x.size
    freqs = np.fft.rfftfreq(Nt, dt)

    # FFT of signal
    hf = np.fft.rfft(x)

    # Notebook’s normalization factor (kept to match results)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2.0))

    # Avoid divide-by-zero
    psd_vals = np.asarray(interp_psd(freqs), float)
    eps = 1e-24
    white_hf = hf / (np.sqrt(psd_vals) + eps) * norm

    return np.fft.irfft(white_hf, n=Nt)

# ---------------------------------------------------------------------
# Frequency shift (matches your notebook)
# ---------------------------------------------------------------------
def reqshift(data: np.ndarray, fshift: float = 100.0, sample_rate: int = 4096) -> np.ndarray:
    """
    Frequency-shift a real, band-passed signal by fshift (Hz) using an FFT-bin roll.
    """
    x = np.asarray(data, dtype=float)
    n = x.size
    X = np.fft.rfft(x)
    T = n / float(sample_rate)
    df = 1.0 / T
    nbins = int(np.round(fshift / df))

    Y = np.roll(X.real, nbins) + 1j * np.roll(X.imag, nbins)
    if nbins > 0:
        Y[:nbins] = 0.0
    elif nbins < 0:
        Y[nbins:] = 0.0

    return np.fft.irfft(Y, n=n)

# ---------------------------------------------------------------------
# WAV writing (used in your audio cells)
# ---------------------------------------------------------------------
def write_wavfile(path: Union[str, Path], fs: int, data: np.ndarray,
                  *, normalize: bool = True, dtype: str = "int16") -> Path:
    """
    Write a mono WAV file (normalizes by peak if requested).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(data).squeeze()
    if normalize and y.size:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak

    if dtype == "int16":
        y_out = (np.clip(y, -1.0, 1.0) * 32767 * 0.9).astype(np.int16)
    elif dtype == "float32":
        y_out = y.astype(np.float32)
    else:
        raise ValueError("dtype must be 'int16' or 'float32'")

    _wavfile.write(str(p), int(fs), y_out)
    return p

# ---------------------------------------------------------------------
# PSD / ASD plotting (encapsulates the “overlap + window” cell)
# ---------------------------------------------------------------------
def plot_psd(strain: np.ndarray, fs: float, *,
             label: Optional[str] = None,
             nperseg: Optional[int] = None,
             noverlap: Optional[int] = None,
             window: str = "blackman",
             fmin: float = 0.0,
             fmax: Optional[float] = None,
             asd: bool = True,
             ax=None):
    """
    Compute and plot PSD (or ASD) using Welch's method.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(strain, dtype=float)
    if nperseg is None:
        nperseg = int(4 * fs)
    if noverlap is None:
        noverlap = nperseg // 2

    win = get_window(window, nperseg, fftbins=True)
    freqs, psd = welch(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    spec = np.sqrt(psd) if asd else psd

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.loglog(freqs, spec, lw=1.2, label=label)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("ASD [1/√Hz]" if asd else "PSD [1/Hz]")
    if fmin is not None and fmin > 0:
        ax.set_xlim(left=fmin)
    if fmax is not None:
        ax.set_xlim(right=fmax)
    if label:
        ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    return freqs, spec, ax
