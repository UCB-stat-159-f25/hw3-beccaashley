# ligotools/utils.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import numpy as np
from scipy.io import wavfile as _wavfile
from scipy.signal import get_window, welch
import matplotlib.pyplot as plt

__all__ = ["whiten", "reqshift", "write_wavfile", "plot_psd"]


def whiten(strain: np.ndarray, interp_psd, dt: float) -> np.ndarray:
    x = np.asarray(strain, dtype=float)
    Nt = x.size
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(x)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2.0))
    psd_vals = np.asarray(interp_psd(freqs), float)
    eps = 1e-24
    white_hf = hf / (np.sqrt(psd_vals) + eps) * norm
    return np.fft.irfft(white_hf, n=Nt)


def reqshift(data: np.ndarray, fshift: float = 100.0, sample_rate: int = 4096) -> np.ndarray:
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


def write_wavfile(path: Union[str, Path], fs: int, data: np.ndarray,
                  *, normalize: bool = True, dtype: str = "int16") -> Path:
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


def plot_psd(strain: np.ndarray, fs: float, *,
             label: Optional[str] = None,
             nperseg: Optional[int] = None,
             noverlap: Optional[int] = None,
             window: str = "blackman",
             fmin: float = 0.0,
             fmax: Optional[float] = None,
             asd: bool = True,
             ax=None):
    """Plot the PSD/ASD of a signal with minimal spectral leakage."""
    x = np.asarray(strain, dtype=float)
    if nperseg is None:
        nperseg = int(4 * fs)
    if noverlap is None:
        noverlap = nperseg // 2
    win = get_window(window, nperseg, fftbins=True)
    freqs, psd = welch(x, fs=fs, window=win, nperseg=nperseg,
                       noverlap=noverlap, detrend="constant")
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

# ---------------------------------------------------------------------
# Spectrogram plotting (moved from notebook)
# ---------------------------------------------------------------------
def plot_spectrogram(strain_whiten, fs, indxt, eventname, tevent, deltat, spec_cmap, plottype, det="H1"):
    """
    Plot and save a whitened spectrogram for a detector.
    Matches the notebook version but modularized.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    NFFT = int(fs / 16.0)
    NOVL = int(NFFT * 15 / 16.0)
    window = np.blackman(NFFT)

    plt.figure(figsize=(10, 6))
    plt.specgram(
        strain_whiten[indxt],
        NFFT=NFFT,
        Fs=fs,
        window=window,
        noverlap=NOVL,
        cmap=spec_cmap,
        xextent=[-deltat, deltat],
    )
    plt.xlabel("time (s) since " + str(tevent))
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.axis([-0.5, 0.5, 0, 500])
    plt.title(f"aLIGO {det} strain data near {eventname}")
    plt.savefig(f"figures/{eventname}_{det}_spectrogram_whitened.{plottype}",
                dpi=150, bbox_inches="tight")
    plt.close()
