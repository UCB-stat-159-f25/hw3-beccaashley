import numpy as np
import pytest
from ligotools.utils import whiten, reqshift, write_wavfile

def test_whiten_output_shape():
    """whiten() should preserve the shape of the input strain array."""
    strain = np.random.randn(4096)
    psd = lambda f: np.ones_like(f)
    dt = 1.0 / 4096
    white = whiten(strain, psd, dt)
    assert white.shape == strain.shape
    assert not np.isnan(white).any()

def test_reqshift_phase_rotation():
    """reqshift() should phase-shift without changing magnitude spectrum."""
    fs = 4096
    t = np.linspace(0, 1, fs)
    x = np.sin(2 * np.pi * 100 * t)
    y = reqshift(x, fshift=50, sample_rate=fs)
    assert len(y) == len(x)
    assert np.isclose(np.std(x), np.std(y), rtol=0.05)

def test_write_wavfile_creates_file(tmp_path):
    """write_wavfile() should create a valid .wav file."""
    filename = tmp_path / "test.wav"
    fs = 4096
    data = np.random.randn(fs)
    write_wavfile(str(filename), fs, data)
    assert filename.exists()
