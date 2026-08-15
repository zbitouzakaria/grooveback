"""Guards on the audio helpers.

The size bounds matter as much as the maths here: an unbounded spectrogram or a
full-length notebook player is a hang, not an error, which is the worst way for
a tool to fail.
"""

import numpy as np
import pytest

from grooveback import audio as ga

SR = 44_100


def tone(seconds: float, freq: float = 440.0, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return np.tile(0.5 * np.sin(2 * np.pi * freq * t), (channels, 1))


def test_spectrogram_is_calibrated_to_dbfs():
    """A full-scale sine must read ~0 dB, not the raw FFT magnitude.

    An uncalibrated `rfft` sits about +53 dB high at n_fft=2048. Nothing errors;
    the plot just disagrees with every real analyser and makes faint content
    look loud. The tolerance covers Hann scalloping loss (up to ~1.4 dB) for a
    tone that does not land on a bin centre.
    """
    t = np.arange(SR, dtype=np.float32) / SR
    sine = np.tile(np.sin(2 * np.pi * 1000 * t), (2, 1))
    assert ga.spectrogram_db(sine).max() == pytest.approx(0.0, abs=1.5)


@pytest.mark.parametrize("amplitude", [1.0, 0.1, 0.01])
def test_band_energy_matches_sine_rms(amplitude):
    """Parseval scaling: a tone's band energy is its time-domain RMS."""
    t = np.arange(SR, dtype=np.float32) / SR
    sine = np.tile(amplitude * np.sin(2 * np.pi * 1000 * t), (2, 1))
    expected = 20 * np.log10(amplitude / np.sqrt(2))
    assert ga.band_energy_db(sine, SR, 900, 1100) == pytest.approx(expected, abs=0.1)


def test_band_energy_excludes_out_of_band_content():
    t = np.arange(SR, dtype=np.float32) / SR
    sine = np.tile(np.sin(2 * np.pi * 1000 * t), (2, 1))
    assert ga.band_energy_db(sine, SR, 5000, 10000) < -60


def test_band_energy_of_empty_band_is_negative_infinity():
    assert ga.band_energy_db(tone(1.0), SR, 30000, 40000) == float("-inf")


def test_spectrogram_frames_bounded_for_long_input():
    """A seven-minute track must not produce 39k columns."""
    spec = ga.spectrogram_db(tone(456.0))
    assert spec.shape[1] <= 4000


def test_spectrogram_keeps_native_hop_when_short():
    audio = tone(5.0)
    spec = ga.spectrogram_db(audio, n_fft=2048, hop=512)
    assert spec.shape[1] == 1 + (audio.shape[1] - 2048) // 512
    assert spec.shape[0] == 1025


def test_spectrogram_survives_input_shorter_than_fft():
    assert ga.spectrogram_db(tone(0.001)).shape[1] >= 1


@pytest.mark.parametrize("target", [-14.0, -23.0])
def test_normalize_loudness_hits_target(target):
    result = ga.normalize_loudness(tone(5.0), SR, target_lufs=target)
    assert ga.loudness(result, SR) == pytest.approx(target, abs=0.1)


def test_normalize_loudness_preserves_shape_and_channels():
    audio = tone(5.0, channels=2)
    assert ga.normalize_loudness(audio, SR).shape == audio.shape


def test_peak_dbfs_of_silence_is_negative_infinity():
    assert ga.peak_dbfs(np.zeros((2, 100), dtype=np.float32)) == float("-inf")


def test_roundtrip_through_disk(tmp_path):
    """Sample rate and channel count survive a write and read."""
    audio = tone(1.0)
    path = tmp_path / "nested" / "out.wav"
    ga.save(path, audio, SR)
    loaded, sr = ga.load(path)
    assert sr == SR
    assert loaded.shape == audio.shape
    assert np.allclose(loaded, audio, atol=1e-6)
