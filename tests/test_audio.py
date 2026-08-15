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
