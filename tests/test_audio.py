"""Guards on the audio helpers.

The size bounds matter as much as the maths here: an unbounded spectrogram or a
full-length notebook player is a hang, not an error, which is the worst way for
a tool to fail.
"""

import numpy as np
import pytest

from grooveback import audio as ga

SR = 44_100


def tone(
    amplitude: float, freq: float = 440.0, seconds: float = 2.0, channels: int = 2
) -> np.ndarray:
    """`(channels, samples)` float32 sine, identical in every channel.

    The phase is computed in float64: over minutes of signal a float32 phase
    argument is quantized by whole radians, which turns the tone into noise.
    """
    t = np.arange(int(seconds * SR)) / SR
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.tile(mono, (channels, 1))


def test_save_load_roundtrip_is_exact(tmp_path):
    """FLOAT wav files carry float32 samples bit for bit, so no tolerance."""
    audio = tone(amplitude=0.5, freq=440.0, seconds=1.0)

    ga.save(tmp_path / "out.wav", audio, SR)
    loaded, sample_rate = ga.load(tmp_path / "out.wav")

    assert sample_rate == SR
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, audio)


def test_save_creates_parent_directories(tmp_path):
    audio = tone(amplitude=0.5, seconds=0.5)

    ga.save(tmp_path / "nested" / "out.wav", audio, SR)

    assert (tmp_path / "nested" / "out.wav").exists()


def test_load_returns_2d_even_for_mono(tmp_path):
    """Downstream code never branches on shape, so mono must come back
    `(1, samples)`, not `(samples,)`."""
    audio = tone(amplitude=0.5, seconds=0.5, channels=1)

    ga.save(tmp_path / "mono.wav", audio, SR)
    loaded, _ = ga.load(tmp_path / "mono.wav")

    assert loaded.shape == audio.shape


def test_loudness_matches_the_bs1770_sine_reference():
    """ITU-R BS.1770 documents the anchor: a 0 dBFS 997 Hz sine in a single
    channel reads -3.01 LKFS. This pins `loudness` to an oracle outside the
    module; the other tests that measure with it stand on this one. abs=0.1
    covers the K-weighting filter's small deviation from unity at 997 Hz.
    """
    sine = tone(amplitude=1.0, freq=997.0, seconds=5.0, channels=1)

    measured = ga.loudness(sine, SR)

    assert measured == pytest.approx(-3.01, abs=0.1)


@pytest.mark.parametrize("target", [-14.0, -23.0])
def test_normalize_loudness_hits_the_target(target):
    sine = tone(amplitude=0.5, seconds=5.0)

    result = ga.normalize_loudness(sine, SR, target_lufs=target)

    assert ga.loudness(result, SR) == pytest.approx(target, abs=0.1)


def test_normalize_loudness_preserves_shape():
    audio = tone(amplitude=0.5, seconds=5.0)

    result = ga.normalize_loudness(audio, SR)

    assert result.shape == audio.shape


def test_peak_dbfs_reads_the_sine_amplitude():
    # A 1 s sine samples densely enough that its true peak sits within
    # 0.01 dB of the amplitude.
    audio = tone(amplitude=0.5, freq=440.0, seconds=1.0)

    assert ga.peak_dbfs(audio) == pytest.approx(20 * np.log10(0.5), abs=0.01)


def test_peak_dbfs_of_silence_is_negative_infinity():
    silence = np.zeros((2, 100), dtype=np.float32)

    assert ga.peak_dbfs(silence) == float("-inf")


def test_spectrogram_is_calibrated_to_dbfs():
    """A full-scale sine must read ~0 dB, not the raw FFT magnitude.

    An uncalibrated `rfft` sits about +53 dB high at n_fft=2048. Nothing errors;
    the plot just disagrees with every real analyser and makes faint content
    look loud. abs=1.5 covers Hann scalloping loss (up to ~1.4 dB) for a tone
    that does not land on a bin centre.
    """
    sine = tone(amplitude=1.0, freq=1000.0)

    spec = ga.spectrogram_db(sine)

    assert spec.max() == pytest.approx(0.0, abs=1.5)


def test_spectrogram_keeps_the_native_hop_when_short():
    audio = tone(amplitude=0.5, seconds=5.0)  # 220_500 samples

    spec = ga.spectrogram_db(audio, n_fft=2048, hop=512)

    # (220_500 - 2_048) // 512 hops after the first frame -> 427 frames;
    # a 2_048-point FFT -> 1_025 bins.
    assert spec.shape == (1_025, 427)


def test_spectrogram_frames_are_bounded_for_long_input():
    """A seven-minute track must not produce 39k columns."""
    audio = tone(amplitude=0.5, seconds=456.0)  # 7.6 minutes

    spec = ga.spectrogram_db(audio)

    assert spec.shape[1] <= 4_000


def test_spectrogram_returns_a_frame_for_input_shorter_than_the_window():
    blip = tone(amplitude=0.5, seconds=0.001)  # 44 samples < n_fft

    spec = ga.spectrogram_db(blip)

    assert spec.shape[1] >= 1


@pytest.mark.parametrize("amplitude", [1.0, 0.1, 0.01])
def test_band_energy_matches_the_sine_rms(amplitude):
    """Parseval scaling: a tone's band energy is its time-domain RMS,
    20*log10(a/sqrt(2)). The tone sits exactly on a 1 Hz bin, so abs=0.1
    is generous."""
    sine = tone(amplitude=amplitude, freq=1000.0, seconds=1.0)

    measured = ga.band_energy_db(sine, SR, low_hz=900, high_hz=1100)

    assert measured == pytest.approx(20 * np.log10(amplitude / np.sqrt(2)), abs=0.1)


def test_band_energy_excludes_content_outside_the_band():
    # An on-bin tone puts only float32 roundoff into other bands (measured
    # about -164 dB); -100 proves exclusion and stays clear of that floor.
    sine = tone(amplitude=1.0, freq=1000.0, seconds=1.0)

    measured = ga.band_energy_db(sine, SR, low_hz=5000, high_hz=10000)

    assert measured < -100.0


def test_band_energy_of_an_empty_band_is_negative_infinity():
    """30-40 kHz holds no bins at all below the 22.05 kHz Nyquist."""
    sine = tone(amplitude=0.5, seconds=1.0)

    assert ga.band_energy_db(sine, SR, low_hz=30_000, high_hz=40_000) == float("-inf")


def test_band_energy_is_precise_at_full_track_lengths():
    """Regression: float32 FFT roundoff at ~20M samples buried a -82 dBFS band
    at -143 dBFS before the float64 cast in `band_energy_db`. The failure
    needs length, so short fixtures never catch it. abs=1.0 covers leakage
    from cutting the excerpt mid-cycle.
    """
    seconds = 456.0  # 7.6 minutes, AN-2's length
    loud = tone(amplitude=0.3, freq=440.0, seconds=seconds)
    # RMS of a sine is a/sqrt(2), so this amplitude puts the band at -82 dBFS.
    quiet = tone(amplitude=10 ** (-82.0 / 20.0) * np.sqrt(2), freq=16_500.0, seconds=seconds)
    audio = loud + quiet

    full = ga.band_energy_db(audio, SR, low_hz=16_000, high_hz=17_000)
    excerpt = ga.band_energy_db(audio[:, : 30 * SR], SR, low_hz=16_000, high_hz=17_000)

    assert full == pytest.approx(-82.0, abs=1.0)
    assert full == pytest.approx(excerpt, abs=1.0)
