"""Cutoff detection for bandwidth-extension models.

Getting this wrong is not a small error. A2SB regenerates everything above the
cutoff it is told, so an underestimate makes it overwrite content that was
already there — and since it works per channel, the rebuilt band comes back
decorrelated and the stereo image collapses toward mono.
"""

import numpy as np
import pytest

from grooveback import audio as ga

SR = 44_100


def band_limited(cutoff_hz: float, seconds: float = 2.0) -> np.ndarray:
    """Pink-ish noise brick-walled at `cutoff_hz`, like a codec leaves it."""
    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spectrum /= np.maximum(freqs, 20.0)  # tilt energy toward the bottom, as music does
    spectrum[freqs >= cutoff_hz] = 0.0
    signal = np.fft.irfft(spectrum, n).astype(np.float32)
    signal /= np.max(np.abs(signal))
    return np.tile(signal, (2, 1))


def smeared(cutoff_hz: float, width_hz: float = 1000.0, seconds: float = 4.0) -> np.ndarray:
    """A rolloff that fades over `width_hz` rather than stopping dead.

    This is what real codecs leave behind, and it is the case a brick-wall
    fixture does not cover: the drop is spread across a kilohertz, so no pair of
    narrow adjacent bands ever shows the whole step.
    """
    rng = np.random.default_rng(1)
    n = int(seconds * SR)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spectrum /= np.maximum(freqs, 20.0)
    taper = np.clip((cutoff_hz + width_hz - freqs) / width_hz, 0.0, 1.0)
    spectrum *= taper**4
    signal = np.fft.irfft(spectrum, n).astype(np.float32)
    signal /= np.max(np.abs(signal))
    return np.tile(signal, (2, 1))


@pytest.mark.parametrize("cutoff", [5000, 8000, 11000, 16000, 20000])
def test_finds_the_brick_wall(cutoff):
    found = ga.bandwidth_hz(band_limited(cutoff), SR)
    assert abs(found - cutoff) <= 1250, f"expected ~{cutoff}, got {found}"


@pytest.mark.parametrize("cutoff", [8000, 12000, 16000])
def test_finds_a_smeared_rolloff(cutoff):
    """The real-world case. Detection must not need a perfectly sharp edge."""
    found = ga.bandwidth_hz(smeared(cutoff), SR)
    assert abs(found - cutoff) <= 2000, f"expected ~{cutoff}, got {found}"


def test_detection_is_stable_across_excerpt_lengths():
    """A short excerpt and a long one must agree, or the same track gets
    restored differently depending on how much of it you passed in."""
    long_signal = smeared(16000, seconds=20.0)
    short = ga.bandwidth_hz(long_signal[:, : 3 * SR], SR)
    full = ga.bandwidth_hz(long_signal, SR)
    assert abs(short - full) <= 1000, f"{short} vs {full}"


def test_full_band_audio_reports_nyquist():
    """No cliff means nothing to extend; must not invent a low cutoff."""
    assert ga.bandwidth_hz(band_limited(22050), SR) >= 20000


def test_not_fooled_by_bass_heavy_energy_distribution():
    """The failure this function exists to avoid.

    A 99% energy rolloff returns ~2 kHz for music regardless of real bandwidth,
    because that is where the energy is. A2SB's own helper does exactly that,
    which is why its shipped config claims a 2 kHz cutoff.
    """
    signal = band_limited(16000)
    t = np.arange(signal.shape[1]) / SR
    signal = signal + 8.0 * np.sin(2 * np.pi * 60 * t)  # dominant bass
    assert ga.bandwidth_hz(signal, SR) > 12000


def test_ignores_cliffs_below_min_hz():
    """A spectral hole in the mids is not a bandwidth limit."""
    found = ga.bandwidth_hz(band_limited(18000), SR, min_hz=3000)
    assert found > 15000
