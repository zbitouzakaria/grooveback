"""Comparison sets must be decided by quality, not by level or clipping.

A sine at -14 LUFS peaks near -13 dBFS, far under the default -1 dBFS ceiling,
so tests that need the headroom branch to fire use a -20 dBFS ceiling the
fixtures actually cross.
"""

import numpy as np
import pytest

from grooveback import audio as ga
from grooveback.evaluation import (
    best_lag,
    bss_sdr_db,
    codec_edge_hz,
    level_matched_set,
    sdr_db,
    si_snr_db,
    spectral_snr_db,
    write_listening_pack,
)

SR = 44_100


def tone(
    amplitude: float, freq: float = 440.0, seconds: float = 2.0, channels: int = 2
) -> np.ndarray:
    """`(channels, samples)` float32 sine, identical in every channel."""
    t = np.arange(int(seconds * SR)) / SR
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.tile(mono, (channels, 1))


def test_sdr_measures_orthogonal_error_below_reference():
    """A 200 Hz error tone 40 dB under a 100 Hz reference.

    Both tones complete whole cycles over the fixture, so they are orthogonal
    and the SDR is exactly the amplitude ratio: 20·log10(1.0/0.01) = 40 dB.
    abs=0.05 absorbs float32 sine-synthesis roundoff in the cross term.
    """
    reference = tone(amplitude=1.0, freq=100.0)
    estimate = reference + tone(amplitude=0.01, freq=200.0)

    assert sdr_db(reference, estimate) == pytest.approx(40.0, abs=0.05)


def test_sdr_of_identical_signals_is_infinite():
    reference = tone(amplitude=0.5)

    assert sdr_db(reference, reference) == float("inf")


def test_sdr_penalizes_pure_gain_error():
    """Halving the signal leaves an error of half the signal: 20·log10(2) dB."""
    reference = tone(amplitude=1.0)

    assert sdr_db(reference, 0.5 * reference) == pytest.approx(6.021, abs=0.01)


def test_sdr_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        sdr_db(tone(amplitude=1.0), tone(amplitude=1.0)[:, :100])


def test_si_snr_ignores_pure_gain_error():
    """The projection absorbs any gain, leaving zero residual."""
    reference = tone(amplitude=1.0)

    assert si_snr_db(reference, 0.5 * reference) == float("inf")


def test_si_snr_measures_orthogonal_error_like_snr():
    """With no gain error the projection is the reference itself, so the score
    reduces to plain SNR: 20·log10(1.0/0.01) = 40 dB."""
    reference = tone(amplitude=1.0, freq=100.0)
    estimate = reference + tone(amplitude=0.01, freq=200.0)

    assert si_snr_db(reference, estimate) == pytest.approx(40.0, abs=0.05)


def test_si_snr_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        si_snr_db(tone(amplitude=1.0), tone(amplitude=1.0)[:, :100])


def test_bss_sdr_of_identical_signals_is_effectively_infinite():
    """A numerical least-squares fit, not symbolic: identity lands around
    250–300 dB depending on roundoff. 60 dB is a loose floor no real
    restoration approaches."""
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((2, 2 * SR)).astype(np.float32)

    assert bss_sdr_db(reference, reference) > 60.0


def test_bss_sdr_absorbs_delays_inside_its_filter_only():
    """The defining difference between the two SDRs: a delay inside the
    512-tap distortion filter is mostly forgiven, while plain SDR sees
    decorrelated noise (about −3 dB for white noise). Absorption is not exact
    — fast_bss_eval's regularized solve leaves ~26 dB at a 100-sample shift,
    measured — and past the filter length it collapses entirely. The shift is
    circular because the filter is fitted with FFT convolution."""
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((1, 2 * SR)).astype(np.float32)

    assert bss_sdr_db(reference, np.roll(reference, 100, axis=1)) > 20.0
    assert sdr_db(reference, np.roll(reference, 100, axis=1)) < 0.0
    assert bss_sdr_db(reference, np.roll(reference, 600, axis=1)) < 0.0


def test_bss_sdr_matches_plain_sdr_for_out_of_band_error():
    """A filter of a 100 Hz reference can only produce 100 Hz content, so a
    200 Hz error tone stays error for both metrics: 20·log10(1.0/0.01) =
    40 dB. abs=0.5 covers the solver's regularization and frame edges."""
    reference = tone(amplitude=1.0, freq=100.0)
    estimate = reference + tone(amplitude=0.01, freq=200.0)

    assert bss_sdr_db(reference, estimate) == pytest.approx(40.0, abs=0.5)


def test_bss_sdr_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        bss_sdr_db(tone(amplitude=1.0), tone(amplitude=1.0)[:, :100])


def test_spectral_snr_of_identical_signals_is_infinite():
    reference = tone(amplitude=0.5)

    assert spectral_snr_db(reference, reference) == float("inf")


def test_spectral_snr_ignores_polarity_while_sdr_does_not():
    """A polarity flip leaves every STFT magnitude untouched but makes the
    waveforms anti-correlated: the error is 2·ref, so SDR is
    10·log10(1/4) = −6.02 dB."""
    reference = tone(amplitude=0.5)

    assert spectral_snr_db(reference, -reference) == float("inf")
    assert sdr_db(reference, -reference) == pytest.approx(-6.021, abs=0.01)


def test_spectral_snr_of_silence_is_zero():
    """Silence has zero magnitude everywhere, so the error is the reference
    itself — the 0 dB baseline an informative fill must beat."""
    reference = tone(amplitude=0.5)

    assert spectral_snr_db(reference, np.zeros_like(reference)) == pytest.approx(
        0.0, abs=1e-6
    )


def band_limited_noise(cutoff_hz: float, seconds: float = 4.0) -> np.ndarray:
    """`(1, samples)` white noise with everything at or above `cutoff_hz`
    zeroed, from a fixed seed."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1, int(seconds * SR)))
    spectrum = np.fft.rfft(noise, axis=1)
    freqs = np.fft.rfftfreq(noise.shape[1], 1.0 / SR)
    spectrum[:, freqs >= cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, n=noise.shape[1], axis=1).astype(np.float32)


def test_codec_edge_reads_a_brickwall_within_one_band():
    reference = band_limited_noise(cutoff_hz=22_050)
    twin = band_limited_noise(cutoff_hz=11_000)

    assert codec_edge_hz(reference, twin, SR) == pytest.approx(11_000, abs=250)


def test_codec_edge_ignores_bands_where_the_reference_is_silent():
    """Above 8 kHz both signals are silence and would 'track' spuriously; the
    edge must still read the twin's own 5 kHz cutoff."""
    reference = band_limited_noise(cutoff_hz=8_000)
    twin = band_limited_noise(cutoff_hz=5_000)

    assert codec_edge_hz(reference, twin, SR) == pytest.approx(5_000, abs=250)


def test_best_lag_finds_a_delayed_estimate():
    """A codec-style delay: the estimate starts 100 samples late."""
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((1, 3 * SR)).astype(np.float32)
    delayed = np.concatenate(
        [np.zeros((1, 100), dtype=np.float32), reference[:, :-100]], axis=1
    )

    assert best_lag(reference, delayed) == 100


def test_best_lag_is_zero_when_aligned():
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((2, 3 * SR)).astype(np.float32)

    assert best_lag(reference, reference) == 0


def test_best_lag_rejects_signals_shorter_than_the_probe():
    with pytest.raises(ValueError, match="too short"):
        best_lag(tone(amplitude=1.0, seconds=1.0), tone(amplitude=1.0, seconds=1.0))


def test_items_are_matched_to_the_target_loudness():
    items = {"a": tone(amplitude=0.05), "b": tone(amplitude=0.2)}

    out = level_matched_set(items, SR, ceiling_dbfs=-1.0)

    assert ga.loudness(out["a"], SR) == pytest.approx(ga.TARGET_LUFS, abs=0.1)
    assert ga.loudness(out["b"], SR) == pytest.approx(ga.TARGET_LUFS, abs=0.1)


def test_peaks_are_pulled_under_the_ceiling():
    """Restoration output routinely lands above the ceiling after matching."""
    items = {"a": tone(amplitude=0.9), "b": tone(amplitude=0.1)}

    out = level_matched_set(items, SR, ceiling_dbfs=-20.0)

    # 1e-6 absorbs float32 roundoff in the applied gain.
    assert ga.peak_dbfs(out["a"]) <= -20.0 + 1e-6
    assert ga.peak_dbfs(out["b"]) <= -20.0 + 1e-6


def test_headroom_gain_is_shared_so_the_match_survives():
    """The whole point: one gain for the set, not per-item normalisation.

    K-weighting hears 3 kHz about 5 dB louder than 100 Hz, so after matching
    the two tones sit at different amplitudes and different peaks. A shared
    headroom gain keeps their loudness equal; scaling each item to the ceiling
    separately would equalise the peaks instead and break the match.
    """
    items = {"low": tone(amplitude=0.5, freq=100.0), "high": tone(amplitude=0.5, freq=3000.0)}

    out = level_matched_set(items, SR, ceiling_dbfs=-20.0)

    assert ga.loudness(out["low"], SR) == pytest.approx(ga.loudness(out["high"], SR), abs=0.1)
    assert ga.peak_dbfs(out["high"]) < ga.peak_dbfs(out["low"]) - 1.0


def test_level_matching_preserves_the_relative_spectrum():
    """Both stages are gains, so they must not change what is being compared.

    The two in-band tones sit 12 dB apart; that distance must survive matching
    and headroom. abs=0.01 allows only float32 scaling roundoff.
    """
    before = tone(amplitude=0.4, freq=200.0) + tone(amplitude=0.1, freq=500.0)

    after = level_matched_set({"a": before}, SR, ceiling_dbfs=-20.0)["a"]

    delta_before = ga.band_energy_db(before, SR, low_hz=300, high_hz=600) - ga.band_energy_db(
        before, SR, low_hz=100, high_hz=300
    )
    delta_after = ga.band_energy_db(after, SR, low_hz=300, high_hz=600) - ga.band_energy_db(
        after, SR, low_hz=100, high_hz=300
    )
    assert delta_after == pytest.approx(delta_before, abs=0.01)


def test_write_listening_pack_writes_level_matched_files(tmp_path):
    written = write_listening_pack({"x": tone(amplitude=0.5)}, SR, tmp_path)

    loaded, sample_rate = ga.load(written["x"])
    assert sample_rate == SR
    assert ga.loudness(loaded, SR) == pytest.approx(ga.TARGET_LUFS, abs=0.1)


def test_write_listening_pack_prefixes_mono_files(tmp_path):
    items = {"x": tone(amplitude=0.5, channels=1), "y": tone(amplitude=0.5)}

    written = write_listening_pack(items, SR, tmp_path)

    assert written["x"].name == "mono_x.wav"
    assert written["y"].name == "y.wav"
    assert written["x"].exists()
    assert written["y"].exists()
