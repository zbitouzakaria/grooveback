"""Comparison sets must be decided by quality, not by level or clipping.

A sine at -14 LUFS peaks near -13 dBFS, far under the default -1 dBFS ceiling,
so tests that need the headroom branch to fire use a -20 dBFS ceiling the
fixtures actually cross.
"""

import numpy as np
import pytest

from grooveback import audio as ga
from grooveback.evaluation import level_matched_set, write_listening_pack

SR = 44_100


def tone(
    amplitude: float, freq: float = 440.0, seconds: float = 2.0, channels: int = 2
) -> np.ndarray:
    """`(channels, samples)` float32 sine, identical in every channel."""
    t = np.arange(int(seconds * SR)) / SR
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.tile(mono, (channels, 1))


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
