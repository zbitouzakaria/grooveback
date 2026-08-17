"""Comparison sets must be decided by quality, not by level or clipping."""

import numpy as np
import pytest

from grooveback import audio as ga
from grooveback.evaluation import level_matched_set, write_listening_pack

SR = 44_100


def tone(amplitude: float, freq: float = 440.0, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return np.tile(amplitude * np.sin(2 * np.pi * freq * t), (2, 1))


def test_nothing_exceeds_the_ceiling():
    """Restoration output routinely lands above 0 dBFS after level matching."""
    out = level_matched_set({"a": tone(0.9), "b": tone(0.1)}, SR, ceiling_dbfs=-1.0)
    for audio in out.values():
        assert ga.peak_dbfs(audio) <= -1.0 + 1e-6


def test_headroom_gain_is_common_so_matching_survives():
    """The whole point: one gain for the set, not per-item normalisation.

    Scaling each item to the ceiling separately would undo the loudness match
    and hand the comparison back to whichever file is loudest.
    """
    out = level_matched_set({"a": tone(0.9), "b": tone(0.1)}, SR)
    a, b = ga.loudness(out["a"], SR), ga.loudness(out["b"], SR)
    assert a == pytest.approx(b, abs=0.1)


def test_quiet_set_is_left_alone():
    out = level_matched_set({"a": tone(0.05)}, SR, ceiling_dbfs=-1.0)
    assert ga.loudness(out["a"], SR) == pytest.approx(ga.TARGET_LUFS, abs=0.1)


def test_relative_spectrum_is_untouched():
    """Headroom is a gain, so it must not change what is being compared."""
    before = tone(0.9)
    after = level_matched_set({"a": before}, SR)["a"]
    assert ga.band_energy_db(after, SR, 300, 600) - ga.band_energy_db(
        after, SR, 100, 300
    ) == pytest.approx(
        ga.band_energy_db(before, SR, 300, 600)
        - ga.band_energy_db(before, SR, 100, 300),
        abs=0.01,
    )


def test_pack_prefixes_mono_files(tmp_path):
    written = write_listening_pack({"x": tone(0.5)[:1], "y": tone(0.5)}, SR, tmp_path)
    assert written["x"].name == "mono_x.wav"
    assert written["y"].name == "y.wav"
    assert all(p.exists() for p in written.values())
