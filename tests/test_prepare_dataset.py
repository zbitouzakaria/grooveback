"""scripts/prepare_dataset.py turns a folder of clean tracks into the
training layout: the original and one aligned MP3 per bitrate. The codec is
replaced by a stand-in, so these run without sox; codec_simu has its own
tests in test_musdb_eval.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from grooveback import audio as ga

SR = 44_100
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"
_spec = importlib.util.spec_from_file_location("prepare_dataset", _SCRIPT)
prepare_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_dataset)


def write_track(path, *, seconds, seed):
    """White noise; 3 s leaves best_lag the 1.2 s it needs around its probe."""
    rng = np.random.default_rng(seed)
    track = rng.uniform(-0.5, 0.5, size=(2, int(seconds * SR))).astype(np.float32)
    ga.save(path, track, SR)
    return track


def test_writes_the_original_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_dataset, "codec_simu", lambda clip, kbps: 0.5 * clip)
    track = write_track(tmp_path / "wav" / "song.wav", seconds=3.0, seed=0)

    prepare_dataset.main([str(tmp_path / "wav"), str(tmp_path / "out"), "--kbps", "24"])

    np.testing.assert_array_equal(ga.load(tmp_path / "out" / "song" / "original.wav")[0], track)


def test_writes_each_bitrate_from_the_codec_at_that_bitrate(tmp_path, monkeypatch):
    """The stand-in scales by kbps / 256, so each file shows the bitrate it was coded at."""
    monkeypatch.setattr(prepare_dataset, "codec_simu", lambda clip, kbps: clip * (kbps / 256))
    track = write_track(tmp_path / "wav" / "song.wav", seconds=3.0, seed=0)

    prepare_dataset.main([str(tmp_path / "wav"), str(tmp_path / "out"), "--kbps", "24", "128"])

    low, _ = ga.load(tmp_path / "out" / "song" / "24k.wav")
    high, _ = ga.load(tmp_path / "out" / "song" / "128k.wav")
    np.testing.assert_array_equal(low, 0.09375 * track)  # 24 / 256
    np.testing.assert_array_equal(high, 0.5 * track)  # 128 / 256


def test_refuses_an_mp3_that_is_one_sample_late(tmp_path, monkeypatch):
    """One sample of shift would teach the model a delay instead of the damage."""
    monkeypatch.setattr(prepare_dataset, "codec_simu", lambda clip, kbps: np.roll(clip, 1, axis=-1))
    write_track(tmp_path / "wav" / "song.wav", seconds=3.0, seed=0)

    with pytest.raises(RuntimeError, match="samples off"):
        prepare_dataset.main([str(tmp_path / "wav"), str(tmp_path / "out"), "--kbps", "24"])

    assert not (tmp_path / "out" / "song" / "24k.wav").exists()
