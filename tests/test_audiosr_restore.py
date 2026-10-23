"""The AudioSR driver, `scripts/audiosr_restore.py`.

The driver runs in AudioSR's own environment but imports the model lazily, so
this environment can run it end to end with the model replaced at its `sr_fn`
argument: no checkpoint, no second environment.
"""

import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "audiosr_restore.py"
_spec = importlib.util.spec_from_file_location("audiosr_restore", _DRIVER)
audiosr_restore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audiosr_restore)

SR = 44_100


class NormalizingIdentity:
    """AudioSR's level behavior without its content: the file it is given,
    scaled to a peak of 0.5 and padded to the model's 5.12 s grain at 48 kHz,
    in the model's `(1, 1, samples)` shape. Records each call's sample rate."""

    def __init__(self):
        self.rates = []

    def __call__(self, wav_path):
        audio, rate = sf.read(wav_path, dtype="float32")
        self.rates.append(rate)
        padded = np.pad(audio, (0, int(10.24 * 48_000) - audio.size))
        return (0.5 * padded / np.abs(padded).max())[None, None, :]


def test_identity_model_returns_the_input_at_its_level(tmp_path):
    """The identity gate (.claude/rules/audio.md) through both resamples, the
    trim of the model's padding, and the gain that undoes its peak scaling.
    The tones are faded in and out with a Hann window, so the resampler has no
    edge to ring on: atol=1e-6 is then above the 1.5e-8 that the 44.1 → 48 →
    44.1 kHz round trip alone measures."""
    t = np.arange(6 * SR) / SR
    tones = np.stack([0.2 * np.sin(2 * np.pi * 1_000 * t), 0.1 * np.sin(2 * np.pi * 5_000 * t)])
    audio = (np.hanning(6 * SR) * tones).astype(np.float32)
    (tmp_path / "mp3" / "track").mkdir(parents=True)
    sf.write(tmp_path / "mp3" / "track" / "0.wav", audio.T, SR, subtype="FLOAT")

    audiosr_restore.main(
        [str(tmp_path / "mp3"), str(tmp_path / "audiosr")], sr_fn=NormalizingIdentity()
    )

    out, rate = sf.read(tmp_path / "audiosr" / "track" / "0.wav", dtype="float32", always_2d=True)
    assert rate == SR
    np.testing.assert_allclose(out.T, audio, rtol=0, atol=1e-6)


def test_model_receives_each_channel_at_48_khz(tmp_path):
    audio = np.zeros((2, 6 * SR), dtype=np.float32)
    audio[:, 1_000] = 0.5
    (tmp_path / "mp3" / "track").mkdir(parents=True)
    sf.write(tmp_path / "mp3" / "track" / "0.wav", audio.T, SR, subtype="FLOAT")
    model = NormalizingIdentity()

    audiosr_restore.main([str(tmp_path / "mp3"), str(tmp_path / "audiosr")], sr_fn=model)

    assert model.rates == [48_000, 48_000]


def test_existing_render_is_kept(tmp_path):
    """A rerun after an interruption restores only what is missing."""
    audio = np.zeros((2, 6 * SR), dtype=np.float32)
    (tmp_path / "mp3" / "track").mkdir(parents=True)
    sf.write(tmp_path / "mp3" / "track" / "0.wav", audio.T, SR, subtype="FLOAT")
    (tmp_path / "audiosr" / "track").mkdir(parents=True)
    sf.write(tmp_path / "audiosr" / "track" / "0.wav", audio.T + 0.25, SR, subtype="FLOAT")
    model = NormalizingIdentity()

    audiosr_restore.main([str(tmp_path / "mp3"), str(tmp_path / "audiosr")], sr_fn=model)

    kept, _ = sf.read(tmp_path / "audiosr" / "track" / "0.wav", dtype="float32")
    assert model.rates == []
    assert np.all(kept == 0.25)
