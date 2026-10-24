"""The AudioSR driver, `scripts/audiosr_restore.py`.

The driver runs in AudioSR's own environment but imports the model lazily, so
this environment can run it end to end with the model replaced at its `sr_fn`
argument: no checkpoint, no second environment.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "audiosr_restore.py"
_spec = importlib.util.spec_from_file_location("audiosr_restore", _DRIVER)
audiosr_restore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audiosr_restore)

SR = 44_100


class NormalizingIdentity:
    """AudioSR's level behavior without its content: the file it is given,
    scaled to a peak of 0.5 and zero-padded to 10.24 s at 48 kHz, in the
    model's `(1, 1, samples)` shape. The real model pads to the next multiple
    of 5.12 s; 10.24 s is the longest chunk the driver sends, so it covers
    every call. Records each call's sample count."""

    def __init__(self):
        self.sizes = []

    def __call__(self, wav_path):
        audio, _ = sf.read(wav_path, dtype="float32")
        self.sizes.append(audio.size)
        padded = np.pad(audio, (0, int(10.24 * 48_000) - audio.size))
        return (0.5 * padded / np.abs(padded).max())[None, None, :]


class InventingModel:
    """A model that keeps its input and adds noise it invents anew on each
    call, as a diffusion model invents the band above the cut-off."""

    def __init__(self):
        self.calls = 0

    def __call__(self, wav_path):
        audio, _ = sf.read(wav_path, dtype="float64")
        self.calls += 1
        return audio + np.random.default_rng(self.calls).normal(0.0, 0.01, audio.size)


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


def test_identity_model_returns_a_long_file_unchanged_across_chunk_seams(tmp_path):
    """25 s is three chunks per channel, so two crossfades and three gains are
    on the path. The tones are faded in and out with a Hann window, so the
    resampler has no edge to ring on: atol=1e-6 is then above the 1.5e-8 that
    the 44.1 -> 48 -> 44.1 kHz round trip alone measures."""
    t = np.arange(25 * SR) / SR
    tones = np.stack([0.2 * np.sin(2 * np.pi * 1_000 * t), 0.1 * np.sin(2 * np.pi * 5_000 * t)])
    audio = (np.hanning(25 * SR) * tones).astype(np.float32)
    (tmp_path / "mp3" / "track").mkdir(parents=True)
    sf.write(tmp_path / "mp3" / "track" / "0.wav", audio.T, SR, subtype="FLOAT")

    audiosr_restore.main(
        [str(tmp_path / "mp3"), str(tmp_path / "audiosr")], sr_fn=NormalizingIdentity()
    )

    out, _ = sf.read(tmp_path / "audiosr" / "track" / "0.wav", dtype="float32", always_2d=True)
    np.testing.assert_allclose(out.T, audio, rtol=0, atol=1e-6)


def test_model_receives_a_long_file_in_chunks_of_at_most_10_24_s(tmp_path):
    """25 s at 48 kHz is 1,200,000 samples. Chunks of 491,520 (10.24 s) start
    every 486,720 (less the 4,800-sample fade): at 0, 486,720 and 973,440, so
    the last is 226,560 long."""
    audio = np.zeros((2, 25 * SR), dtype=np.float32)
    audio[:, ::1_000] = 0.5
    (tmp_path / "mp3" / "track").mkdir(parents=True)
    sf.write(tmp_path / "mp3" / "track" / "0.wav", audio.T, SR, subtype="FLOAT")
    model = NormalizingIdentity()

    audiosr_restore.main([str(tmp_path / "mp3"), str(tmp_path / "audiosr")], sr_fn=model)

    assert model.sizes == [491_520, 491_520, 226_560, 491_520, 491_520, 226_560]


def test_content_each_chunk_invents_differently_is_3_db_quieter_mid_seam(tmp_path):
    """The test .claude/rules/audio.md asks for before chunking a model whose
    output changes from call to call. At the middle of a crossfade, the two
    chunks' inventions each weigh 1/2, so unrelated content keeps 1/4 + 1/4 of
    its power: -3 dB. The window is the middle fifth of the fade, where the
    loss is 2.8 to 3.0 dB; abs=0.5 covers the spread of a power estimate over
    its 960 noise samples (about 0.2 dB)."""
    channel = (0.5 * np.sin(2 * np.pi * 440 * np.arange(600_000) / 48_000)).astype(np.float32)

    out = audiosr_restore.restore_channel(channel, sr_fn=InventingModel(), tmp_dir=tmp_path)

    invented = out.astype(np.float64) - channel
    seam = invented[486_720 + 2_400 - 480 : 486_720 + 2_400 + 480]  # the second chunk's fade
    away = invented[100_000:200_000]
    assert 10 * np.log10(np.mean(seam**2) / np.mean(away**2)) == pytest.approx(-3.0, abs=0.5)


def test_model_receives_each_channel_resampled_to_48_khz(tmp_path):
    """6 s at 48 kHz is 288,000 samples; unresampled, a channel would be 264,600."""
    audio = np.zeros((2, 6 * SR), dtype=np.float32)
    audio[:, 1_000] = 0.5
    (tmp_path / "mp3" / "track").mkdir(parents=True)
    sf.write(tmp_path / "mp3" / "track" / "0.wav", audio.T, SR, subtype="FLOAT")
    model = NormalizingIdentity()

    audiosr_restore.main([str(tmp_path / "mp3"), str(tmp_path / "audiosr")], sr_fn=model)

    assert model.sizes == [288_000, 288_000]


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
    assert model.sizes == []
    assert np.all(kept == 0.25)
