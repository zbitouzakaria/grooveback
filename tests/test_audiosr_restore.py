"""The AudioSR driver's chunked inference path.

The driver runs inside the audiosr venv, but it imports the model lazily, so
the main environment can load the file and drive the whole path with the
model faked at its `sr_fn` seam — no checkpoint, no second venv.
"""

import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "audiosr_restore.py"
_spec = importlib.util.spec_from_file_location("audiosr_restore", _DRIVER)
audiosr_restore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audiosr_restore)

SR = 48_000
GRAIN = int(5.12 * SR)


def padding_identity(wav_path: str) -> np.ndarray:
    """The real model's length behavior without its content: returns the
    chunk unchanged, padded up to the 5.12 s grain, in the model's
    `(1, samples)` shape."""
    audio, _ = sf.read(wav_path, dtype="float32")
    pad = (-audio.size) % GRAIN
    return np.pad(audio, (0, pad))[None, :]


def test_identity_model_through_the_chunked_path_returns_the_input(tmp_path):
    """The audio.md identity gate at the driver's seam: chunk positioning,
    per-chunk trimming of the model's padding, the crossfaded joins, and the
    final length must all cancel. 48 kHz input so both resamples are skipped
    and equality is down to crossfade arithmetic (atol as in the Apollo
    identity test). The length is 7.3 s against 2 s chunks: several joins
    plus a partial last chunk that still exceeds the fade.
    """
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.5, 0.5, size=(2, int(7.3 * SR))).astype(np.float32)
    sf.write(str(tmp_path / "in.wav"), audio.T, SR, subtype="FLOAT")

    audiosr_restore.main(
        [str(tmp_path / "in.wav"), str(tmp_path / "out.wav"), "--chunk-seconds", "2"],
        sr_fn=padding_identity,
    )

    out, rate = sf.read(str(tmp_path / "out.wav"), dtype="float32", always_2d=True)
    assert rate == SR
    assert out.T.shape == audio.shape
    np.testing.assert_allclose(out.T, audio, rtol=0, atol=1e-6)


def normalizing_identity(wav_path: str) -> np.ndarray:
    """audiosr 0.0.7's real level behavior: whatever comes in, the render is
    peak-normalized to 0.5 (its DC removal is left out — reconstructing a DC
    offset is not the driver's contract)."""
    audio, _ = sf.read(wav_path, dtype="float32")
    pad = (-audio.size) % GRAIN
    audio = 0.5 * audio / (np.max(np.abs(audio)) + 1e-8)
    return np.pad(audio, (0, pad))[None, :]


def test_package_peak_normalization_is_undone_per_chunk(tmp_path):
    """The package discards level (every render peaks at 0.5), so without
    the driver's per-chunk gain fit a quiet chunk would come back louder
    than a loud one. Two chunks 20 dB apart must come back 20 dB apart.
    atol covers the fit's float32 arithmetic on an exact-recovery case."""
    rng = np.random.default_rng(0)
    loud = rng.uniform(-0.8, 0.8, size=2 * SR).astype(np.float32)
    quiet = 0.1 * rng.uniform(-0.8, 0.8, size=2 * SR).astype(np.float32)
    audio = np.concatenate([loud, quiet])[None, :]
    sf.write(str(tmp_path / "in.wav"), audio.T, SR, subtype="FLOAT")

    audiosr_restore.main(
        [str(tmp_path / "in.wav"), str(tmp_path / "out.wav"), "--chunk-seconds", "2"],
        sr_fn=normalizing_identity,
    )

    out, _ = sf.read(str(tmp_path / "out.wav"), dtype="float32", always_2d=True)
    np.testing.assert_allclose(out.T, audio, rtol=0, atol=1e-5)
