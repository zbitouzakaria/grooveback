"""SAME latent space: encode, decode, and the MP3 transport vector.

SAME is the autoencoder Stable Audio 3 generates into, so it is the space any
future prior will live in. It compresses 44.1 kHz stereo by 4096x in time into
256 channels — about 10.8 latent frames per second.

Like A2SB, SAME runs behind a subprocess boundary: it needs `stable-audio-3`
and its pinned torch, which we do not want in this project's environment.
`scripts/same_codec.py` is the entry point on the other side.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from grooveback import audio as ga

SAME_SAMPLE_RATE = 44_100
SAME_VARIANTS = ("same-s", "same-l")
"""Only two exist. Stable Audio 3 small generates into `same-s`."""

_REPO = Path(__file__).resolve().parents[2]
_SA3_DIR = _REPO / "third_party" / "stable-audio-3"
_SA3_PYTHON = _SA3_DIR / ".venv" / "bin" / "python"
_DRIVER = _REPO / "scripts" / "same_codec.py"


def _run(args: list[str]) -> None:
    if not _SA3_PYTHON.exists():
        raise FileNotFoundError(
            f"stable-audio-3 environment missing at {_SA3_PYTHON}. Create it:\n"
            f"  git clone https://github.com/Stability-AI/stable-audio-3 {_SA3_DIR}\n"
            f"  cd {_SA3_DIR} && uv sync --frozen"
        )
    result = subprocess.run(
        [str(_SA3_PYTHON), str(_DRIVER), *args],
        cwd=_REPO, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"same_codec failed.\n{(result.stderr or result.stdout)[-2000:]}")


def encode_tree(directory: str | Path, variant: str = "same-s", device: str = "auto") -> None:
    """Encode every wav under `directory` to `<name>.<variant>.npy`.

    One model load for the whole tree, which matters for SAME-L where loading
    costs more than encoding a clip.
    """
    _run(["--model", variant, "--encode-tree", str(directory), "--device", device])


def encode(
    audio: np.ndarray, sample_rate: int, variant: str = "same-s", device: str = "auto"
) -> np.ndarray:
    """Audio `(channels, samples)` to latents `(256, frames)`."""
    if sample_rate != SAME_SAMPLE_RATE:
        raise ValueError(f"SAME expects {SAME_SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    with tempfile.TemporaryDirectory() as tmp:
        wav, npy = Path(tmp) / "in.wav", Path(tmp) / "z.npy"
        ga.save(wav, audio, sample_rate)
        _run(["--model", variant, "--input", str(wav), "--latents", str(npy),
              "--device", device])
        return np.load(npy)


def decode(
    latents: np.ndarray, variant: str = "same-s", device: str = "auto"
) -> np.ndarray:
    """Latents `(256, frames)` back to audio `(2, samples)`.

    Output length is the latent count times 4096, so it can overrun the
    original by up to one frame. Callers comparing against a source should trim
    both to the shorter length.
    """
    with tempfile.TemporaryDirectory() as tmp:
        npy, wav = Path(tmp) / "z.npy", Path(tmp) / "out.wav"
        np.save(npy, latents)
        _run(["--model", variant, "--from-latents", str(npy), "--decoded", str(wav),
              "--device", device])
        decoded, _ = ga.load(wav)
        return decoded


def roundtrip(
    audio: np.ndarray, sample_rate: int, variant: str = "same-s", device: str = "auto"
) -> np.ndarray:
    """`decode(encode(audio))` — what survives a trip through the latent space.

    Not an identity, and not even close on bandlimited input: the decoder
    invents high-frequency content that was not in the source. Any comparison
    of a latent-space method must use this as its reference, never the input
    file, or the autoencoder's own behaviour gets attributed to the method.
    """
    return decode(encode(audio, sample_rate, variant, device), variant, device)


def roundtrip_with_shift(
    audio: np.ndarray,
    sample_rate: int,
    shift: np.ndarray,
    variant: str = "same-s",
    device: str = "auto",
) -> np.ndarray:
    """`decode(encode(audio) + shift)` — the transport-vector restoration.

    `shift` is one `(256,)` vector applied to every latent frame. This is the
    whole method: no model, no iteration, one addition.
    """
    if shift.shape != (256,):
        raise ValueError(f"shift must be (256,), got {shift.shape}.")
    latents = encode(audio, sample_rate, variant, device)
    return decode(latents + shift[:, None], variant, device)


def transport_vector(
    clean: list[np.ndarray], degraded: list[np.ndarray]
) -> np.ndarray:
    """Mean latent offset from degraded to clean, over paired windows.

    Each pair is `(256, frames)` for the same audio before and after the codec.
    Returns the `(256,)` vector that `roundtrip_with_shift` applies.
    """
    if len(clean) != len(degraded) or not clean:
        raise ValueError("Need matching non-empty lists of clean and degraded latents.")
    per_window = []
    for zc, zd in zip(clean, degraded, strict=True):
        n = min(zc.shape[1], zd.shape[1])
        per_window.append((zc[:, :n] - zd[:, :n]).mean(axis=1))
    return np.stack(per_window).mean(axis=0)


def transport_linearity(
    clean: list[np.ndarray], degraded: list[np.ndarray], shift: np.ndarray
) -> dict[str, float]:
    """How well a single constant vector describes the damage.

    `variance_explained` is the share of frame-level difference energy the
    constant accounts for; `cosine_mean` is how consistently windows agree on
    its direction. Direction can be stable while magnitude is not, so both
    matter.
    """
    frames = []
    centroids = []
    for zc, zd in zip(clean, degraded, strict=True):
        n = min(zc.shape[1], zd.shape[1])
        diff = zc[:, :n] - zd[:, :n]
        frames.append(diff)
        centroids.append(diff.mean(axis=1))
    frames = np.concatenate(frames, axis=1)
    centroids = np.stack(centroids)
    total = float((frames**2).sum())
    residual = float(((frames - shift[:, None]) ** 2).sum())
    cos = centroids @ shift / (
        np.linalg.norm(centroids, axis=1) * np.linalg.norm(shift) + 1e-12
    )
    return {
        "norm": float(np.linalg.norm(shift)),
        "variance_explained": 1.0 - residual / total,
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
    }
