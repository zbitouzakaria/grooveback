"""Restore every WAV under a directory with AudioSR (Liu et al., arXiv:2309.07314).

    third_party/audiosr/.venv/bin/python scripts/audiosr_restore.py mp3/ audiosr/

Runs in the environment that scripts/audiosr_setup.sh builds: the package pins
torch 2.0.1, which conflicts with this project's torch, so this file imports
nothing from grooveback. The model loads once for the whole directory, and a
file whose render already exists is skipped.

The model is mono and 48 kHz: each file is resampled to 48 kHz, each channel is
restored on its own, and the result is resampled back to the file's rate and
cut to its length. The package pads each input to a multiple of 5.12 s and
restores it in one pass, so a 6 s clip costs as much as 10.24 s.

Sampling uses 50 DDIM steps, the default of AudioSR's command-line tool (its
Python function defaults to 200), and the package's own seed and guidance scale.

The package scales each input to a peak of 0.5 before the model. Below the
cut-off it detects, the render is the input's own spectrum, so the
least-squares gain of the render against its input brings back the input's
level.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

NATIVE_RATE = 48_000


def restore_channel(channel: np.ndarray, *, sr_fn, tmp_dir: Path) -> np.ndarray:
    """Restore one mono 48 kHz channel; `sr_fn(wav_path)` runs the model on a file."""
    wav = tmp_dir / "channel.wav"
    sf.write(wav, channel, NATIVE_RATE, subtype="FLOAT")
    render = np.asarray(sr_fn(str(wav)), dtype=np.float64).squeeze()[: channel.size]
    reference = channel.astype(np.float64)
    projection = np.dot(render, reference)
    gain = np.dot(reference, reference) / projection if projection > 0 else 0.0
    return (gain * render).astype(np.float32)


def restore_file(audio: np.ndarray, rate: int, *, sr_fn, tmp_dir: Path) -> np.ndarray:
    """`(channels, samples)` at `rate` in, the same shape and rate out."""
    native = soxr.resample(audio.T, rate, NATIVE_RATE, quality="VHQ").T
    restored = np.stack(
        [restore_channel(channel, sr_fn=sr_fn, tmp_dir=tmp_dir) for channel in native]
    )
    out = soxr.resample(restored.T, NATIVE_RATE, rate, quality="VHQ").T[:, : audio.shape[1]]
    return np.pad(out, ((0, 0), (0, audio.shape[1] - out.shape[1])))


def main(argv: list[str] | None = None, *, sr_fn=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    todo = [
        path
        for path in sorted(args.input_dir.rglob("*.wav"))
        if not (args.output_dir / path.relative_to(args.input_dir)).exists()
    ]
    if sr_fn is None:
        from audiosr import build_model, super_resolution

        model = build_model(model_name="basic", device=args.device)

        def sr_fn(wav_path):
            return super_resolution(model, wav_path, ddim_steps=args.steps)

    with tempfile.TemporaryDirectory() as tmp:
        for i, path in enumerate(todo):
            began = time.perf_counter()
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            out = restore_file(audio.T, rate, sr_fn=sr_fn, tmp_dir=Path(tmp))
            target = args.output_dir / path.relative_to(args.input_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, out.T, rate, subtype="FLOAT")
            seconds = time.perf_counter() - began
            print(f"restore: {i + 1}/{len(todo)} {target} in {seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main()
