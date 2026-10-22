"""Run AudioSR (Liu et al., arXiv:2309.07314) on one audio file.

    third_party/audiosr/.venv/bin/python scripts/audiosr_restore.py in.wav out.wav

Runs inside the audiosr venv (built by scripts/audiosr_setup.sh): the package
pins torch 2.0.1, which conflicts with the project's stack, so this file
imports only what both environments have and loads audiosr lazily.

Contract: any sample rate in, the input's own rate, exact length and channel
count out. The model is natively 48 kHz mono, so the file is resampled to
48 kHz once (their internal resampler then never fires), each channel is
restored separately, and the render is resampled back. A genuinely low-rate
input loses its invented top band to that return resample; upsample such a
file first if the band is wanted.

Sampling flags left unset are not passed to the model, so the pinned
package's own defaults apply (seed 42, 200 ddim steps, guidance 3.5 in
0.0.7). Chunking is ours — the pinned release has no long-audio path: chunks
are cut on input boundaries, each render is trimmed to its chunk before the
model's padding can accumulate, and consecutive chunks are joined by a short
linear crossfade. The fade is equal-gain, not equal-power, so identical
content passes through exactly (the identity gate in
.claude/rules/audio.md); its cost on independently invented content is
measured, not assumed (ADR-0012).

The package discards level in both directions — input and output are each
peak-normalized to 0.5 — so each chunk's render is scaled back to its
chunk's level by a least-squares gain fit before joining (see
restore_channel).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

NATIVE_RATE = 48_000
GRAIN_SECONDS = 5.12
"""The model pads its input up to a multiple of this (its latent grain)."""

FADE_SECONDS = 0.1


def chunk_spans(
    n_samples: int, chunk_samples: int, fade_samples: int
) -> list[tuple[int, int]]:
    """`[(start, end)]` covering the input, consecutive spans overlapping by
    `fade_samples`. The last span absorbs the remainder, so it is at most one
    chunk long and always longer than the fade."""
    if fade_samples >= chunk_samples:
        raise ValueError(
            f"fade ({fade_samples}) must be shorter than a chunk ({chunk_samples})."
        )
    spans = []
    start = 0
    while start + chunk_samples < n_samples:
        spans.append((start, start + chunk_samples))
        start += chunk_samples - fade_samples
    spans.append((start, n_samples))
    return spans


def restore_channel(
    channel: np.ndarray,
    *,
    sr_fn,
    chunk_samples: int,
    fade_samples: int,
    tmp_dir: Path,
) -> np.ndarray:
    """Restore one mono 48 kHz channel chunk by chunk.

    `sr_fn(wav_path)` runs the model on one mono 48 kHz file and returns its
    waveform; the model pads to its 5.12 s grain, so each render is trimmed
    back to its chunk before joining.
    """
    spans = chunk_spans(channel.size, chunk_samples, fade_samples)
    out = np.zeros(channel.size, dtype=np.float32)
    for i, (start, end) in enumerate(spans):
        chunk = channel[start:end]
        wav = tmp_dir / f"chunk_{i}.wav"
        sf.write(str(wav), chunk, NATIVE_RATE, subtype="FLOAT")
        render = np.asarray(sr_fn(str(wav)), dtype=np.float32).squeeze()
        if render.ndim != 1 or render.size < end - start:
            raise RuntimeError(
                f"model returned shape {render.shape} for a {end - start}-sample chunk."
            )
        piece = render[: end - start].copy()
        # The package peak-normalizes each render to 0.5 whatever the input's
        # level, so joining raw renders would jump in gain chunk to chunk.
        # The render's kept band is the input's own content (the package
        # splices it back in), so a least-squares gain fit against the chunk
        # recovers the input's level exactly up to the invented band; it is
        # exactly 1 when the render equals the chunk, and a silent chunk maps
        # to silence.
        projection = float(np.dot(piece, chunk))
        piece *= float(np.dot(chunk, chunk)) / projection if projection > 0.0 else 0.0
        if start > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            out[start : start + fade_samples] *= 1.0 - fade_in
            piece[:fade_samples] *= fade_in
        out[start:end] += piece
    return out


def main(argv: list[str] | None = None, *, sr_fn=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--steps", type=int, default=None, help="ddim steps; unset: the package default"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=None, help="unset: the package default"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="unset: the package default"
    )
    parser.add_argument("--model", default="basic", choices=["basic", "speech"])
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=2 * GRAIN_SECONDS,
        help="chunk length; the default is two of the model's 5.12 s grains",
    )
    parser.add_argument(
        "--device", default=None, help="unset: audiosr picks cuda, then mps, then cpu"
    )
    args = parser.parse_args(argv)

    audio, rate = sf.read(str(args.input), dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio.T)  # (channels, samples)
    if rate == NATIVE_RATE:
        native = audio
    else:
        native = soxr.resample(audio.T, rate, NATIVE_RATE, quality="VHQ").T
        native = np.ascontiguousarray(native.astype(np.float32))

    sampling = {
        key: value
        for key, value in {
            "ddim_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
        }.items()
        if value is not None
    }
    if sr_fn is None:
        from importlib.metadata import version

        from audiosr import build_model, super_resolution

        model = build_model(model_name=args.model, device=args.device)

        def sr_fn(wav_path):
            return super_resolution(model, wav_path, **sampling)

        knobs = " ".join(f"{k}={v}" for k, v in sampling.items()) or "package defaults"
        print(f"restore: audiosr {version('audiosr')} model={args.model} {knobs}",
              flush=True)

    chunk_samples = int(round(args.chunk_seconds * NATIVE_RATE))
    fade_samples = int(round(FADE_SECONDS * NATIVE_RATE))
    n_chunks = len(chunk_spans(native.shape[1], chunk_samples, fade_samples))
    print(f"restore: {n_chunks} chunks of {args.chunk_seconds:g} s per channel, "
          f"{FADE_SECONDS:g} s crossfade", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        restored = np.stack([
            restore_channel(
                channel,
                sr_fn=sr_fn,
                chunk_samples=chunk_samples,
                fade_samples=fade_samples,
                tmp_dir=Path(tmp),
            )
            for channel in native
        ])

    if rate != NATIVE_RATE:
        restored = soxr.resample(restored.T, NATIVE_RATE, rate, quality="VHQ").T
        restored = np.ascontiguousarray(restored.astype(np.float32))
    out = restored[:, : audio.shape[1]]
    if out.shape[1] < audio.shape[1]:
        out = np.pad(out, ((0, 0), (0, audio.shape[1] - out.shape[1])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), out.T, rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
