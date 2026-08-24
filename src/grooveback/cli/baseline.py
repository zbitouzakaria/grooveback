"""Run a baseline restoration method over a track.

    uv run python -m grooveback.cli.baseline data/track.mp3 artifacts/track.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from grooveback import audio as ga
from grooveback.baselines import run_a2sb, run_apollo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="apollo", choices=["apollo", "a2sb"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--chunk-seconds", type=float, default=12.0, help="apollo only")
    parser.add_argument("--overlap-seconds", type=float, default=1.0, help="apollo only")
    parser.add_argument("--chunk-pad-seconds", type=float, default=1.0, help="apollo only")
    parser.add_argument("--batch-size", type=int, default=1, help="apollo only")
    parser.add_argument("--steps", type=int, default=20, help="a2sb sampling steps")
    parser.add_argument(
        "--single-split",
        action="store_true",
        help="a2sb: 1-split checkpoint instead of the 2-split ensemble",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=None,
        help="a2sb: cutoff Hz; default: detected fork-side (pass it for excerpts)",
    )
    parser.add_argument(
        "--start", type=float, default=None, help="excerpt start in seconds"
    )
    parser.add_argument(
        "--seconds", type=float, default=None, help="excerpt length in seconds"
    )
    parser.add_argument(
        "--match-loudness",
        action="store_true",
        help=f"Normalize the output to {ga.TARGET_LUFS} LUFS before writing.",
    )
    args = parser.parse_args(argv)

    signal, sample_rate = ga.load(args.input)

    if args.start is not None or args.seconds is not None:
        a = int((args.start or 0.0) * sample_rate)
        b = a + int(args.seconds * sample_rate) if args.seconds else signal.shape[1]
        signal = signal[:, a:b]
    print(
        f"{args.input.name}: {sample_rate} Hz, {signal.shape[0]}ch, "
        f"{signal.shape[1] / sample_rate:.1f}s, "
        f"{ga.loudness(signal, sample_rate):.1f} LUFS"
    )

    started = time.perf_counter()
    if args.method == "apollo":
        restored = run_apollo(
            signal,
            sample_rate,
            device=args.device,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            chunk_pad_seconds=args.chunk_pad_seconds,
            batch_size=args.batch_size,
        )
    else:
        restored = run_a2sb(
            signal,
            sample_rate,
            n_steps=args.steps,
            cutoff_hz=args.cutoff_hz,
            single_split=args.single_split,
            device="mps" if args.device == "auto" else args.device,
        )
    elapsed = time.perf_counter() - started
    print(
        f"{args.method}: {elapsed:.1f}s "
        f"({signal.shape[1] / sample_rate / elapsed:.1f}x realtime)"
    )

    if args.match_loudness:
        restored = ga.normalize_loudness(restored, sample_rate)
    print(
        f"out: {ga.loudness(restored, sample_rate):.1f} LUFS, "
        f"peak {ga.peak_dbfs(restored):.1f} dBFS"
    )

    ga.save(args.output, restored, sample_rate)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
