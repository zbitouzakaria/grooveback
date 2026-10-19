"""Apollo's test protocol, rebuilt on the MUSDB18-HQ test split.

    uv run python -m grooveback.cli.musdb_eval build musdb18hq/test testset/24k --kbps 24
    uv run python -m grooveback.cli.musdb_eval score testset/24k scores.csv

The paper scores 6 s clips from MUSDB18-HQ and MoisesDB in four categories,
MP3-compressed at 24-128 kbps; that set was never released. `build` rebuilds
it on MUSDB18-HQ alone from the authors' released code:

- sessions: the 6 s windows, at a 3 s hop, that Apollo-data-preprocess's VAD
  keeps from each stem
- clips: per category, stems from random tracks and sessions, each gained by
  a whole number of dB in [-10, 10] and summed — the training mixer in
  look2hear/datas
- damage: their `codec_simu` — torchaudio's sox MP3 there and back, cut to
  length, rolled into alignment by their `match2`
- both rescaled by their joint peak

and writes their test layout, `{category}/{index}/ori_wav.wav` and
`codec_wav.wav`. `score` runs Apollo once over each whole clip, as their
test.py does, and scores it and the untouched input with their SDR and SI-SNR.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from grooveback import audio as ga
from grooveback.baselines import load_apollo, run_apollo
from grooveback.evaluation import bss_sdr_db, bss_si_sdr_db

SR = 44_100
CLIP_SAMPLES = 6 * SR
INSTRUMENTS = ("bass", "drums", "other")
CATEGORIES = ("vocal", "single-stem", "multi-stems", "multi-stems+vocal")


def vad_starts(audio: np.ndarray, win: int = 441) -> list[int]:
    """Starts of the 6 s sessions Apollo's preprocessing keeps from a stem.

    musdb_preprocess.VAD: over 10 ms frames, a session is kept when more than
    half its frames are louder than the 15th percentile of the stem's
    non-silent frames.
    """
    n = audio.shape[-1] // SR * SR
    power = (audio[:, :n] ** 2).reshape(audio.shape[0], -1, win).sum(axis=(0, 2))
    active = power[power > 1e-3]
    if not active.size:
        return []
    threshold = np.quantile(active, 0.15)
    size, hop = CLIP_SAMPLES // win, CLIP_SAMPLES // 2 // win
    count = (n - CLIP_SAMPLES) // (CLIP_SAMPLES // 2)
    return [
        s * hop * win
        for s in range(count)
        if (power[s * hop : s * hop + size] > threshold).mean() > 0.5
    ]


def pick_stems(category: str, rng: np.random.Generator) -> list[str]:
    if category == "vocal":
        return ["vocals"]
    if category == "single-stem":
        return [str(rng.choice(INSTRUMENTS))]
    count = rng.integers(2, 4) if category == "multi-stems" else rng.integers(1, 4)
    stems = [str(stem) for stem in rng.choice(INSTRUMENTS, size=count, replace=False)]
    return stems + ["vocals"] if category == "multi-stems+vocal" else stems


def codec_simu(clip: np.ndarray, kbps: int) -> np.ndarray:
    """look2hear's codec_simu: sox MP3 at `kbps` and back to `clip`'s length,
    the shortfall filled from the clean clip, then rolled by the lag their
    phase-transform cross-correlation (`match2`) finds on the first channel."""
    import torchaudio

    wav = torch.from_numpy(clip)
    coded = torchaudio.functional.apply_codec(wav, SR, format="mp3", compression=kbps)
    if coded.shape[-1] >= wav.shape[-1]:
        coded = coded[..., : wav.shape[-1]]
    else:
        coded = torch.cat([coded, wav[..., coded.shape[-1] :]], -1)
    phi = torch.fft.rfft(coded, dim=-1) * torch.fft.rfft(wav, dim=-1).conj()
    phi = phi / (phi.abs() + 1e-3)
    phi[:, 0] = 0
    lag = int(torch.argmax(torch.fft.irfft(phi, dim=-1).abs(), dim=-1)[0])
    return torch.roll(coded, -lag, -1).numpy()


def build(
    musdb_test: Path, out: Path, *, kbps: int, per_category: int, seed: int = 0
) -> None:
    """Write `per_category` clean/MP3 pairs per category under `out`."""
    stem_paths = {
        (path.parent.name, path.stem): path for path in sorted(musdb_test.glob("*/*.flac"))
    }
    sessions = {
        key: vad_starts(sf.read(path, always_2d=True)[0].T)
        for key, path in stem_paths.items()
    }
    rng = np.random.default_rng(seed)
    for category in CATEGORIES:
        for index in range(per_category):
            ori = np.zeros((2, CLIP_SAMPLES), np.float32)
            for stem in pick_stems(category, rng):
                keys = [key for key in sessions if key[1] == stem and sessions[key]]
                key = keys[rng.integers(len(keys))]
                start = int(rng.choice(sessions[key]))
                audio = sf.read(stem_paths[key], start=start, frames=CLIP_SAMPLES,
                                dtype="float32", always_2d=True)[0].T
                ori += audio * float(np.sqrt(10 ** (rng.integers(-10, 11) / 10)))
            codec = codec_simu(ori, kbps)
            peak = max(np.abs(ori).max(), np.abs(codec).max())
            pair = out / category / f"{index:04d}"
            ga.save(pair / "ori_wav.wav", ori / peak, SR)
            ga.save(pair / "codec_wav.wav", codec / peak, SR)
        print(f"built {per_category} {category} clips", flush=True)


def score(test_set: Path, model) -> list[dict]:
    """SDR and SI-SNR of Apollo's render and of the untouched input, per clip."""
    rows = []
    for ori_path in sorted(test_set.glob("*/*/ori_wav.wav")):
        ori, _ = ga.load(ori_path)
        codec, _ = ga.load(ori_path.with_name("codec_wav.wav"))
        restored = run_apollo(codec, SR, model=model, chunk_seconds=None)
        rows.append({
            "category": ori_path.parent.parent.name,
            "clip": ori_path.parent.name,
            "input_sdr": bss_sdr_db(ori, codec),
            "input_si_snr": bss_si_sdr_db(ori, codec),
            "apollo_sdr": bss_sdr_db(ori, restored),
            "apollo_si_snr": bss_si_sdr_db(ori, restored),
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build_args = commands.add_parser("build")
    build_args.add_argument("musdb_test", type=Path, help="{track}/{stem}.flac")
    build_args.add_argument("out", type=Path)
    build_args.add_argument("--kbps", type=int, default=24)
    build_args.add_argument("--per-category", type=int, default=250)
    build_args.add_argument("--seed", type=int, default=0)
    score_args = commands.add_parser("score")
    score_args.add_argument("test_set", type=Path)
    score_args.add_argument("csv", type=Path)
    score_args.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if args.command == "build":
        build(args.musdb_test, args.out, kbps=args.kbps,
              per_category=args.per_category, seed=args.seed)
        return

    rows = score(args.test_set, load_apollo(device=args.device))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ["input_sdr", "input_si_snr", "apollo_sdr", "apollo_si_snr"]
    print(f"{'':20}" + "".join(f"{m:>15}" for m in metrics))
    for category in [*CATEGORIES, "overall"]:
        picked = rows if category == "overall" else [r for r in rows if r["category"] == category]
        if picked:
            means = [np.mean([r[m] for r in picked]) for m in metrics]
            print(f"{category:20}" + "".join(f"{v:15.2f}" for v in means))


if __name__ == "__main__":
    main()
