"""Make the LoRA's validation pairs, MP3 and clean, from a folder of clean tracks.

    uv run python scripts/prepare_dataset.py data/lora/wav/valid data/lora/valid

For every `{song}.wav` in the folder it writes `{song}/original.wav` and one
MP3 version per bitrate, `{song}/{kbps}k.wav`, decoded back to float WAV and
sample-aligned with the original. Float WAV keeps MP3 overshoot past full
scale intact. The MP3s come from `codec_simu`, the same codec simulation
`musdb_eval` scores with, and the one training codes its windows with, so
validation damage matches training and evaluation damage. Training itself
reads clean tracks and codes them as it goes (`grooveback.dataset`). It needs
sox (`apt-get install -y sox libsox-dev` on a pod).

With `--chains 6` it writes six coded versions instead, `{song}/1k.wav` to
`{song}/6k.wav`, read with `bitrates=[1, 2, 3, 4, 5, 6]` (ADR-0017). Each is
the whole song through its own chain of one or two codecs, drawn as training
draws them, from configs/train_lora.yaml's `augmentations`. Every chain is
drawn first, under a fixed seed, and recorded in `chains.json`, so a
restarted run codes the same ones. It also needs ffmpeg.

A file that already exists is kept, so an interrupted run can be restarted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from grooveback import audio as ga
from grooveback import dataset as gd
from grooveback.cli import musdb_eval
from grooveback.cli.musdb_eval import SR, codec_simu
from grooveback.evaluation import best_lag

BITRATES = (24, 32, 48, 64, 96, 128)
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_lora.yaml"


def prepare(wav_dir: Path, out_dir: Path, *, bitrates: tuple[int, ...] = BITRATES) -> None:
    tracks = sorted(wav_dir.glob("*.wav"))
    if not tracks:
        raise FileNotFoundError(f"No .wav files in {wav_dir}.")
    for track in tracks:
        song = out_dir / track.stem
        audio, sample_rate = ga.load(track)
        audio = ga.resample(audio, sr_in=sample_rate, sr_out=SR)
        if not (song / "original.wav").exists():
            ga.save(song / "original.wav", audio, SR)
        for kbps in bitrates:
            target = song / f"{kbps}k.wav"
            if target.exists():
                continue
            mp3 = codec_simu(audio, kbps)
            # One sample of shift would teach the model a delay instead of the damage.
            lag = best_lag(audio, mp3)
            if lag != 0:
                raise RuntimeError(
                    f"{track.name} at {kbps} kbps is {lag} samples off the original."
                )
            ga.save(target, mp3, SR)
        print(f"{track.stem}: original + {len(bitrates)} MP3s", flush=True)


def prepare_chains(wav_dir: Path, out_dir: Path, *, versions: int, seed: int = 0) -> None:
    tracks = sorted(wav_dir.glob("*.wav"))
    if not tracks:
        raise FileNotFoundError(f"No .wav files in {wav_dir}.")
    augmentations = gd.Augmentations(**OmegaConf.to_container(OmegaConf.load(CONFIG).augmentations))
    # Every chain is drawn before any is coded, so a restart draws the same ones.
    torch.manual_seed(seed)
    chains = {
        track.stem: [gd.draw_codecs(augmentations) for _ in range(versions)] for track in tracks
    }
    record = out_dir / "chains.json"
    if record.exists() and json.loads(record.read_text()) != json.loads(json.dumps(chains)):
        raise ValueError(f"{record} holds other chains; delete {out_dir} to draw again.")
    out_dir.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(chains, indent=1))
    for track in tracks:
        song = out_dir / track.stem
        audio, sample_rate = ga.load(track)
        audio = ga.resample(audio, sr_in=sample_rate, sr_out=SR)
        if not (song / "original.wav").exists():
            ga.save(song / "original.wav", audio, SR)
        for number, chain in enumerate(chains[track.stem], start=1):
            target = song / f"{number}k.wav"
            if target.exists():
                continue
            coded = audio
            for codec, kbps in chain:
                coded = gd.round_trip(coded, codec=codec, kbps=kbps)
            coded = musdb_eval.align(coded, clean=audio)
            lag = best_lag(audio, coded)
            if lag != 0:
                raise RuntimeError(
                    f"{track.name} through {chain} is {lag} samples off the original."
                )
            ga.save(target, coded, SR)
        print(f"{track.stem}: original + {versions} chains", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wav_dir", type=Path, help="folder of clean {song}.wav files")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--kbps", type=int, nargs="+", default=list(BITRATES))
    parser.add_argument("--chains", type=int, help="this many random codec chains per song")
    args = parser.parse_args(argv)
    if args.chains:
        prepare_chains(args.wav_dir, args.out_dir, versions=args.chains)
    else:
        prepare(args.wav_dir, args.out_dir, bitrates=tuple(args.kbps))


if __name__ == "__main__":
    main()
