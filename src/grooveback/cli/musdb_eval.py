"""Score restorations of MP3 twins of the MUSDB18-HQ test mixtures.

    uv run python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps 24 --ae earvae

Each test mixture is cut into 6 s clips, each clip is MP3-compressed,
and Apollo restores the MP3. The MP3, Apollo's output and, with `--ae`, an
autoencoder's round trip of the MP3 and of Apollo's output are scored against
the clean clip with the paper's SDR and SI-SNR, and with spectral SNR and
log-spectral distance (ADR-0007). Every step follows the authors' released
code; each function names its source. ADR-0015 records how we checked each
step against their code. The released checkpoint was not trained
on these songs (JusperLee/Apollo#46).

Models with their own environment restore the MP3s outside this process
(`musdb_eval` is short for `uv run python -m grooveback.cli.musdb_eval`):

    musdb_eval musdb18hq/test mp3.csv --kbps 24 --no-apollo --clips 100 --mp3-dir mp3
    third_party/audiosr/.venv/bin/python scripts/audiosr_restore.py mp3 audiosr
    musdb_eval musdb18hq/test scores.csv --kbps 24 --no-apollo --clips 100 --render audiosr=audiosr
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from grooveback import latents as gl
from grooveback.baselines import load_apollo, run_apollo
from grooveback.evaluation import (
    bss_sdr_db,
    bss_si_sdr_db,
    log_spectral_distance_db,
    spectral_snr_db,
)
from grooveback.solvers import roundtrip

SR = 44_100
CLIP_SAMPLES = 6 * SR


def vad_starts(audio: np.ndarray, win: int = 441) -> list[int]:
    """Starts of the 6 s clips Apollo's preprocessing keeps from a track.

    Source: `VAD` in Apollo-data-preprocess's musdb_preprocess.py (bb37a7b),
    rewritten with array operations; it keeps the same clips. Over 10 ms
    frames, a clip is kept when more than half its frames are louder than
    the 15th percentile of the track's non-silent frames.
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


def codec_simu(clip: np.ndarray, kbps: int) -> np.ndarray:
    """MP3 round trip of `clip` at `kbps`, as Apollo's training pairs are made.

    Source: `codec_simu` and `match2` in Apollo's
    look2hear/datas/musdb_moisesdb_datamodule.py, copied with the random bitrate
    replaced by `kbps`: the sox round trip (`sox_round_trip`), then `align`.
    """
    return align(sox_round_trip(clip, kbps), clean=clip)


def sox_round_trip(clip: np.ndarray, kbps: int) -> np.ndarray:
    """`clip` through torchaudio's sox MP3 at `kbps` and back to `clip`'s length,
    the shortfall filled from `clip`. Not aligned: LAME's delay leaves the MP3
    1,105 samples late."""
    import torchaudio

    wav = torch.from_numpy(clip)
    coded = torchaudio.functional.apply_codec(wav, SR, format="mp3", compression=kbps)
    if coded.shape[-1] >= wav.shape[-1]:
        coded = coded[..., : wav.shape[-1]]
    else:
        coded = torch.cat([coded, wav[..., coded.shape[-1] :]], -1)
    return coded.numpy()


def align(coded: np.ndarray, *, clean: np.ndarray) -> np.ndarray:
    """`coded` rolled by the lag that the phase-transform cross-correlation
    (GCC-PHAT) of `coded` and `clean` finds on the first channel. Source: `match2`
    in Apollo's look2hear/datas/musdb_moisesdb_datamodule.py.

    One change: both signals are tapered with a Hann window before the
    correlation. Untapered, the jump from a clip's last sample back to its
    first sits at lag 0 in both the clean clip and the MP3. The phase transform
    weights that jump like the music, and above the MP3's cut-off it is all the
    MP3 holds, so on low-bitrate clips with loud edges `match2` returned lag 0
    and left the MP3 1,105 samples late (2.5% of the MUSDB18-HQ test clips at
    24 kbps).
    """
    coded, wav = torch.from_numpy(coded), torch.from_numpy(clean)
    window = torch.hann_window(wav.shape[-1], periodic=False)
    phi = torch.fft.rfft(coded * window, dim=-1) * torch.fft.rfft(wav * window, dim=-1).conj()
    phi = phi / (phi.abs() + 1e-3)
    phi[:, 0] = 0
    lag = int(torch.argmax(torch.fft.irfft(phi, dim=-1).abs(), dim=-1)[0])
    return torch.roll(coded, -lag, -1).numpy()


def mixtures(musdb_test: Path) -> list[Path]:
    return sorted(
        path for path in musdb_test.glob("*/mixture.*") if path.suffix in (".wav", ".flac")
    )


def clip_starts(musdb_test: Path, *, clips: int | None = None) -> list[tuple[Path, int]]:
    """`(mixture, start)` of every clip `vad_starts` keeps or, given `clips`,
    the first `clips` of one fixed random order. The draw does not depend on
    the bitrate, so every bitrate scores the same clips, and a smaller draw
    is part of every larger one."""
    found = [
        (path, start)
        for path in mixtures(musdb_test)
        for start in vad_starts(sf.read(path, always_2d=True)[0].T)
    ]
    if clips is None:
        return found
    drawn = np.random.default_rng(0).permutation(len(found))[:clips]
    return [found[i] for i in sorted(drawn)]


def evaluate(
    musdb_test: Path,
    *,
    kbps: int,
    model=None,
    autoencoders: dict | None = None,
    renders: dict[str, Path] | None = None,
    clips: int | None = None,
    mp3_dir: Path | None = None,
) -> list[dict]:
    """Per clip, SDR, SI-SNR, spectral SNR and LSD of the untouched MP3
    (`input`) and of each restoration of it:

    - `apollo`, given the model;
    - `{name}` for each autoencoder's round trip of the MP3 and, given the
      model, `apollo_{name}` for its round trip of Apollo's output (ADR-0011);
    - `{name}` for each directory of renders, `{track}/{start}.wav`, made
      outside this process from the MP3s that `mp3_dir` receives.

    Sources in Apollo's repository: both signals are divided by their joint
    peak as in `MusdbMoisesdbDataset.__getitem__`
    (look2hear/datas/musdb_moisesdb_datamodule.py), the model restores each
    whole clip in one pass as in test.py, and SDR and SI-SNR are those of
    look2hear/metrics/wrapper.py.
    """
    autoencoders, renders = autoencoders or {}, renders or {}
    rows = []
    for path, picked in itertools.groupby(clip_starts(musdb_test, clips=clips), key=lambda s: s[0]):
        audio = sf.read(path, always_2d=True)[0].T  # float64, as their preprocessing reads it
        starts = [start for _, start in picked]
        for start in starts:
            clean = audio[:, start : start + CLIP_SAMPLES].astype(np.float32)
            codec = codec_simu(clean, kbps)
            peak = max(np.abs(clean).max(), np.abs(codec).max())
            clean, codec = clean / peak, codec / peak
            clip_path = Path(path.parent.name) / f"{start}.wav"
            if mp3_dir is not None:
                (mp3_dir / clip_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(mp3_dir / clip_path, codec.T, SR, subtype="FLOAT")
            outputs = {"input": codec}
            if model is not None:
                outputs["apollo"] = run_apollo(codec, SR, model=model, chunk_seconds=None)
            for ae, handle in autoencoders.items():
                outputs[ae] = roundtrip(handle, codec, SR, ae=ae)
                if model is not None:
                    outputs[f"apollo_{ae}"] = roundtrip(handle, outputs["apollo"], SR, ae=ae)
            for name, render_dir in renders.items():
                render = sf.read(render_dir / clip_path, dtype="float32", always_2d=True)[0]
                outputs[name] = render.T
            row = {"track": path.parent.name, "start": start}
            for name, output in outputs.items():
                row |= {
                    f"{name}_sdr": bss_sdr_db(clean, output),
                    f"{name}_si_snr": bss_si_sdr_db(clean, output),
                    f"{name}_spectral_snr": spectral_snr_db(clean, output),
                    f"{name}_lsd": log_spectral_distance_db(clean, output),
                }
            rows.append(row)
        print(f"{path.parent.name}: {len(starts)} clips", flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("musdb_test", type=Path, help="{track}/mixture.wav or .flac")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--kbps", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-apollo", action="store_true")
    parser.add_argument(
        "--ae",
        action="append",
        default=[],
        choices=sorted(gl.AUTOENCODERS),
        help="also score this autoencoder's round trip of the MP3 and of Apollo's output",
    )
    parser.add_argument(
        "--render",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="also score the renders {track}/{start}.wav in DIR, made from --mp3-dir's MP3s",
    )
    parser.add_argument(
        "--clips", type=int, help="score this many clips drawn at random; default: all"
    )
    parser.add_argument("--mp3-dir", type=Path, help="write each MP3 as {track}/{start}.wav")
    args = parser.parse_args(argv)
    if not mixtures(args.musdb_test):
        parser.error(f"no {{track}}/mixture.wav or .flac under {args.musdb_test}")

    rows = evaluate(
        args.musdb_test,
        kbps=args.kbps,
        model=None if args.no_apollo else load_apollo(device=args.device),
        autoencoders={ae: gl.load_ae(ae, device=args.device) for ae in args.ae},
        renders={name: Path(path) for name, path in (r.split("=", 1) for r in args.render)},
        clips=args.clips,
        mp3_dir=args.mp3_dir,
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in [column.removesuffix("_sdr") for column in rows[0] if column.endswith("_sdr")]:
        means = ", ".join(
            f"{metric} {np.mean([row[f'{name}_{metric}'] for row in rows]):.2f} dB"
            for metric in ("sdr", "si_snr", "spectral_snr", "lsd")
        )
        print(f"{name}: {means} over {len(rows)} clips")


if __name__ == "__main__":
    main()
