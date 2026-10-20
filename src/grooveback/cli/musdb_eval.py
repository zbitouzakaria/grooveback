"""Score Apollo on MP3 twins of the MUSDB18-HQ test mixtures.

    uv run python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps 24

Every 6 s session of each test mixture that Apollo-data-preprocess's VAD
keeps is MP3-compressed with Apollo's own `codec_simu`, both signals are
rescaled by their joint peak as in their data module, and Apollo restores the
MP3 in one pass, as their test.py does. Apollo's output and the untouched MP3
are scored against the clean session with the paper's SDR and SI-SNR
(look2hear/metrics/wrapper.py). The released checkpoint was not trained on
these songs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from grooveback.baselines import load_apollo, run_apollo
from grooveback.evaluation import bss_sdr_db, bss_si_sdr_db

SR = 44_100
CLIP_SAMPLES = 6 * SR


def vad_starts(audio: np.ndarray, win: int = 441) -> list[int]:
    """Starts of the 6 s sessions Apollo's preprocessing keeps from a track.

    musdb_preprocess.VAD: over 10 ms frames, a session is kept when more than
    half its frames are louder than the 15th percentile of the track's
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


def mixtures(musdb_test: Path) -> list[Path]:
    return sorted(
        path for path in musdb_test.glob("*/mixture.*") if path.suffix in (".wav", ".flac")
    )


def evaluate(musdb_test: Path, *, kbps: int, model) -> list[dict]:
    """SDR and SI-SNR of the model's restoration and of the untouched MP3, per session."""
    rows = []
    for path in mixtures(musdb_test):
        audio = sf.read(path, always_2d=True)[0].T  # float64, as their preprocessing reads it
        starts = vad_starts(audio)
        for start in starts:
            clean = audio[:, start : start + CLIP_SAMPLES].astype(np.float32)
            codec = codec_simu(clean, kbps)
            peak = max(np.abs(clean).max(), np.abs(codec).max())
            clean, codec = clean / peak, codec / peak
            restored = run_apollo(codec, SR, model=model, chunk_seconds=None)
            rows.append({
                "track": path.parent.name,
                "start": start,
                "input_sdr": bss_sdr_db(clean, codec),
                "input_si_snr": bss_si_sdr_db(clean, codec),
                "apollo_sdr": bss_sdr_db(clean, restored),
                "apollo_si_snr": bss_si_sdr_db(clean, restored),
            })
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
    args = parser.parse_args(argv)
    if not mixtures(args.musdb_test):
        parser.error(f"no {{track}}/mixture.wav or .flac under {args.musdb_test}")

    rows = evaluate(args.musdb_test, kbps=args.kbps, model=load_apollo(device=args.device))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ("input", "apollo"):
        sdr = np.mean([row[f"{name}_sdr"] for row in rows])
        si_snr = np.mean([row[f"{name}_si_snr"] for row in rows])
        print(f"{name}: SDR {sdr:.2f} dB, SI-SNR {si_snr:.2f} dB over {len(rows)} clips")


if __name__ == "__main__":
    main()
