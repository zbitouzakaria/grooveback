"""Clean/MP3 twins for estimating the transport vector.

The clean master is encoded and decoded as a whole track, so the codec sees
continuous input the way it would on a real rip, and only then cut into
windows. ffmpeg handles LAME's encoder delay, so the twins line up sample for
sample — verified here rather than assumed.

Windows overlapping the held-out excerpts are excluded from the pool, so the
vector is never estimated on audio it is later applied to.

  data/twins/<source>_<bitrate>.wav         full decoded MP3
  data/twins/cut/pool###_<bitrate>/         clean.wav + degraded.wav
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from grooveback import audio as ga
from scripts.pick_excerpts import DURATION_S, OFFSETS_S  # noqa: E402

POOL_SOURCES = {"aerofunk": "data/Aerofunk - Nice One (Cpu Cant Hack It Mix) 258.wav"}
"""Clean, long enough to yield pool windows. AN-2 has no clean master; the
codec asset is 6 s. Both are application targets only."""

APPLY_SOURCES = {"codec": "data/original_codec_wav.wav"}
"""Clean but too short to pool — twinned so the vector can be applied to them."""

BITRATES = ("128k", "192k")
OUT = Path("data/twins")


def mp3_twin(source: str, name: str, bitrate: str) -> tuple[np.ndarray, int]:
    """Encode to MP3 and decode back, returning the aligned degraded audio."""
    wav = OUT / f"{name}_{bitrate}.wav"
    if not wav.exists():
        mp3 = OUT / f"{name}_{bitrate}.mp3"
        mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", source,
                        "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        "-c:a", "pcm_f32le", str(wav)], check=True)
    return ga.load(wav)


def alignment_lag(clean: np.ndarray, degraded: np.ndarray, sr: int,
                  max_lag: int = 4096) -> int:
    """Sample offset of `degraded` relative to `clean`, by cross-correlation."""
    centre, half = clean.shape[1] // 2, sr
    a = clean.mean(0)[centre - half : centre + half]
    b = degraded.mean(0)[centre - half - max_lag : centre + half + max_lag]
    return int(np.arange(-max_lag, max_lag + 1)[np.argmax(np.correlate(b, a, "valid"))])


def main() -> None:
    for name, source in {**POOL_SOURCES, **APPLY_SOURCES}.items():
        clean, sr = ga.load(source)
        pools = name in POOL_SOURCES
        for bitrate in BITRATES:
            degraded, dsr = mp3_twin(source, name, bitrate)
            assert dsr == sr, f"{name} {bitrate}: sample rate changed"

            lag = alignment_lag(clean, degraded, sr) if pools else 0
            if lag:
                degraded = np.roll(degraded, -lag, axis=1)
            print(f"{name} {bitrate}: lag={lag}")

            if not pools:
                ga.save(OUT / "cut" / f"apply_{name}_{bitrate}" / "clean.wav", clean, sr)
                ga.save(OUT / "cut" / f"apply_{name}_{bitrate}" / "degraded.wav",
                        degraded[:, : clean.shape[1]], sr)
                continue

            window = int(DURATION_S * sr)
            total = min(clean.shape[1], degraded.shape[1])
            held = 0
            for i in range(total // window):
                start_s = i * DURATION_S
                offset = next((o for o in OFFSETS_S
                               if start_s < o + DURATION_S and o < start_s + DURATION_S),
                              None)
                if offset is not None:
                    # A held-out excerpt: keep it, but as an application target
                    # rather than pool material.
                    held += 1
                    cut = OUT / "cut" / f"apply_{name}_{int(offset)}s_{bitrate}"
                else:
                    cut = OUT / "cut" / f"pool{i:03d}_{bitrate}"
                ga.save(cut / "clean.wav", clean[:, i * window : (i + 1) * window], sr)
                ga.save(cut / "degraded.wav",
                        degraded[:, i * window : (i + 1) * window], sr)
            print(f"  {total // window - held} pool windows, {held} held out")


if __name__ == "__main__":
    main()
