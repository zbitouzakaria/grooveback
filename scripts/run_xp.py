"""The whole benchmark, one entry point.

  uv run python scripts/run_xp.py [device]

Cut one chunk per source, make MP3 twins at three bitrates, restore each twin
with every method, and score everything against the original chunk:

  artifacts/xp/{source}/original.wav             the clean chunk
  artifacts/xp/{source}/{bitrate}/input.wav      the MP3 round-trip
  artifacts/xp/{source}/{bitrate}/{method}.wav   one render per method
  artifacts/xp/{source}/{bitrate}/listen/        level-matched copies to A/B
  artifacts/xp/results.json                      SDR and SI-SNR per render

A render is skipped when its file exists, so re-running is safe and renders
produced on a GPU pod are picked up as-is. Scoring always re-runs, over
whatever renders exist.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from grooveback import audio as ga
from grooveback import latents as gl
from grooveback.baselines import load_apollo, run_a2sb, run_apollo
from grooveback.evaluation import best_lag, sdr_db, si_snr_db, write_listening_pack

SR = 44_100
SOURCES = {
    "aerofunk": ("data/Aerofunk - Nice One (Cpu Cant Hack It Mix) 258.wav", 60.0, 12.0),
    "codec": ("data/original_codec_wav.wav", 0.0, 6.0),
}
"""name -> (path, start_s, duration_s). One chunk per source, one fixed rule."""

BITRATES = ("64k", "128k", "192k")
METHODS = ("same-s", "same-l", "apollo", "a2sb")
XP = Path("artifacts/xp")


def render_path(name: str, bitrate: str, method: str) -> Path:
    return XP / name / bitrate / f"{method}.wav"


def cached(wav: Path) -> np.ndarray | None:
    """An existing render, or None if it has to be made."""
    return ga.load(wav)[0] if wav.exists() else None


def cut_chunk(name: str) -> np.ndarray:
    wav = XP / name / "original.wav"
    if wav.exists():
        return ga.load(wav)[0]
    path, start_s, duration_s = SOURCES[name]
    source, rate = ga.load(path)
    if rate != SR:
        raise ValueError(f"{path} is {rate} Hz, the benchmark runs at {SR} Hz.")
    start = int(start_s * SR)
    chunk = source[:, start : start + int(duration_s * SR)]
    ga.save(wav, chunk, SR)
    return chunk


def build_twin(name: str, bitrate: str, original: np.ndarray) -> np.ndarray:
    """MP3-compress the chunk and read it back, verified sample-aligned."""
    wav = XP / name / bitrate / "input.wav"
    if not wav.exists():
        mp3 = wav.with_suffix(".mp3")
        mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(XP / name / "original.wav"),
                        "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        "-c:a", "pcm_f32le", str(wav)], check=True)
    degraded = ga.load(wav)[0][:, : original.shape[1]]
    # One sample of shift would wreck the waveform metrics, so refuse to score
    # a twin that is not exactly aligned. ffmpeg's decoder honours the LAME
    # header, so in practice the round-trip lands at lag 0.
    lag = best_lag(original[:, : degraded.shape[1]], degraded)
    if lag != 0:
        raise RuntimeError(f"{wav} is {lag} samples off its original.")
    return degraded


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "auto"

    chunks = {name: cut_chunk(name) for name in SOURCES}
    twins = {(name, bitrate): build_twin(name, bitrate, chunks[name])
             for name in SOURCES for bitrate in BITRATES}

    # Render whatever is missing, loading each model at most once.
    for variant in ("same-s", "same-l"):
        todo = [key for key in twins if not render_path(*key, variant).exists()]
        if todo:
            model = gl.load_same(variant, device=device)
            for name, bitrate in todo:
                print(f"render {variant}: {name} {bitrate}", flush=True)
                out = gl.roundtrip(twins[(name, bitrate)], SR, model=model)
                ga.save(render_path(name, bitrate, variant),
                        out[:, : chunks[name].shape[1]], SR)
            del model

    todo = [key for key in twins if not render_path(*key, "apollo").exists()]
    if todo:
        model = load_apollo(device=device)
        for name, bitrate in todo:
            print(f"render apollo: {name} {bitrate}", flush=True)
            out = run_apollo(twins[(name, bitrate)], SR, model=model)
            ga.save(render_path(name, bitrate, "apollo"), out, SR)
        del model

    for name, bitrate in twins:
        if not render_path(name, bitrate, "a2sb").exists():
            print(f"render a2sb: {name} {bitrate}", flush=True)
            # 50 steps is the paper's default; ADR-0006 uses it as canonical.
            out = run_a2sb(twins[(name, bitrate)], SR, n_steps=50,
                           device="mps" if device == "auto" else device)
            ga.save(render_path(name, bitrate, "a2sb"), out, SR)

    # Score whatever exists against the original, and write listening sets.
    results: dict = {}
    for name in SOURCES:
        results[name] = {}
        for bitrate in BITRATES:
            pack = {"original": chunks[name], "input": twins[(name, bitrate)]}
            for method in METHODS:
                render = cached(render_path(name, bitrate, method))
                if render is not None:
                    pack[method] = render
            shortest = min(item.shape[1] for item in pack.values())
            pack = {label: item[:, :shortest] for label, item in pack.items()}
            results[name][bitrate] = {
                label: {"sdr_db": round(sdr_db(pack["original"], item), 2),
                        "si_snr_db": round(si_snr_db(pack["original"], item), 2)}
                for label, item in pack.items() if label != "original"
            }
            write_listening_pack(pack, SR, XP / name / bitrate / "listen")
            print(name, bitrate, results[name][bitrate], flush=True)

    (XP / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {XP / 'results.json'}")


if __name__ == "__main__":
    main()
