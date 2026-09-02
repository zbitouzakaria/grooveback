"""Can Stable Audio 3 generate clean masters? One probe, one entry point.

  uv run python scripts/run_probe.py [device]

Every released model type generates the same track — one fixed prompt, fixed
seeds. Each clip is MP3-compressed and the compression is scored against the
clip itself with the benchmark's five metrics. A generated clip is its own
reference: if MP3 removes less from it than it removes from a real master at
the same bitrate, the clip was missing the content MP3 normally eats. The
master anchors are the degraded-input rows of artifacts/xp/results.json.

  artifacts/probe/{variant}/seed{n}.wav            12 s scoring clips
  artifacts/probe/{variant}/listen_45s.wav         one longer clip, for ears
  artifacts/probe/{variant}/{bitrate}/seed{n}.wav  the MP3 round-trips
  artifacts/probe/{variant}/listen/                level-matched A/B pack
  artifacts/probe/results.json                     per variant x bitrate

A step whose output exists is skipped. A variant that fails to load or render
is recorded in results.json and the run continues.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from grooveback import audio as ga
from grooveback import priors as gp
from grooveback.evaluation import (
    best_lag,
    bss_sdr_db,
    log_spectral_distance_db,
    sdr_db,
    si_snr_db,
    spectral_snr_db,
    write_listening_pack,
)

SR = gp.PRIOR_SAMPLE_RATE
PROMPT = ("house music. Like Caribou, bonobo, small Round basseline that pops "
          "on uk style speakers. clean elements and minimalist. "
          "No cheesy drops.")
SEEDS = (0, 1, 2)
CLIP_SECONDS = 12.0
LISTEN_SECONDS = 45.0
BITRATES = ("64k", "128k", "192k")
METRIC_KEYS = ("bss_sdr_db", "sdr_db", "si_snr_db", "spectral_snr_db", "lsd_db")
PROBE = Path("artifacts/probe")
XP_RESULTS = Path("artifacts/xp/results.json")


def mp3_roundtrip(wav_in: Path, bitrate: str, wav_out: Path) -> np.ndarray:
    """Compress to MP3 and back; refuse anything not exactly aligned."""
    if not wav_out.exists():
        mp3 = wav_out.with_suffix(".mp3")
        mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_in),
                        "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        "-c:a", "pcm_f32le", str(wav_out)], check=True)
    original = ga.load(wav_in)[0]
    degraded = ga.load(wav_out)[0][:, : original.shape[1]]
    lag = best_lag(original[:, : degraded.shape[1]], degraded)
    if lag != 0:
        # Dense noise-like or heavily periodic clips can fool the correlation
        # probe (measured: an sfx-base clip read lag -2605 while correlating
        # 0.9986 at zero). Alignment itself is what the gate protects, so
        # judge the match at lag zero: aligned round-trips sit above 0.98,
        # a real shift decorrelates far below 0.9.
        ref = original.mean(axis=0)[: degraded.shape[1]]
        est = degraded.mean(axis=0)[: ref.size]
        at_zero = float(np.dot(ref, est)
                        / (np.linalg.norm(ref) * np.linalg.norm(est) + 1e-12))
        if at_zero < 0.9:
            raise RuntimeError(f"{wav_out} is {lag} samples off its original.")
    return degraded


def render_variant(variant: str, device: str) -> None:
    """Generate the variant's clips, loading the model only if any is missing."""
    clips = {seed: PROBE / variant / f"seed{seed}.wav" for seed in SEEDS}
    listen = PROBE / variant / "listen_45s.wav"
    if all(wav.exists() for wav in clips.values()) and listen.exists():
        return
    model = gp.load_prior(variant, device=device)
    for seed, wav in clips.items():
        if not wav.exists():
            print(f"render {variant}: seed {seed}", flush=True)
            clip = gp.generate(model, PROMPT, seconds=CLIP_SECONDS, seed=seed,
                               **gp.PRIOR_VARIANTS[variant])
            ga.save(wav, clip, SR)
    if not listen.exists():
        print(f"render {variant}: {LISTEN_SECONDS:.0f} s listening clip",
              flush=True)
        clip = gp.generate(model, PROMPT, seconds=LISTEN_SECONDS, seed=SEEDS[0],
                           **gp.PRIOR_VARIANTS[variant])
        ga.save(listen, clip, SR)


def score_variant(variant: str, results: dict) -> None:
    """Codec-cost every existing clip of one variant and write its pack."""
    clips = [PROBE / variant / f"seed{seed}.wav" for seed in SEEDS]
    clips = [wav for wav in clips if wav.exists()]
    if not clips:
        return
    block = results["variants"].setdefault(variant, {})
    for bitrate in BITRATES:
        per_seed = []
        for wav in clips:
            clip = ga.load(wav)[0]
            twin = mp3_roundtrip(wav, bitrate, PROBE / variant / bitrate / wav.name)
            shortest = min(clip.shape[1], twin.shape[1])
            clip, twin = clip[:, :shortest], twin[:, :shortest]
            per_seed.append({
                "bss_sdr_db": round(bss_sdr_db(clip, twin), 2),
                "sdr_db": round(sdr_db(clip, twin), 2),
                "si_snr_db": round(si_snr_db(clip, twin), 2),
                "spectral_snr_db": round(spectral_snr_db(clip, twin), 2),
                "lsd_db": round(log_spectral_distance_db(clip, twin), 2),
            })
        block[bitrate] = {
            "mean": {key: round(float(np.mean([s[key] for s in per_seed])), 2)
                     for key in METRIC_KEYS},
            "seeds": per_seed,
        }
        print(variant, bitrate, block[bitrate]["mean"], flush=True)

    listen = PROBE / variant / "listen_45s.wav"
    if listen.exists():
        clip = ga.load(listen)[0]
        twin = mp3_roundtrip(listen, "64k",
                             PROBE / variant / "64k" / "listen_45s.wav")
        shortest = min(clip.shape[1], twin.shape[1])
        write_listening_pack(
            {"generated": clip[:, :shortest], "generated_64k": twin[:, :shortest]},
            SR, PROBE / variant / "listen")


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "auto"
    results: dict = {"prompt": PROMPT, "variants": {}, "anchors": {}}

    for variant in gp.PRIOR_VARIANTS:
        try:
            render_variant(variant, device)
        except Exception as error:  # one missing gate or wheel must not end the run
            print(f"render {variant} FAILED: {error}", flush=True)
            traceback.print_exc()
            results["variants"][variant] = {
                "error": f"{type(error).__name__}: {error}"
            }

    # Score whatever exists, whether rendered now or on an earlier pass.
    for variant in gp.PRIOR_VARIANTS:
        try:
            score_variant(variant, results)
        except Exception as error:
            print(f"score {variant} FAILED: {error}", flush=True)
            traceback.print_exc()
            results["variants"].setdefault(variant, {})["error"] = (
                f"{type(error).__name__}: {error}"
            )

    # The master anchors: what MP3 removes from real masters, same metrics.
    if XP_RESULTS.exists():
        xp = json.loads(XP_RESULTS.read_text())
        for source, bitrates in xp.items():
            results["anchors"][f"{source} master"] = {
                bitrate: {key: block["input"][key] for key in METRIC_KEYS}
                for bitrate, block in bitrates.items()
            }

    PROBE.mkdir(parents=True, exist_ok=True)
    (PROBE / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {PROBE / 'results.json'}")


if __name__ == "__main__":
    main()
