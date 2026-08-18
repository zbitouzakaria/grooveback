"""Everything behind ADR-0007, end to end.

  uv run python scripts/run_experiments.py [device] [variant ...]

Four steps, each skipped if its output already exists:

  1. cut 12 s excerpts at 1:00 and 3:00 of every source
  2. build clean/MP3 twins of the clean sources and cut them into windows
  3. encode every window to SAME latents
  4. run the round-trip and transport experiments, writing listening sets

Listening sets are named for the operation that produced them, so a directory
listing says what was run:

  1_input_mp3.wav                        before the transformation
  2_output_decode_encode_plus_shift.wav  after it
  ref_*.wav                              references, never the thing under test
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from grooveback import audio as ga
from grooveback import latents as gl
from grooveback.align import align_to, measure_drift
from grooveback.evaluation import write_listening_pack

SOURCES = {
    "aerofunk": "data/Aerofunk - Nice One (Cpu Cant Hack It Mix) 258.wav",
    "an2": "data/AN-2 - Moonshine (Deep Boogie Version) [2003].mp3",
    "codec": "data/original_codec_wav.wav",
}
VINYL = "data/AN-2 - Moonshine (Deep Boogie Version) - vinyl rip.flac"
POOL_SOURCE = "aerofunk"
"""The only source long enough and clean enough to fit a shift vector on."""

OFFSETS_S, DURATION_S = (60.0, 180.0), 12.0
BITRATES = ("128k", "192k")
BANDS = [(20, 250), (250, 1000), (1000, 4000), (4000, 8000),
         (8000, 12000), (12000, 16000), (16000, 20000), (20000, 22050)]

EXCERPTS, TWINS, OUT = Path("data/excerpts"), Path("data/twins"), Path("artifacts/same")


def trim(*arrays: np.ndarray) -> list[np.ndarray]:
    n = min(a.shape[1] for a in arrays)
    return [a[:, :n] for a in arrays]


def cached(path: Path) -> np.ndarray | None:
    """An existing render, or None if it has to be made.

    Renders are expensive and some are produced elsewhere — SAME-L runs on a
    GPU pod, being far too slow locally — so measurement must not depend on
    having generated them here.
    """
    return ga.load(path)[0] if path.exists() else None


def deltas(x: np.ndarray, reference: np.ndarray, sr: int) -> dict[str, float]:
    return {f"{lo}-{hi}": round(ga.band_energy_db(x, sr, lo, hi)
                                - ga.band_energy_db(reference, sr, lo, hi), 2)
            for lo, hi in BANDS}


# --- 1. excerpts -----------------------------------------------------------

def cut_excerpts() -> None:
    """12 s at 1:00 and 3:00. One rule: intro, then main body."""
    for name, path in SOURCES.items():
        source, sr = ga.load(path)
        if source.shape[1] / sr < max(OFFSETS_S) + DURATION_S:
            ga.save(EXCERPTS / f"{name}.wav", source, sr)
            continue
        for offset in OFFSETS_S:
            start = int(offset * sr)
            ga.save(EXCERPTS / f"{name}_{int(offset)}s.wav",
                    source[:, start : start + int(DURATION_S * sr)], sr)

    # The vinyl transfer of AN-2 is the one real clean reference we have. It
    # runs at a slightly wrong speed, so it needs resampling onto the rip's
    # timebase before the same offsets mean the same music.
    vinyl, sr = ga.load(VINYL)
    rip, _ = ga.load(SOURCES["an2"])
    drift = measure_drift(vinyl, rip, sr)
    aligned = align_to(vinyl, rip, sr, drift)
    print(f"vinyl: speed {drift.speed_ratio:.6f}, offset {drift.offset_samples}, "
          f"confidence {drift.confidence:.2f}, residual {drift.residual_samples:.0f} samples")
    for offset in OFFSETS_S:
        start = int(offset * sr)
        ga.save(EXCERPTS / f"an2_vinyl_{int(offset)}s.wav",
                aligned[:, start : start + int(DURATION_S * sr)], sr)
    return drift


# --- 2. twins --------------------------------------------------------------

def build_twins() -> None:
    """Encode whole tracks to MP3 and back, then cut windows from both."""
    for name in ("aerofunk", "codec"):
        clean, sr = ga.load(SOURCES[name])
        for bitrate in BITRATES:
            wav = TWINS / f"{name}_{bitrate}.wav"
            if not wav.exists():
                mp3 = TWINS / f"{name}_{bitrate}.mp3"
                mp3.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SOURCES[name],
                                "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)], check=True)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                                "-c:a", "pcm_f32le", str(wav)], check=True)
            degraded, _ = ga.load(wav)

            window = int(DURATION_S * sr)
            total = min(clean.shape[1], degraded.shape[1])
            if name != POOL_SOURCE:  # too short to pool; an application target
                cut = TWINS / "cut" / f"apply_{name}_{bitrate}"
                ga.save(cut / "clean.wav", clean[:, :total], sr)
                ga.save(cut / "degraded.wav", degraded[:, :total], sr)
                continue
            for i in range(total // window):
                start_s = i * DURATION_S
                held = next((o for o in OFFSETS_S
                             if start_s < o + DURATION_S and o < start_s + DURATION_S), None)
                cut = TWINS / "cut" / (f"apply_{name}_{int(held)}s_{bitrate}" if held
                                       is not None else f"pool{i:03d}_{bitrate}")
                ga.save(cut / "clean.wav", clean[:, i * window : (i + 1) * window], sr)
                ga.save(cut / "degraded.wav", degraded[:, i * window : (i + 1) * window], sr)


# --- 4. experiments --------------------------------------------------------

def roundtrip_experiment(variants: list[str], device: str) -> dict:
    """What a trip through the latent space costs, per variant."""
    results = {}
    for wav in sorted(EXCERPTS.glob("*.wav")):
        source, sr = ga.load(wav)
        pack = {"1_input": source}
        for variant in variants:
            decoded = gl.roundtrip(source, sr, variant=variant, device=device)
            src, dec = trim(source, decoded)
            tag = variant.replace("-", "_")
            pack[f"2_output_decode_encode_{tag}"] = dec
            residual = src - dec
            ga.save(OUT / "roundtrip" / wav.stem / f"residual_input_minus_{tag}.wav",
                    residual, sr)
            results[f"{wav.stem}.{variant}"] = {
                "residual_below_signal_db": round(float(
                    20 * np.log10(np.sqrt((residual**2).mean()) + 1e-12)
                    - 20 * np.log10(np.sqrt((src**2).mean()) + 1e-12)), 1),
                "band_delta_db": deltas(dec, src, sr),
            }
            print(f"  {wav.stem} {variant}: "
                  f"{results[f'{wav.stem}.{variant}']['residual_below_signal_db']} dB")
        write_listening_pack(dict(zip(pack, trim(*pack.values()), strict=True)),
                             sr, OUT / "roundtrip" / wav.stem)
    return results


def transport_experiment(variants: list[str], device: str) -> dict:
    """Fit one shift vector per bitrate, apply it to held-out excerpts."""
    cut = TWINS / "cut"
    report: dict = {}
    for variant in variants:
        for bitrate in BITRATES:
            pool = sorted(d for d in cut.iterdir()
                          if d.name.startswith("pool") and d.name.endswith(bitrate))
            clean_z = [np.load(d / f"clean.{variant}.npy") for d in pool]
            degraded_z = [np.load(d / f"degraded.{variant}.npy") for d in pool]
            shift = gl.transport_vector(clean_z, degraded_z)
            stats = gl.transport_linearity(clean_z, degraded_z, shift)
            stats["pool_windows"] = len(pool)
            (OUT / "transport").mkdir(parents=True, exist_ok=True)
            np.save(OUT / "transport" / f"shift_{bitrate}.{variant}.npy", shift)
            print(f"  {variant} {bitrate}: {stats}")
            key = f"{variant}.{bitrate}"
            report[key] = {"shift": stats, "excerpts": {}}

            targets = [(d.name.replace("apply_", "").replace(f"_{bitrate}", ""), d, None)
                       for d in sorted(cut.iterdir())
                       if d.name.startswith("apply") and d.name.endswith(bitrate)]
            # The real rip: already degraded, and now with a vinyl reference.
            for offset in OFFSETS_S:
                targets.append((f"an2_{int(offset)}s", None,
                                EXCERPTS / f"an2_{int(offset)}s.wav"))

            for name, twin_dir, rip_path in targets:
                if twin_dir is not None:
                    clean, sr = ga.load(twin_dir / "clean.wav")
                    mp3, _ = ga.load(twin_dir / "degraded.wav")
                else:
                    mp3, sr = ga.load(rip_path)
                    vinyl_path = EXCERPTS / f"{name.replace('an2', 'an2_vinyl')}.wav"
                    clean = ga.load(vinyl_path)[0] if vinyl_path.exists() else None

                out_dir = OUT / "transport" / f"{name}_{bitrate}_{variant}"
                shifted = cached(out_dir / "2_output_decode_encode_plus_shift.wav")
                if shifted is None:
                    shifted = gl.roundtrip_with_shift(mp3, sr, shift, variant, device)
                plain = cached(out_dir / "ref_decode_encode_input.wav")
                if plain is None:
                    plain = gl.roundtrip(mp3, sr, variant, device)

                pack = {"1_input_mp3": mp3,
                        "2_output_decode_encode_plus_shift": shifted,
                        "ref_decode_encode_input": plain}
                if clean is not None:
                    ceiling = cached(out_dir / "ref_decode_encode_clean.wav")
                    if ceiling is None:
                        ceiling = gl.roundtrip(clean, sr, variant, device)
                    pack["ref_clean_master"] = clean
                    pack["ref_decode_encode_clean"] = ceiling
                pack = dict(zip(pack, trim(*pack.values()), strict=True))
                write_listening_pack(pack, sr, out_dir)

                reference = pack.get("ref_clean_master", pack["ref_decode_encode_input"])
                report[key]["excerpts"][name] = {
                    "scored_against": ("clean reference" if clean is not None
                                       else "unshifted round-trip"),
                    **{k: deltas(v, reference, sr) for k, v in pack.items()},
                }
                print(f"    {name}")
    return report


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "auto"
    variants = sys.argv[2:] or ["same-s"]

    print("1. excerpts")
    cut_excerpts()
    print("2. twins")
    build_twins()
    print("3. encoding")
    for variant in variants:
        gl.encode_tree(TWINS / "cut", variant, device)
    print("4. experiments")
    results = {"roundtrip": roundtrip_experiment(variants, device),
               "transport": transport_experiment(variants, device)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'results.json'}")
