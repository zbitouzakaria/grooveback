"""Produce the SAME round-trip and transport-vector results.

Two experiments, both writing level-matched listening sets whose filenames say
which operation produced them:

  artifacts/same/roundtrip/<excerpt>/
      1_input.wav
      2_output_decode_encode_same_s.wav
      2_output_decode_encode_same_l.wav
      residual_input_minus_same_s.wav        (unprocessed, before level matching)

  artifacts/same/transport/<excerpt>_<bitrate>/
      1_input_mp3.wav                        before the transformation
      2_output_decode_encode_plus_shift.wav  after the transformation
      ref_clean_master.wav                   the target
      ref_decode_encode_input.wav            same path, shift omitted
      ref_decode_encode_clean.wav            ceiling: best SAME can do

Prerequisites: scripts/pick_excerpts.py, scripts/build_twins.py,
scripts/encode_twins.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from grooveback import audio as ga
from grooveback import latents as gl
from grooveback.evaluation import write_listening_pack

BANDS = [(20, 250), (250, 1000), (1000, 4000), (4000, 8000),
         (8000, 12000), (12000, 16000), (16000, 20000), (20000, 22050)]
OUT = Path("artifacts/same")


def band_deltas(x: np.ndarray, reference: np.ndarray, sr: int) -> dict[str, float]:
    return {
        f"{lo}-{hi}": round(
            ga.band_energy_db(x, sr, lo, hi) - ga.band_energy_db(reference, sr, lo, hi), 2
        )
        for lo, hi in BANDS
    }


def trim(*arrays: np.ndarray) -> list[np.ndarray]:
    n = min(a.shape[1] for a in arrays)
    return [a[:, :n] for a in arrays]


def roundtrip_experiment(device: str = "auto") -> dict:
    """Encode and decode each excerpt through both SAME variants."""
    results = {}
    for wav in sorted(Path("data/excerpts").glob("*.wav")):
        source, sr = ga.load(wav)
        pack = {"1_input": source}
        for variant in gl.SAME_VARIANTS:
            decoded = gl.roundtrip(source, sr, variant=variant, device=device)
            source_t, decoded_t = trim(source, decoded)
            tag = variant.replace("-", "_")
            pack[f"2_output_decode_encode_{tag}"] = decoded_t

            residual = source_t - decoded_t
            ga.save(OUT / "roundtrip" / wav.stem / f"residual_input_minus_{tag}.wav",
                    residual, sr)
            results[f"{wav.stem}.{variant}"] = {
                "residual_below_signal_db": round(float(
                    20 * np.log10(np.sqrt((residual**2).mean()) + 1e-12)
                    - 20 * np.log10(np.sqrt((source_t**2).mean()) + 1e-12)), 1),
                "band_delta_db": band_deltas(decoded_t, source_t, sr),
            }
            print(f"{wav.stem} {variant}: "
                  f"{results[f'{wav.stem}.{variant}']['residual_below_signal_db']} dB")
        write_listening_pack(dict(zip(pack, trim(*pack.values()), strict=True)),
                             sr, OUT / "roundtrip" / wav.stem)
    return results


REAL_RIPS = {"an2_60s": "data/excerpts/an2_60s.wav",
             "an2_180s": "data/excerpts/an2_180s.wav"}
"""Already-degraded 128 kbps YouTube rips. No clean master exists, so these are
scored against the unshifted round-trip rather than a target — the point is
whether a vector fitted on synthetic twins does anything sane on a real rip."""


def transport_experiment(device: str = "auto") -> dict:
    """Estimate one shift vector per bitrate, apply it to held-out excerpts."""
    cut = Path("data/twins/cut")
    report = {}
    for bitrate in ("128k", "192k"):
        pool = sorted(d for d in cut.iterdir()
                      if d.name.startswith("pool") and d.name.endswith(bitrate))
        clean_z = [np.load(d / "clean.same-s.npy") for d in pool]
        degraded_z = [np.load(d / "degraded.same-s.npy") for d in pool]

        shift = gl.transport_vector(clean_z, degraded_z)
        stats = gl.transport_linearity(clean_z, degraded_z, shift)
        stats["pool_windows"] = len(pool)
        (OUT / "transport").mkdir(parents=True, exist_ok=True)
        np.save(OUT / "transport" / f"shift_{bitrate}.npy", shift)
        print(f"{bitrate}: {stats}")
        report[bitrate] = {"shift": stats, "excerpts": {}}

        targets = [(d.name.replace("apply_", "").replace(f"_{bitrate}", ""), d, None)
                   for d in sorted(cut.iterdir())
                   if d.name.startswith("apply") and d.name.endswith(bitrate)]
        targets += [(name, None, path) for name, path in REAL_RIPS.items()]

        for name, twin_dir, rip_path in targets:
            if twin_dir is not None:
                clean, sr = ga.load(twin_dir / "clean.wav")
                mp3, _ = ga.load(twin_dir / "degraded.wav")
            else:
                clean, (mp3, sr) = None, ga.load(rip_path)

            after = gl.roundtrip_with_shift(mp3, sr, shift, device=device)
            no_shift = gl.roundtrip(mp3, sr, device=device)

            pack = {"1_input_mp3": mp3,
                    "2_output_decode_encode_plus_shift": after,
                    "ref_decode_encode_input": no_shift}
            if clean is not None:
                pack["ref_clean_master"] = clean
                pack["ref_decode_encode_clean"] = gl.roundtrip(clean, sr, device=device)

            values = trim(*pack.values())
            pack = dict(zip(pack, values, strict=True))
            write_listening_pack(pack, sr, OUT / "transport" / f"{name}_{bitrate}")

            reference = pack.get("ref_clean_master", pack["ref_decode_encode_input"])
            report[bitrate]["excerpts"][name] = {
                "scored_against": (
                    "clean master" if clean is not None else "unshifted round-trip"),
                **{key: band_deltas(value, reference, sr) for key, value in pack.items()},
            }
            print(f"  {name} {bitrate}: done")
    return report


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "auto"
    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "roundtrip": roundtrip_experiment(device),
        "transport": transport_experiment(device),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'results.json'}")
