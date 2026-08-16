"""Overnight analysis: our A2SB wrapper vs their own script, and the full AN-2 run."""

import os
from pathlib import Path

import numpy as np

from grooveback import audio as ga
from grooveback.evaluation import write_listening_pack

BANDS = [(4000, 5000), (6000, 7000), (9000, 10000), (13000, 14000), (18000, 19000)]


def table(title: str, columns: dict[str, np.ndarray], sr: int, bands=BANDS) -> None:
    print(f"\n=== {title}")
    print(f"{'band':>13} " + " ".join(f"{name:>10}" for name in columns))
    for lo, hi in bands:
        row = " ".join(
            f"{ga.band_energy_db(a, sr, lo, hi):+10.1f}" for a in columns.values()
        )
        print(f"{lo / 1000:5.1f}-{hi / 1000:4.1f} kHz {row}")


def main() -> None:
    sr = 44100

    # 1. our wrapper against their script, same input and settings
    vanilla_path = os.environ.get("VANILLA", "")
    ours_path = Path("artifacts/a2sb/mono_codec_ours_2split.wav")
    if vanilla_path and Path(vanilla_path).exists() and ours_path.exists():
        vanilla, _ = ga.load(vanilla_path)
        ours, _ = ga.load(ours_path)
        n = min(vanilla.shape[1], ours.shape[1])
        vanilla, ours = vanilla[:1, :n], ours[:1, :n]
        table("ours vs vanilla (mono_codec_wav_cut4k, 2-split, 50 steps)",
              {"vanilla": vanilla, "ours": ours}, sr)

        # These are stochastic samplers, so the audio will not be identical.
        # What matters is that the spectra agree; a wrapper bug shows up as a
        # band-level difference, not as sample-level noise.
        deltas = [
            abs(ga.band_energy_db(ours, sr, lo, hi) - ga.band_energy_db(vanilla, sr, lo, hi))
            for lo, hi in BANDS
        ]
        print(f"\nmax band difference: {max(deltas):.2f} dB")
        print("VERDICT:", "wrapper agrees with upstream" if max(deltas) < 3.0
              else "DIVERGENCE - investigate")
    else:
        print("skipped wrapper comparison; vanilla or ours missing")

    # 2. the full AN-2 track
    an2 = Path("artifacts/a2sb/mono_an2_a2sb_full.wav")
    if an2.exists():
        before, _ = ga.load("data/mono_an2.wav")
        cut, _ = ga.load("data/mono_an2_cut.wav")
        after, _ = ga.load(an2)
        n = min(before.shape[1], cut.shape[1], after.shape[1])
        before, cut, after = before[:1, :n], cut[:1, :n], after[:1, :n]
        print(f"\nAN-2 full track: {n / sr / 60:.2f} min restored")
        table("AN-2 full track", {"before": before, "cut@knee": cut, "a2sb": after},
              sr, bands=[(15000, 16000), (16000, 17000), (17000, 18000),
                         (19000, 20000), (21000, 22000)])

        apollo = Path("artifacts/a2sb/mono_an2_apollo_full.wav")
        items = {"an2_before": before, "an2_a2sb": after}
        if apollo.exists():
            ap, _ = ga.load(apollo)
            items["an2_apollo"] = ap[:1, :n]
        written = write_listening_pack(items, sr, "artifacts/listen")
        print("\nlistening pack:")
        for name, path in written.items():
            a, _ = ga.load(path)
            print(f"  {path.name:28} {ga.loudness(a, sr):+7.2f} LUFS "
                  f"peak {ga.peak_dbfs(a):+6.2f} dBFS")
    else:
        print("\nAN-2 output missing")


if __name__ == "__main__":
    main()
