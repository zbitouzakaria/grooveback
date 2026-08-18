"""Cut the excerpt set: 12 s at 1:00 and at 3:00 of each source.

One fixed rule, no content analysis. 1:00 lands in the intro or first build,
3:00 in the main body, which is enough variety to hear whether a result holds
across a track. Sources too short for the rule are taken whole.

  data/excerpts/aerofunk_60s.wav    clean master
  data/excerpts/aerofunk_180s.wav
  data/excerpts/an2_60s.wav         real 128 kbps YouTube rip, no clean version
  data/excerpts/an2_180s.wav
  data/excerpts/codec.wav           6 s clean, severe case (whole file)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
from grooveback import audio as ga

SOURCES = {
    "aerofunk": "data/Aerofunk - Nice One (Cpu Cant Hack It Mix) 258.wav",
    "an2": "data/AN-2 - Moonshine (Deep Boogie Version) [2003].mp3",
    "codec": "data/original_codec_wav.wav",
}
OFFSETS_S = (60.0, 180.0)
DURATION_S = 12.0
OUT = Path("data/excerpts")


def main() -> None:
    for name, path in SOURCES.items():
        source, sr = ga.load(path)
        duration = source.shape[1] / sr
        if duration < max(OFFSETS_S) + DURATION_S:
            ga.save(OUT / f"{name}.wav", source, sr)
            print(f"{name}: whole file ({duration:.1f} s, too short for the rule)")
            continue
        for offset in OFFSETS_S:
            start = int(offset * sr)
            ga.save(OUT / f"{name}_{int(offset)}s.wav",
                    source[:, start : start + int(DURATION_S * sr)], sr)
        print(f"{name}: {DURATION_S:.0f} s at " +
              ", ".join(f"{int(o)}s" for o in OFFSETS_S))


if __name__ == "__main__":
    main()
