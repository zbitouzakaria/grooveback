"""Build the published listening page — docs/listen/, served by GitHub Pages.

  uv run --group notebooks python scripts/publish_listen.py

The curated set from ADR-0011's listening verdict: six tracks per
(source x bitrate), each labeled with its 1-9 solo shortcut, for codec and
aerofunk at 32/64/128 kbps; the same-artist pair stays out of the published set. Unlike `demo/`, the audio
is committed: copies at -14 LUFS under one common -1 dBFS gain per pack,
written as 24-bit FLAC with aerofunk at its full three minutes — 556 MB,
approved over the initial 500 MB budget.
"""

from __future__ import annotations

import json
import shutil
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from build_demo import PAGE, SCRIPT, player_config, save_spectrogram  # noqa: E402

from grooveback import audio as ga  # noqa: E402
from grooveback.evaluation import level_matched_set  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
XP = REPO / "artifacts" / "xp"
OUT = REPO / "docs" / "listen"
SR = 44_100
SIZE_BUDGET_MB = 600  # 556 MB at 24-bit/3 min, approved

SOURCES = ("codec", "aerofunk")
BITRATES = ("32k", "64k", "128k")
TRACKS = (
    ("1 - Ground Truth", "original"),
    ("2 - Degraded Input", "input"),
    ("3 - Apollo", "apollo"),
    ("4 - A²SB", "a2sb"),
    ("5 - Apollo → εar-VAE", "apollo-earvae"),
    ("6 - εar-VAE − MP3 damage", "earvae-sub"),
)
"""Label -> benchmark method, in the ADR-0011 keeper order. The prefix is the
trackswitch solo key."""


def pack_items(source: str, bitrate: str) -> dict[str, np.ndarray]:
    original = ga.load(XP / source / "original.wav")[0]
    items = {"original": original}
    for _, method in TRACKS[1:]:
        audio = ga.load(XP / source / bitrate / f"{method}.wav")[0]
        items[method] = audio[:, : original.shape[1]]
    lengths = {audio.shape[1] for audio in items.values()}
    if len(lengths) != 1:
        raise ValueError(f"{source} {bitrate}: track lengths differ: {sorted(lengths)}.")
    return items


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    (REPO / "docs" / ".nojekyll").touch()
    for asset in ("trackswitch.js", "trackswitch-interactive-worker.js"):
        shutil.copy2(REPO / "demo" / asset, OUT / asset)

    sections = []
    for source in SOURCES:
        for bitrate in BITRATES:
            pack = f"{source}_{bitrate}"
            print(f"publish: {pack}", flush=True)
            matched = level_matched_set(pack_items(source, bitrate), SR)
            pack_dir = OUT / "audio" / pack
            for _, method in TRACKS:
                ga.save(pack_dir / f"{method}.flac", matched[method], SR, subtype="PCM_24")
            (OUT / "media" / pack).mkdir(parents=True, exist_ok=True)
            for _, method in TRACKS:
                save_spectrogram(
                    OUT / "media" / pack / f"{method}.png", matched[method], SR
                )
            tracks = [(label, f"{method}.flac") for label, method in TRACKS]
            config_json = json.dumps(
                player_config(pack, tracks, f"audio/{pack}", pack_dir), indent=1
            ).replace("</", "<\\/")
            sections.append(
                f"<h2>{escape(pack)}</h2>\n"
                "<trackswitch-player>\n"
                f'<script type="application/json">{config_json}</script>\n'
                "</trackswitch-player>"
            )

    sections.append(
        '<p class="note">Powered by '
        '<a href="https://audiolabs.github.io/trackswitch.js/">trackswitch.js</a>'
        " (AudioLabs Erlangen)</p>"
    )
    page = PAGE.replace('src="../trackswitch.js"', 'src="trackswitch.js"')
    (OUT / "index.html").write_text(
        page.format(name="published packs", sections="\n".join(sections), script=SCRIPT)
    )

    total_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {OUT / 'index.html'} — {total_mb:.0f} MB total")
    if total_mb > SIZE_BUDGET_MB:
        raise SystemExit(
            f"{total_mb:.0f} MB exceeds the {SIZE_BUDGET_MB} MB budget — "
            "shorten the aerofunk cut or drop a bitrate."
        )


if __name__ == "__main__":
    main()
