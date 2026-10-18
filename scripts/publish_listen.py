"""Build the published listening page — docs/listen/, served by GitHub Pages.

  uv run --group notebooks python scripts/publish_listen.py [--page-only]

The curated set from ADR-0011's listening verdict: six tracks per
(source x bitrate), each labeled with its 1-9 solo shortcut, for codec and
aerofunk at 32/64/128 kbps; the same-artist pair stays out of the published
set. Unlike `demo/`, the audio
is committed: copies at -14 LUFS under one common -1 dBFS gain per pack,
written as 24-bit FLAC with aerofunk at its full three minutes — 556 MB,
approved over the initial 500 MB budget.

`--page-only` rewrites index.html from the audio already on disk — for text
and layout edits. A full rebuild re-encodes every FLAC and PNG, and a library
version difference can change their bytes with no audible or visible change,
which git would then store in full.
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

INTRO = """\
<p>This page is the listening companion to
<a href="https://github.com/zbitouzakaria/grooveback">grooveback</a>, a project about
restoring music that only survives as lossy rips — records whose best remaining copy
is an MP3 someone uploaded years ago. Each set below is one controlled experiment: a
track with a clean master is compressed to MP3 at a known bitrate, and every method
tries to restore that MP3. You hear the same excerpt in six versions, and switching
keeps the playhead, so any difference you hear is the method.</p>
<p>The six tracks of every set:</p>
<ol class="note">
<li><b>Ground Truth</b> — the clean master the MP3 was made from: the target.</li>
<li><b>Degraded Input</b> — the MP3 itself, the file every method receives. The
spectrogram shows the visible damage: above the codec's cutoff frequency nothing is
kept. The audible damage also sits below the cutoff, where the codec smears
transients and thins out detail.</li>
<li><b>Apollo</b> — the strongest published model for this damage: a supervised
network trained on clean/MP3 pairs, restoring the full band in one pass.</li>
<li><b>A²SB</b> — NVIDIA's diffusion model for bandwidth extension. By design it
only generates content above the cutoff and passes the band below through
unchanged.</li>
<li><b>Apollo → εar-VAE</b> — this project's construction: Apollo's output passed
through the εar-VAE autoencoder (encode, then decode). The autoencoder is trained on
clean music only, so its decoder smooths the artifacts Apollo introduces on heavily
damaged input.</li>
<li><b>εar-VAE − MP3 damage</b> — the other construction: the input is encoded into
the autoencoder's latent space, the average shift that MP3 compression causes in
that space (measured on a different track) is subtracted, and the result is decoded.
It repairs spectral balance rather than fine detail.</li>
</ol>
<p class="note">The spectrogram under the buttons belongs to the selected track —
watch the region above the cutoff change as you switch, and click it to seek.
Headphones or monitors recommended: laptop speakers hide most of what these methods
disagree about.</p>"""

SOURCE_NOTES = {
    "codec": "A 6-second clip — the test asset from Apollo's release, the brighter of "
             "the two sources, short enough to loop and compare quickly.",
    "aerofunk": "A three-minute excerpt of an electronic track, compared against its "
                "clean master — long enough to judge the methods on real musical "
                "material.",
}
"""Source -> the sentence a visitor needs about the material."""

BITRATE_NOTES = {
    "32k": "At 32 kbps the codec keeps nothing above roughly 5.5 kHz: any treble in a "
           "restoration is invented, and the kept band is heavily smeared. This severe "
           "case is where Apollo → εar-VAE earns its place — by ear it is cleaner than "
           "Apollo alone, because the autoencoder smooths the artifacts Apollo "
           "introduces at this damage level. Compare tracks 2, 3 and 5 against 1.",
    "64k": "At 64 kbps the kept band reaches about 11 kHz. This is the crossover case: "
           "by ear, Apollo and Apollo → εar-VAE split the verdict — the smoothing "
           "still removes some of Apollo's artifacts, but begins to soften detail "
           "Apollo got right.",
    "128k": "At 128 kbps the kept band reaches about 16.5 kHz — this is what a typical "
            "rip looks like — and the remaining damage is subtle and sits inside the "
            "kept band. Here Apollo alone stays closest to the master: there is little "
            "left to smooth, so the autoencoder only moves the sound away from it. "
            "Expect small differences.",
}
"""Bitrate -> what this damage level means and ADR-0011's listening verdict on it."""


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


def main(page_only: bool = False) -> None:
    if not page_only:
        shutil.rmtree(OUT, ignore_errors=True)
        OUT.mkdir(parents=True)
        (REPO / "docs" / ".nojekyll").touch()
        for asset in ("trackswitch.js", "trackswitch-interactive-worker.js"):
            shutil.copy2(REPO / "demo" / asset, OUT / asset)

    sections = [INTRO]
    for source in SOURCES:
        for bitrate in BITRATES:
            pack = f"{source}_{bitrate}"
            print(f"publish: {pack}", flush=True)
            pack_dir = OUT / "audio" / pack
            if not page_only:
                matched = level_matched_set(pack_items(source, bitrate), SR)
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
                f'<p class="note">{SOURCE_NOTES[source]} {BITRATE_NOTES[bitrate]}</p>\n'
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
    main(page_only="--page-only" in sys.argv[1:])
