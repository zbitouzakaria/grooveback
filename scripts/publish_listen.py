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
<p>These are the listening sets of
<a href="https://github.com/zbitouzakaria/grooveback">grooveback</a>, a project
restoring records that survive only as MP3/AAC/Opus rips. Each set below is one
experiment: a track with a clean master is compressed to MP3 at a known bitrate,
and each method restores that MP3. All tracks of a set hold the same excerpt, so
switching between them compares the methods directly.</p>
<p>The six tracks:</p>
<ol class="note">
<li><b>Ground Truth</b> — the clean master the MP3 was made from; the reference.</li>
<li><b>Degraded Input</b> — the MP3 each method receives. Above the codec's cutoff
frequency nothing is kept, which the spectrogram shows plainly; below the cutoff
the signal survives but degrades — transients smear, and reverb tails and stereo
detail thin out.</li>
<li><b><a href="https://arxiv.org/abs/2409.08514">Apollo</a></b> — a supervised
model trained on MP3-compressed music; it restores the full band in one pass.</li>
<li><b><a href="https://arxiv.org/abs/2501.11311">A²SB</a></b> — a diffusion model
for bandwidth extension; by design it generates only the band above the cutoff and
leaves the band below unchanged.</li>
<li><b>Apollo → <a href="https://arxiv.org/abs/2509.14912">εar-VAE</a></b> — our
chained method: Apollo's output passed through the εar-VAE autoencoder. The
autoencoder is trained on clean music only, so the round-trip smooths the
artifacts Apollo introduces on heavily degraded input.</li>
<li><b>εar-VAE − MP3 damage</b> — our latent damage subtraction: the input is
encoded, the mean shift that MP3 compression causes in εar-VAE's latent space
(measured on a different track) is subtracted, and the result is decoded. It
restores spectral balance rather than waveform detail.</li>
</ol>
<p class="note">The spectrogram belongs to the selected track, and clicking it
seeks. Differences are clearest on monitoring headphones or speakers.</p>"""

SOURCE_NOTES = {
    "codec": "A 6 s clip, the test asset from Apollo's release — the brighter of "
             "the two sources.",
    "aerofunk": "A 180 s excerpt of an electronic track, compared against its "
                "clean master.",
}
"""Source -> one sentence on the material."""

BITRATE_NOTES = {
    "32k": "At 32 kbps the codec keeps nothing above about 5.5 kHz; everything a "
           "method shows above that edge is generated, not recovered. This is the "
           "severe case, where Apollo → εar-VAE sounds cleaner than Apollo alone: "
           "the round-trip removes the artifacts Apollo introduces at this damage "
           "level.",
    "64k": "At 64 kbps the kept band reaches about 11 kHz. This is the crossover "
           "case: by ear, the verdict splits between Apollo and Apollo → εar-VAE.",
    "128k": "At 128 kbps the kept band reaches about 16.5 kHz — the damage level "
            "of a typical rip — and most of the remaining damage sits inside the "
            "kept band. Here Apollo alone stays closest to the master; the "
            "smoothing that helps at 32 kbps moves the output away from it.",
}
"""Bitrate -> the damage level and ADR-0011's listening verdict on it."""


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
