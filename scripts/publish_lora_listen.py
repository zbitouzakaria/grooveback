"""Build the LoRA's listening page, Apollo next to it — docs/lora/, served by GitHub Pages.

  uv run --group notebooks python scripts/publish_lora_listen.py select <scores.csv> <clips dir>
  uv run --group notebooks python scripts/publish_lora_listen.py build <scores.csv> \\
      <renders dir> <figure.png>

<scores.csv> holds every MUSDB18-HQ test clip's scores at six bitrates for the
MP3, Apollo and the LoRA: columns track, start, kbps and
{input,apollo,lora}_{sdr,si_snr,spectral_snr,lsd}, from ADR-0015's Apollo run
and the LoRA's run on the same clips. For each model and each bitrate on the
page, `select` picks the 3 songs where its SDR gains most over the MP3 it was
given and the 3 where it gains least, each by its most extreme clip, and
writes them, one list per bitrate, to <clips dir>/clips-{k}k.csv for rendering.
`build` reads those renders,
{renders dir}/{k}k/{track}@{start}/{original,input,apollo,lora-remix,master-roundtrip}.wav,
scores them, and writes the page with the bitrate figure. Like docs/listen/,
the audio is committed: copies at -14 LUFS under one common -1 dBFS gain per
clip, 16-bit FLAC with triangular dither.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
from build_demo import PAGE, SCRIPT, player_config, save_spectrogram  # noqa: E402
from publish_listen import dithered  # noqa: E402

from grooveback import audio as ga  # noqa: E402
from grooveback.evaluation import (  # noqa: E402
    bss_sdr_db,
    bss_si_sdr_db,
    level_matched_set,
    log_spectral_distance_db,
    spectral_snr_db,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "lora"
SR = 44_100
BITRATES = (32, 64)
PICKED = 3
MODELS = {"apollo": "Apollo", "lora": "the LoRA"}
"""Score column prefix -> name in the section headings."""
TRACKS = (
    ("1 - Master", "original"),
    ("2 - MP3", "input"),
    ("3 - Apollo", "apollo"),
    ("4 - LoRA", "lora-remix"),
    ("5 - Master through SA3's autoencoder", "master-roundtrip"),
)
"""Label -> render file. The prefix is the trackswitch solo key."""
METRICS = {
    "sdr": ("SDR", bss_sdr_db),
    "si_snr": ("SI-SNR", bss_si_sdr_db),
    "spectral_snr": ("spectral SNR", spectral_snr_db),
    "lsd": ("LSD (lower is better)", log_spectral_distance_db),
}

INTRO = """\
<p>An initial run of <a href="https://github.com/zbitouzakaria/grooveback">grooveback</a>'s
own restorer: Stable Audio 3 medium-base, fine-tuned with a LoRA for 5 hours on one GPU to
turn MP3s back into their masters, next to <a href="https://arxiv.org/abs/2409.08514">Apollo</a>,
a model trained for the same task. The LoRA trained on fresh windows of the 86 MUSDB18-HQ
training songs and on remixes of their stems, as Apollo does. Both restore every 6 s clip of
the 50 MUSDB18-HQ test songs, 3,677 clips, at six MP3 bitrates; the figure shows their mean
scores against the master. We are now iterating on data augmentation and training
strategies.</p>
<p><img src="bitrates.png?v={figure_version}" style="width:100%"
alt="SDR, SI-SNR, spectral SNR and LSD per MP3 bitrate on all 3,677 MUSDB18-HQ test clips,
for the MP3, Apollo and the LoRA"></p>
<p>{summary}</p>
<p>Each player below holds one clip, five ways:</p>
<ol class="note">
<li><b>Master</b> — the clean clip; the reference.</li>
<li><b>MP3</b> — the master coded at the section's bitrate, which both models receive.</li>
<li><b>Apollo</b> — Apollo's restoration of the MP3.</li>
<li><b>LoRA</b> — the LoRA's restoration, Euler sampler at 32 steps.</li>
<li><b>Master through SA3's autoencoder</b> — the master encoded and decoded by Stable Audio
3's autoencoder, with nothing else done: the closest the LoRA's output can get to the
master, since everything it makes goes through that decoder.</li>
</ol>
<p class="note">For each model and bitrate, the clips come from the 3 songs where its SDR
gains most over the MP3 it was given and the 3 where it gains least, one clip per song, the
most extreme. A clip can appear more than once. Scores are in dB against the master,
measured on these files before level matching. The
spectrogram belongs to the selected track, and clicking it seeks.</p>"""


SUMMARY = (
    "Apollo scores above the LoRA on all four metrics at every bitrate: at 32 kbps, 11.8 "
    "against 7.1 dB SDR, and 9.1 against 12.1 dB LSD. The LoRA beats the untouched MP3 only on "
    "LSD, which counts the band above the codec's cutoff that the MP3 lost: 12.1 against 32.0 "
    "dB at 32 kbps."
)
"""What the figure shows, from the means that compare_full prints."""


def slug(track: str, start: int) -> str:
    """A URL-safe folder name for one clip: "Al James - Schoolboy Facination" at 0
    is al-james-schoolboy-facination-0."""
    return f"{re.sub(r'[^a-z0-9]+', '-', track.lower()).strip('-')}-{start}"


def picks(scores: pd.DataFrame) -> list[tuple[int, str, str, pd.DataFrame]]:
    """(kbps, model, "most" or "least", the 3 clips) for every section of the page."""
    sections = []
    for kbps in BITRATES:
        clips = scores[scores["kbps"] == kbps]
        for model in MODELS:
            gain = clips[f"{model}_sdr"] - clips["input_sdr"]
            ranked = clips.assign(gain=gain).sort_values("gain", ascending=False)
            # Three different songs, each by its most extreme clip: neighbouring clips
            # share 3 s, so a song's best clips come in runs.
            sections.append((kbps, model, "most", ranked.drop_duplicates("track").head(PICKED)))
            least = ranked.iloc[::-1].drop_duplicates("track").head(PICKED)
            sections.append((kbps, model, "least", least))
    return sections


def select(scores: pd.DataFrame, clips_dir: Path) -> None:
    clips_dir.mkdir(parents=True, exist_ok=True)
    chosen = pd.concat([clips.assign(kbps=kbps) for kbps, _, _, clips in picks(scores)])
    for kbps in BITRATES:
        unique = chosen[chosen["kbps"] == kbps][["track", "start"]].drop_duplicates()
        unique.to_csv(clips_dir / f"clips-{kbps}k.csv", index=False)
        print(f"{kbps}k: {len(unique)} clips to render")


def build(scores: pd.DataFrame, renders_dir: Path, figure: Path) -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    for asset in ("trackswitch.js", "trackswitch-interactive-worker.js"):
        shutil.copy2(REPO / "demo" / asset, OUT / asset)
    shutil.copy2(figure, OUT / "bitrates.png")

    sections = [
        INTRO.format(figure_version=int((OUT / "bitrates.png").stat().st_mtime), summary=SUMMARY)
    ]
    written = set()
    current = None
    for kbps, model, end, clips in picks(scores):
        if kbps != current:
            sections.append(f"<h2>{kbps} kbps</h2>")
            current = kbps
        sections.append(f"<h3>Where {MODELS[model]} gains {end} over the MP3</h3>")
        for clip in clips.itertuples():
            name = slug(clip.track, clip.start)
            pack_dir = OUT / "audio" / f"{kbps}k" / name
            source = renders_dir / f"{kbps}k" / f"{clip.track}@{clip.start}"
            items = {stem: ga.load(source / f"{stem}.wav")[0] for _, stem in TRACKS}
            table = {
                stem: {
                    metric: score(items["original"], items[stem])
                    for metric, (_, score) in METRICS.items()
                }
                for _, stem in TRACKS[1:]
            }
            # The renders must be the clips the scores belong to: the MP3 and Apollo are
            # the same computation, the LoRA the same sampling but batched there.
            for stem, column, tolerance in (
                ("input", "input", 1e-3),
                ("apollo", "apollo", 1e-2),
                ("lora-remix", "lora", 0.1),
            ):
                gap = abs(table[stem]["sdr"] - getattr(clip, f"{column}_sdr"))
                if gap > tolerance:
                    raise SystemExit(
                        f"{name} at {kbps}k: {stem} SDR is {gap:.3f} dB off its score."
                    )
            if (kbps, name) not in written:
                matched = level_matched_set(items, SR)
                (OUT / "media" / f"{kbps}k-{name}").mkdir(parents=True, exist_ok=True)
                for _, stem in TRACKS:
                    ga.save(
                        pack_dir / f"{stem}.flac", dithered(matched[stem]), SR, subtype="PCM_16"
                    )
                    save_spectrogram(
                        OUT / "media" / f"{kbps}k-{name}" / f"{stem}.png", matched[stem], SR
                    )
                written.add((kbps, name))
            header = "".join(f"<th>{escape(label)}</th>" for label, _ in METRICS.values())
            rows = "".join(
                f"<tr><td>{escape(label.split(' - ', 1)[1])}</td>"
                + "".join(f"<td>{table[stem][metric]:.2f}</td>" for metric in METRICS)
                + "</tr>"
                for label, stem in TRACKS[1:]
            )
            gain = getattr(clip, f"{model}_sdr") - clip.input_sdr
            seconds = f"{clip.start / SR:.0f}–{clip.start / SR + 6:.0f} s"
            title = f"{clip.track}, {seconds} ({gain:+.2f} dB SDR over the MP3)"
            tracks = [(label, f"{stem}.flac") for label, stem in TRACKS]
            config = json.dumps(
                player_config(f"{kbps}k-{name}", tracks, f"audio/{kbps}k/{name}", pack_dir),
                indent=1,
            ).replace("</", "<\\/")
            sections.append(
                f"<h4>{escape(title)}</h4>\n"
                f'<table class="metrics"><tr><th>dB</th>{header}</tr>{rows}</table>\n'
                "<trackswitch-player>\n"
                f'<script type="application/json">{config}</script>\n'
                "</trackswitch-player>"
            )
            print(f"{kbps}k {model} {end}: {name}", flush=True)

    sections.append(
        '<p class="note">The earlier listening sets, every open method on two tracks, are '
        '<a href="../listen/">here</a>. Powered by '
        '<a href="https://audiolabs.github.io/trackswitch.js/">trackswitch.js</a>'
        " (AudioLabs Erlangen)</p>"
    )
    # PAGE is a str.format template, so its CSS braces are doubled.
    page = PAGE.replace('src="../trackswitch.js"', 'src="trackswitch.js"').replace(
        "</style>",
        "  h4 {{ margin: 1.4rem 0 .3rem; }}\n"
        "  table.metrics {{ border-collapse: collapse; margin: .3rem 0 .6rem; font-size: 14px; }}\n"
        "  table.metrics th, table.metrics td {{ padding: .15rem .7rem; text-align: right;"
        " border-bottom: 1px solid #ddd; }}\n"
        "  table.metrics th:first-child, table.metrics td:first-child {{ text-align: left; }}\n"
        "</style>",
    )
    (OUT / "index.html").write_text(
        page.format(name="LoRA fine-tune and Apollo", sections="\n".join(sections), script=SCRIPT)
    )
    total_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {OUT / 'index.html'}: {len(written)} clips, {total_mb:.0f} MB")


if __name__ == "__main__":
    command, scores_path, target = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    scores = pd.read_csv(scores_path)
    if command == "select":
        select(scores, target)
    else:
        build(scores, target, Path(sys.argv[4]))
