"""One listening page per config, built on trackswitch.js.

  uv run --group notebooks python scripts/build_demo.py [demo/base/config.yaml ...]

No arguments: every demo/*/config.yaml. The audio is served from where the
experiment wrote it: `demo/artifacts` is a symlink to the repo's artifacts
directory, players reference each pack's files through it by relative URL,
and the site root stays demo/ (`python3 -m http.server 8880 -d demo`).

The script writes only pages and spectrogram PNGs — each config's media/
directory is wiped and rebuilt from the linked pack files on every run.
Before a pack is served, a gate proves it is one level-matched set — every
track's integrated loudness within 0.1 LU of the others' — which is what the
page's loudness note promises.

A `tracks` mapping (label -> filename) makes one player per pack. `tracks: all`
takes every flac in the pack and makes one player per variant family
(`small-music_s8`, `medium-base_s50`, ...), each a noise-level sweep behind the
input / master / apollo anchors; those players are created only when their
section is expanded, and at most one keeps decoded audio at a time.
"""

from __future__ import annotations

import json
import shutil
import sys
from html import escape
from pathlib import Path

import matplotlib
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grooveback import audio as ga  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ANCHORS = [
    ("input", "input.flac"),
    ("master", "original.flac"),
    ("apollo", "apollo.flac"),
    ("a2sb", "a2sb.flac"),
]
"""Label -> filename of the non-sdedit tracks every `tracks: all` player carries."""

SWEEP_FAMILIES = {
    "n": "sdedit noise sweep",
    "t": "sdedit theta sweep",
    "p": "sdedit prompted, cfg sweep at n 0.25",
    "q": "sdedit prompted positive-only, cfg sweep at n 0.25",
}
"""Family of a `sdedit_{family}_{value}.flac` render -> its player section."""

AX_RECT = (0.050, 0.20, 0.870, 0.72)
"""The plot box inside the figure, as fractions: left, bottom, width, height.

The player's seek margins below are derived from it, so the playhead and
click-to-seek span exactly the plotted time axis.
"""
CBAR_RECT = (0.930, 0.20, 0.011, 0.72)
SEEK_MARGIN_LEFT = AX_RECT[0] * 100.0
SEEK_MARGIN_RIGHT = (1.0 - AX_RECT[0] - AX_RECT[2]) * 100.0


def save_spectrogram(path: Path, audio, sample_rate: int) -> None:
    """A labeled spectrogram figure: time and frequency axes plus a level bar.

    Absolute color range on every image — the -100 dB floor of
    `spectrogram_db` up to full scale — so tracks and packs are comparable
    by eye and against the shared colorbar.
    """
    spec = ga.spectrogram_db(audio, n_fft=1024, max_frames=1600)
    duration_s = audio.shape[1] / sample_rate

    fig = plt.figure(figsize=(16, 2.8), dpi=100)
    ax = fig.add_axes(AX_RECT)
    image = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-100.0,
        vmax=0.0,
        extent=(0.0, duration_s, 0.0, sample_rate / 2000.0),
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [kHz]")
    fig.colorbar(image, cax=fig.add_axes(CBAR_RECT), label="[dBFS]")
    fig.savefig(path)
    plt.close(fig)


def variant_families(pack_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Family -> [(label, filename)] for every non-anchor flac, by sweep value.

    A render is named `sdedit_{family}_{value}.flac`: `sdedit_n_0.15.flac`
    belongs to the noise sweep with label `n 0.15`; SWEEP_FAMILIES lists the
    families and fixes the section order.
    """
    anchor_files = {filename for _, filename in ANCHORS}
    families: dict[str, list[tuple[float, str, str]]] = {}
    for f in sorted(pack_dir.glob("*.flac")):
        if f.name in anchor_files:
            continue
        _, family, value = f.stem.split("_")
        families.setdefault(SWEEP_FAMILIES[family], []).append(
            (float(value), f"{family} {value}", f.name)
        )
    section_order = list(SWEEP_FAMILIES.values())
    return {
        family: [(label, name) for _, label, name in sorted(entries)]
        for family, entries in sorted(
            families.items(), key=lambda item: section_order.index(item[0])
        )
    }


def players_of(pack_name: str, spec: dict, pack_dir: Path) -> list[tuple[str, list]]:
    """(section title, [(label, filename)]) per player of one pack."""
    if spec["tracks"] == "all":
        return [
            (family, ANCHORS + sweep)
            for family, sweep in variant_families(pack_dir).items()
        ]
    return [(pack_name, list(spec["tracks"].items()))]


def player_config(pack_name: str, tracks: list[tuple[str, str]], audio_base: str) -> dict:
    """The trackswitch player JSON: solo the first track, one image per track."""
    media, track_ids = {}, []
    for i, (label, filename) in enumerate(tracks):
        track_id = f"t{i}"
        media[track_id] = {
            "type": "audio",
            "title": label,
            "src": f"{audio_base}/{filename}",
            "imageID": f"{track_id}Plot",
            **({"solo": True} if i == 0 else {}),
        }
        media[f"{track_id}Plot"] = {
            "type": "image",
            "src": f"media/{pack_name}/{Path(filename).with_suffix('.png')}",
        }
        track_ids.append(track_id)
    return {
        "media": media,
        "views": [
            {
                "type": "navigationBar",
                "controls": ["playback", "globalVolume", "looping", "seekBar", "timer"],
                "repeatEnabled": True,
            },
            {
                "type": "perTrackImage",
                "seekable": True,
                "seekMarginLeft": SEEK_MARGIN_LEFT,
                "seekMarginRight": SEEK_MARGIN_RIGHT,
            },
            {"type": "trackList", "tracks": track_ids, "soloGroup": 0},
        ],
    }


def build_media(config_dir: Path, pack_name: str, pack_dir: Path, filenames: set[str]) -> None:
    """One spectrogram per track under media/{pack}, and the level gate.

    The audio itself is never copied or rewritten — players reference the
    pack files where the experiment wrote them.
    """
    media_dir = config_dir / "media" / pack_name
    media_dir.mkdir(parents=True, exist_ok=True)

    loudness_by_file, rates, lengths = {}, set(), set()
    for filename in sorted(filenames):
        audio, rate = ga.load(pack_dir / filename)
        rates.add(rate)
        lengths.add(audio.shape[1])
        loudness_by_file[filename] = ga.loudness(audio, rate)
        save_spectrogram((media_dir / filename).with_suffix(".png"), audio, rate)

    if len(rates) != 1:
        raise ValueError(f"{pack_dir} mixes sample rates {sorted(rates)}.")
    if len(lengths) != 1:
        raise ValueError(f"{pack_dir} tracks differ in length: {sorted(lengths)} samples.")
    spread = max(loudness_by_file.values()) - min(loudness_by_file.values())
    if spread > 0.1:
        raise ValueError(
            f"{pack_dir} is not one level-matched set (loudness spans "
            f"{spread:.2f} LU) — regenerate the listen pack before serving it."
        )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>grooveback listening — {name}</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
         padding: 0 1rem; background: #fafafa; color: #222; }}
  h2 {{ margin: 2rem 0 .5rem; }}
  details {{ margin: .4rem 0; }}
  summary {{ cursor: pointer; font-weight: 600; }}
  trackswitch-player {{ display: block; margin: .5rem 0 1.5rem; }}
  .note {{ color: #555; }}
</style>
<script src="../trackswitch.js"></script>
</head>
<body>
<h1>grooveback listening — {name}</h1>
<p class="note">Every track is level-matched to −14 LUFS with one common headroom gain per
player, master included. Solo is exclusive: keys 1–9 or a click switch tracks in place.
Space plays, R toggles repeat, loop a section with the loop controls or A/B/L.</p>
{sections}
<script>
{script}
</script>
</body>
</html>
"""

SCRIPT = """\
if (!customElements.get(TrackSwitch.TRACKSWITCH_DEFAULT_ELEMENT_NAME)) {
  TrackSwitch.defineTrackswitchDefaultElement();
}

// The bundle's own declarative boot never completes for parser-inserted
// players carrying an inline config (dynamically created ones boot fine),
// so kick each one explicitly; the element's config-load generation counter
// makes this later call the one that wins.
document.querySelectorAll("trackswitch-player").forEach((player) => {
  if (!player.currentConfig) player.loadDeclarativeConfig();
});

// Collapsed sections hold their player config as an inline JSON script; the
// player element is created around it the first time the section is opened,
// and audio decodes on first play — memory follows what is actually
// listened to. The config script must be inside the element before it
// connects, so it is moved in first.
document.querySelectorAll("details").forEach((section) => {
  const config = section.querySelector("script[data-player-config]");
  if (!config) return;
  section.addEventListener("toggle", () => {
    if (!section.open || section.dataset.loaded) return;
    section.dataset.loaded = "1";
    const player = document.createElement("trackswitch-player");
    player.appendChild(config);
    section.appendChild(player);
  });
});

// At most one decoded player per page, torn down as soon as another one
// STARTS loading — the freed audio does not wait for the new decode to
// finish, so the memory peak is one decoded player plus the file currently
// decoding, never two full players. destroyController drops the audio;
// applyCurrentConfig re-mounts the idle UI (power button) from the same
// config. A re-mounted player gets a fresh controller, so arming re-runs
// from the sweep below; the "loaded" listener covers decodes faster than
// one sweep tick.
const unloadOtherPlayers = (active) => {
  document.querySelectorAll("trackswitch-player").forEach((other) => {
    if (other === active || !other.controller) return;
    if (!other.controller.getState().isLoaded) return;
    other.destroyController();
    other.applyCurrentConfig();
    delete other.dataset.soleLoadedArmed;
  });
};
const armSoleLoadedPlayer = (player) => {
  const controller = player.controller;
  if (!controller || player.dataset.soleLoadedArmed) return;
  player.dataset.soleLoadedArmed = "1";
  controller.on("loaded", () => unloadOtherPlayers(player));
};
setInterval(() => {
  const players = [...document.querySelectorAll("trackswitch-player")];
  players.forEach(armSoleLoadedPlayer);
  const loading = players.find(
    (p) => p.controller && p.controller.getState().isLoading
  );
  if (loading) unloadOtherPlayers(loading);
}, 500);
"""


def build_page(config_path: Path) -> None:
    config_dir = config_path.parent
    config = OmegaConf.to_container(OmegaConf.load(config_path))
    shutil.rmtree(config_dir / "media", ignore_errors=True)

    sections = []
    for pack_name, spec in config.items():
        rel_dir = Path(spec["dir"])
        if rel_dir.is_absolute():
            if not rel_dir.is_relative_to(REPO):
                raise ValueError(f"{rel_dir} is outside the repo; the page cannot reach it.")
            rel_dir = rel_dir.relative_to(REPO)
        if rel_dir.parts[0] != "artifacts":
            raise ValueError(
                f"{rel_dir} is not under artifacts/ — the pages reach audio only "
                "through the demo/artifacts symlink."
            )
        pack_dir = REPO / rel_dir
        audio_base = f"../{rel_dir.as_posix()}"
        players = players_of(pack_name, spec, pack_dir)

        filenames = {filename for _, tracks in players for _, filename in tracks}
        print(f"{config_dir.name}: {pack_name} — {len(filenames)} tracks,"
              f" {len(players)} player(s)", flush=True)
        build_media(config_dir, pack_name, pack_dir, filenames)

        lazy = spec["tracks"] == "all"
        if lazy:
            sections.append(f"<h2>{escape(pack_name)}</h2>")
        for title, tracks in players:
            # "</" cannot appear inside a script block; "<\/" is the same JSON value.
            config_json = json.dumps(
                player_config(pack_name, tracks, audio_base), indent=1
            ).replace("</", "<\\/")
            if lazy:
                sections.append(
                    f"<details><summary>{escape(title)} ({len(tracks)} tracks)</summary>\n"
                    f'<script type="application/json" data-player-config>{config_json}</script>\n'
                    "</details>"
                )
            else:
                sections.append(
                    f"<h2>{escape(title)}</h2>\n"
                    "<trackswitch-player>\n"
                    f'<script type="application/json">{config_json}</script>\n'
                    "</trackswitch-player>"
                )

    (config_dir / "index.html").write_text(
        PAGE.format(name=escape(config_dir.name), sections="\n".join(sections), script=SCRIPT)
    )
    print(f"wrote {config_dir / 'index.html'}")


def main() -> None:
    link = REPO / "demo" / "artifacts"
    if not link.is_symlink():
        link.symlink_to(Path("..") / "artifacts")

    configs = [Path(arg) for arg in sys.argv[1:]] or sorted(REPO.glob("demo/*/config.yaml"))
    if not configs:
        raise SystemExit("no demo/*/config.yaml found")
    for config_path in configs:
        build_page(config_path)


if __name__ == "__main__":
    main()
