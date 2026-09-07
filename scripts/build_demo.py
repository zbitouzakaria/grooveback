"""One listening page per config, built on trackswitch.js.

  uv run --group notebooks python scripts/build_demo.py [demo/base/config.yaml ...]

No arguments: every demo/*/config.yaml. For each config, cut the configured
window from every track of every pack, re-level-match the excerpts as one set
(the packs were matched over their full length; a window can drift between
tracks by fractions of a dB), write them as flac with one spectrogram PNG
each, and generate index.html plus one trackswitch player JSON per section —
all next to the config file. Everything is regenerated on every run.

A `tracks` mapping (label -> filename) makes one player per pack. `tracks: all`
takes every flac in the pack and makes one player per variant family
(`small-music_s8`, `medium-base_s50`, ...), each a noise-level sweep behind the
input / master / apollo anchors; those players load only when their section is
expanded, so a page over hundreds of files stays openable.
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib import colormaps
from omegaconf import OmegaConf
from PIL import Image

from grooveback import audio as ga
from grooveback.evaluation import level_matched_set

REPO = Path(__file__).resolve().parents[1]
ANCHORS = [("input", "input.flac"), ("master", "original.flac"), ("apollo", "apollo.flac")]
"""Label -> filename of the non-sdedit tracks every `tracks: all` player carries."""


def load_window(path: Path, window_s: tuple[float, float] | None) -> tuple[np.ndarray, int]:
    """Read `(channels, samples)` float32, only the configured window.

    Reads the window directly rather than through `ga.load` so a 90 s excerpt
    of a 360 s render does not decode the other 270 s.
    """
    with sf.SoundFile(str(path)) as f:
        rate = f.samplerate
        start, frames = 0, -1
        if window_s is not None:
            start_s, duration_s = window_s
            start = int(start_s * rate)
            frames = min(int(duration_s * rate), f.frames - start)
        f.seek(start)
        audio = f.read(frames=frames, dtype="float32", always_2d=True)
    return np.ascontiguousarray(audio.T), rate


def save_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write `(channels, samples)` as 24-bit flac (flac has no float subtype)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sample_rate, subtype="PCM_24")


def save_spectrogram(path: Path, audio: np.ndarray) -> None:
    """A compact seek-strip spectrogram: 256 x <=1600 px, one px per cell.

    Absolute color range across every image — the -100 dB floor of
    `spectrogram_db` up to full scale — so tracks and packs are comparable by
    eye. Written as a 256-color palette PNG (a colormap is a palette), which
    keeps the committed images a fraction of full-color size.
    """
    spec = ga.spectrogram_db(audio, n_fft=1024, max_frames=1600)
    spec = spec[:512].reshape(256, 2, -1).mean(axis=1)
    index = np.clip((spec + 100.0) * (255.0 / 100.0), 0.0, 255.0).astype(np.uint8)
    palette = (colormaps["magma"](np.arange(256) / 255.0)[:, :3] * 255).astype(np.uint8)
    image = Image.fromarray(index[::-1], mode="P")  # flip: low frequencies at the bottom
    image.putpalette(palette.tobytes())
    image.save(path, optimize=True)


def variant_families(pack_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Family -> [(label, filename)] for every non-anchor flac, by noise level.

    `small-music_s8_0.16.flac` belongs to family `small-music_s8` and gets the
    label `noise 0.16`.
    """
    anchor_files = {filename for _, filename in ANCHORS}
    families: dict[str, list[tuple[float, str, str]]] = {}
    for f in sorted(pack_dir.glob("*.flac")):
        if f.name in anchor_files:
            continue
        family, noise = f.stem.rsplit("_", 1)
        families.setdefault(family, []).append((float(noise), f"noise {noise}", f.name))
    return {
        family: [(label, name) for _, label, name in sorted(entries)]
        for family, entries in sorted(families.items())
    }


def players_of(pack_name: str, spec: dict, pack_dir: Path) -> list[tuple[str, str, list]]:
    """(section id, section title, [(label, filename)]) per player of one pack."""
    if spec["tracks"] == "all":
        return [
            (f"{pack_name}--{family}", family, ANCHORS + sweep)
            for family, sweep in variant_families(pack_dir).items()
        ]
    return [(pack_name, pack_name, list(spec["tracks"].items()))]


def player_config(pack_name: str, tracks: list[tuple[str, str]]) -> dict:
    """The trackswitch player JSON: solo the first track, one image per track."""
    media, track_ids = {}, []
    for i, (label, filename) in enumerate(tracks):
        track_id = f"t{i}"
        media[track_id] = {
            "type": "audio",
            "title": label,
            "src": f"media/{pack_name}/{filename}",
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
            {"type": "perTrackImage", "seekable": True, "seekMarginLeft": 0, "seekMarginRight": 0},
            {"type": "trackList", "tracks": track_ids, "soloGroup": 0},
        ],
    }


def build_media(config_dir: Path, pack_name: str, spec: dict, filenames: set[str]) -> None:
    """Cut, re-level-match and write one pack's excerpts and spectrograms."""
    pack_dir = Path(spec["dir"])
    if not pack_dir.is_absolute():
        pack_dir = REPO / pack_dir
    window_s = tuple(spec["window_s"]) if "window_s" in spec else None

    excerpts, rates = {}, set()
    for filename in sorted(filenames):
        excerpts[filename], rate = load_window(pack_dir / filename, window_s)
        rates.add(rate)
    if len(rates) != 1:
        raise ValueError(f"{pack_dir} mixes sample rates {sorted(rates)}.")
    lengths = {audio.shape[1] for audio in excerpts.values()}
    if len(lengths) != 1:
        raise ValueError(f"{pack_dir} excerpts differ in length: {sorted(lengths)} samples.")

    matched = level_matched_set(excerpts, rate)
    for filename, audio in matched.items():
        out = config_dir / "media" / pack_name / filename
        save_flac(out, audio, rate)
        save_spectrogram(out.with_suffix(".png"), audio)


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

// Collapsed sections hold only a config path; the player element is created
// the first time its section is opened, and audio decodes on first play —
// memory follows what is actually listened to.
document.querySelectorAll("details[data-config]").forEach((section) => {
  section.addEventListener("toggle", () => {
    if (!section.open || section.dataset.loaded) return;
    section.dataset.loaded = "1";
    const player = document.createElement("trackswitch-player");
    player.setAttribute("config-src", section.dataset.config);
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

    sections = []
    for pack_name, spec in config.items():
        pack_dir = Path(spec["dir"])
        if not pack_dir.is_absolute():
            pack_dir = REPO / pack_dir
        players = players_of(pack_name, spec, pack_dir)

        filenames = {filename for _, _, tracks in players for _, filename in tracks}
        print(f"{config_dir.name}: {pack_name} — {len(filenames)} tracks,"
              f" {len(players)} player(s)", flush=True)
        build_media(config_dir, pack_name, spec, filenames)

        lazy = spec["tracks"] == "all"
        if lazy:
            sections.append(f"<h2>{escape(pack_name)}</h2>")
        for section_id, title, tracks in players:
            (config_dir / f"{section_id}.json").write_text(
                json.dumps(player_config(pack_name, tracks), indent=1)
            )
            if lazy:
                sections.append(
                    f'<details data-config="{section_id}.json">'
                    f"<summary>{escape(title)} ({len(tracks)} tracks)</summary></details>"
                )
            else:
                sections.append(
                    f"<h2>{escape(title)}</h2>\n"
                    f'<trackswitch-player config-src="{section_id}.json"></trackswitch-player>'
                )

    (config_dir / "index.html").write_text(
        PAGE.format(name=escape(config_dir.name), sections="\n".join(sections), script=SCRIPT)
    )
    print(f"wrote {config_dir / 'index.html'}")


def main() -> None:
    configs = [Path(arg) for arg in sys.argv[1:]] or sorted(REPO.glob("demo/*/config.yaml"))
    if not configs:
        raise SystemExit("no demo/*/config.yaml found")
    for config_path in configs:
        build_page(config_path)


if __name__ == "__main__":
    main()
