# 10. Listen in the browser with trackswitch.js

Date: 2026-09-07

## Status

Accepted

## Context

Listening verdicts (ADR-0006, the open sdedit comparison) are made by A/B-ing level-matched renders. A file player
restarts from zero on every switch, which makes real comparison impossible; what the comparisons need is switching
that keeps the playhead, exclusive solo under one key each, loopable sections, per-track spectrograms, and the
method's real name on every button. Surveyed and passed over: jaakkopasanen/ABX (blind-test shaped, anonymizes
labels), webMUSHRA-class suites (rating machinery we don't want), a hand-rolled Web Audio page (the maintained
purpose-built player already exists).

## Decision

Vendor trackswitch.js v2.0.1 (AudioLabs Erlangen, MIT) into `demo/` and generate one static page per hand-edited
config: `demo/{base,sdedit}/config.yaml` → `scripts/build_demo.py` → `index.html`, one player JSON per section, and
excerpts cut from `artifacts/*/listen` packs — re-level-matched after cutting (a 90 s window of a full-length-matched
pack can drift between tracks by fractions of a dB), written as flac with one spectrogram strip each.

- **`base`** — the curated set: input / master / apollo / picked sdedits per (source × bitrate). New methods land
  here, and this is the page that gets published.
- **`sdedit`** — the instrument: every variant of every pack, one player per family, created only when its section
  is expanded so hundreds of tracks stay openable.

## Consequences

- Local for now, served with `python3 -m http.server 8880 -d demo`. Publishing `base` to GitHub Pages is a later,
  explicit step; a README cannot run the players, so it will link the page.
- Generated media (flac excerpts and spectrogram PNGs) stays out of git; page code, configs and player JSONs are
  committed.
- The excerpts are what the pages compare: the −14 LUFS gate holds for exactly what is heard, but a verdict about a
  track's ending needs the window moved in the config first.
