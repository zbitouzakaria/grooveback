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

Vendor trackswitch.js (AudioLabs Erlangen, MIT; the docs-site build — the v2.0.1 release predates its documented
config schema) into `demo/` and generate one static page per hand-edited config: `demo/{base,sdedit}/config.yaml` →
`scripts/build_demo.py` → `index.html` (player configs inline) plus one spectrogram strip per track. A pack with a
configured window is cut and re-level-matched as one set (a window of a full-length-matched pack can drift between
tracks by fractions of a dB) and written as flac; a pack without one serves its full tracks, symlinked straight from
`artifacts/*/listen` behind a gate proving the set is still level-matched.

- **`base`** — the curated set: input / master / apollo / picked sdedits per (source × bitrate), on 90 s windows.
  New methods land here, and this is the page that gets published.
- **`sdedit`** — the instrument: every variant of every pack at full length, one player per family, created only
  when its section is expanded; at most one player stays decoded at a time.

## Consequences

- Local for now, served with `python3 -m http.server 8880 -d demo`. Publishing `base` to GitHub Pages is a later,
  explicit step; a README cannot run the players, so it will link the page.
- Generated media (flac excerpts and spectrogram PNGs) stays out of git; page code and configs are committed.
- What each page compares is exactly what its config says: `sdedit` plays whole tracks; `base` plays its 90 s
  windows, so a verdict there about a track's ending needs the window moved first.
