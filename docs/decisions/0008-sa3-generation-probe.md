# 8. Probe whether Stable Audio 3 generates master-like audio

Date: 2026-09-02

## Status

Accepted

## Context

The prior will be adapted from Stable Audio 3 (ADR-0004), and ADR-0005's last
open step is probing the base checkpoints: can they generate audio with the
production quality of a clean master? A generated track has no ground truth,
so nothing can be scored against a reference directly.

The probe turns the MP3 codec into the measuring instrument. A generated clip
is its own reference: compress it, score the compression against it with the
benchmark's five metrics (ADR-0007), and compare that cost with what the same
compression removes from a real master — the degraded-input rows of the
benchmark. A clip that loses *less* to MP3 than a master does was missing the
content MP3 normally eats: air, transient detail. The known confound: sparse
or dull real music also compresses easily, so the probe bounds rather than
proves, and listening stays the judge.

## Decision

- `grooveback.priors` wraps the `stable-audio-3` in-process API: `load_prior`
  / `generate`, all six released model types with their family's sampling
  settings (post-trained: 8 unguided steps; `-base`: 50 steps at cfg 7).
- `scripts/run_probe.py` generates the same track from every type — one fixed
  prompt, seeds 0–2, 12 s scoring clips plus a 45 s listening clip — makes
  MP3 twins at 64/128/192 kbps, and scores twin against clip. Twins are
  refused unless aligned; on dense noise-like clips the correlation probe can
  misread the lag, so alignment is judged by the match at lag zero.
- `notebooks/probe.ipynb` colours by distance to the aerofunk master anchor —
  raw best/worst would praise the dullest clip, since the easiest track to
  compress wins SDR.

## Results (A100 run, 2026-09-02)

What MP3 @ 64 kbps removes, means over 3 seeds; the anchor is what it removes
from the real master. Reading: lower SDR = more was removed; LSD near the
anchor = master-like top end.

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| small-music | 13.9 | 12.6 | 12.4 | 14.3 | 19.0 |
| small-music-base | 12.0 | 10.3 | 9.9 | 11.3 | 18.4 |
| small-sfx | 28.3 | 23.3 | 27.9 | 24.1 | 12.4 |
| small-sfx-base | 17.5 | 13.8 | 13.5 | 14.5 | 13.6 |
| medium | 18.5 | 16.9 | 18.0 | 18.8 | 7.6 |
| medium-base | 26.8 | 21.3 | 26.1 | 22.3 | 6.6 |
| *aerofunk master* | *19.4* | *17.6* | *18.1* | *19.2* | *17.1* |

The same ordering holds at 128 and 192 kbps (`artifacts/probe/results.json`).

Three observations, weighted for what three seeds of one prompt can carry:

- **The music models' output loses about as much to MP3 as a real master
  does** — 10–22 dB against the master's 17.6 at 64k. There is
  codec-relevant substance in what they generate. This is the robust
  reading.
- **The sfx types compress far too easily** (23.3 / 13.8 vs 17.6) — the
  confound demonstrated on genuinely sparse output; they are controls, not
  candidates.
- The finer patterns — `small-music` losing more than the master, `medium`
  paying far less LSD — are single numbers from minimalist clips, and SDR is
  relative to each clip's own content, which differs per variant. Recorded,
  not concluded from; the listening packs decide.
- All 24 generations rendered without failure — the six checkpoints load and
  sample through `grooveback.priors` on CUDA with the flash-attn wheel.

### Timing (NVIDIA A100-SXM4-80GB, measured)

Load is download + load on a cold cache; generation is timed after a warmup
call. Sampling is nearly duration-independent — the model produces the whole
sequence in one pass — so a clip's length barely matters:

| | load (cold) | 12 s clip | longest clip | max length |
|---|---|---|---|---|
| small-music | 62 s | 0.53 s | 0.66 s (110 s) | 120 s |
| small-music-base | 44 s | 3.15 s | 3.24 s (110 s) | 120 s |
| small-sfx | 46 s | 0.52 s | — | 120 s |
| small-sfx-base | 45 s | 3.17 s | — | 120 s |
| medium | 82 s | 0.80 s | 1.12 s (370 s) | 380 s |
| medium-base | 79 s | 4.80 s | 5.94 s (370 s) | 380 s |

A full 7-minute track (420 s) exceeds every checkpoint's maximum length, so
it takes at least two calls: `medium` in two windows costs about 2 s of
compute after the 80 s load (base: ~12 s); the small models need four
~110 s windows for under 3 s (base: ~13 s). Compute is not the obstacle —
musical continuity across windows is: consecutive windows are independent
draws, and chaining them coherently (the API's `inpaint_audio` /
`init_audio` are the available tools) is untested here.

### The length cap, and two ways past it

The cap is soft. `generate()` clamps the requested duration to its
`sample_size` argument (`min()` in `_adapt_sample_size`), whose default is
120 s; the checkpoint's `model_config.json` carries the trained length
(120 s small, 380 s medium) but nothing enforces it in-process — pass a
larger `sample_size=` and the model runs past it. The remaining trained
envelope is the `seconds_total` conditioner, calibrated 0–384 s on every
checkpoint. Note the clamp is silent: any call past 120 s without an
explicit `sample_size` quietly returns a shorter clip.

Both routes to 8 minutes work mechanically on `medium` (A100, measured):

- **Forced single call** — `duration=480, sample_size=490*SR`: 480 s in
  3.3 s of compute, 26% past the trained length, energy and top band stable
  across the whole track.
- **Continuation by inpainting** — 380 s, then a second window keeping the
  last 60 s as context (`inpaint_audio`, mask regenerating everything after
  it — continuation was a training task) and 100 s new, crossfaded inside
  the context: 480 s in 4.2 s total, equally stable.

Which one sounds right is a listening question —
`artifacts/probe/eight/listen/` holds both, level-matched. The winning
route is what a `generate_long` in `grooveback.priors` should implement.

## Consequences

- The generation plumbing the prior work needs exists and is exercised:
  every released checkpoint, in-process, seeded, numpy out.
- The probe alone does not establish master-grade output either way; what
  it establishes is the pipeline and a repeatable yardstick, with the master
  anchor as the behaviour a fine-tuned prior should approach. It closes
  ADR-0005's checklist.
- Three seeds and one prompt bound how far these numbers generalise; the
  listening packs under `artifacts/probe/*/listen/` carry the perceptual
  verdict.

## Revisit triggers

- Fine-tuned checkpoints exist → rerun the probe; the anchor to approach is
  the master row.
- The probe's verdict contradicts listening.
