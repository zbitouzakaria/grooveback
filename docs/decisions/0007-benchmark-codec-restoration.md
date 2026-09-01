# 7. Benchmark codec restoration with SDR and SI-SNR on MP3 twins

Date: 2026-09-01

Rewritten from scratch; the earlier transport-vector version of this record
lives in git history.

## Status

Accepted

## Context

ADR-0005 requires checking the SAME autoencoder on real material before any
prior work, and ADR-0006 left Apollo and A2SB as the standing baselines. What
was missing is one harness that scores all of them the same way against a
known clean reference.

Real rips have no clean reference, so the benchmark makes its own damage:
MP3-compress clean chunks and score every method against the original.

## Decision

One script (`scripts/run_xp.py`), metrics in `grooveback.evaluation`, one
notebook (`notebooks/xp.ipynb`) showing one source at a time.

- **Sources**: one chunk each of two clean files — aerofunk (12 s at 1:00)
  and the codec asset (6 s whole). No rips, no vinyl, no alignment fitting.
- **Damage**: LAME MP3 at 64, 128 and 192 kbps, decoded back to wav. The
  round-trip is verified sample-aligned (`best_lag == 0`) before scoring,
  because one sample of shift wrecks a waveform metric.
- **Methods**: `decode(encode(x))` through SAME-S and SAME-L, Apollo, and
  A2SB. The untouched MP3 is scored too — the do-nothing floor every method
  must beat.
- **Metrics**: plain SDR and SI-SNR against the original — one scalar per
  render — plus spectrograms and level-matched listening sets. Plain SDR
  rather than BSS-eval SDR: no distortion filter and no extra dependency, so
  the numbers compare within this table but not against published ones.
- **SAME runs in-process.** `stable-audio-3` is a project dependency pinned
  to a commit, which pins torch to 2.7.1 for the whole project. The
  subprocess boundary remains only for A2SB, whose environment genuinely
  conflicts.
- Everything renders in one `run_xp.py` pass, on a GPU pod by default; every
  step is skipped when its output already exists.

## Consequences

- Waveform metrics punish phase re-realisation: SAME round-trips will score
  far below Apollo however they sound. The table ranks waveform fidelity,
  not perceived quality; listening stays the judge (audio.md).
- The torch pin follows stable-audio-3. Apollo runs on 2.7.1 — verified by
  the suite and a render — and anything that needs a newer torch now
  conflicts with the prior's stack.
- Synthetic twins are cleaner than real rips: no re-encodes, no unknown
  encoder chain. Results bound the easy case, not the library case.

## Revisit triggers

- Numbers need to be compared against a published table → add BSS-eval SDR.
- Later work needs a torch newer than the stable-audio-3 pin.
- A metric that is fair to phase re-realisation becomes necessary →
  log-spectral distance is the standard next step.
