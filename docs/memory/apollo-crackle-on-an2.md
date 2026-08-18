---
name: apollo-crackle-on-an2
description: "RESOLVED: Apollo chunk-seam crackle fixed by discard-mode chunking, confirmed by ear 2026-08-18"
metadata: 
  node_type: memory
  type: project
  originSessionId: d5c93125-9700-44bb-967a-f3423d0ebd90
  modified: 2026-08-18T09:09:10.242Z
---

Listening on 2026-08-15, Apollo's output for "AN-2 — Moonshine (Deep Boogie Version)" carries small punctual
"grésillements" — faint intermittent crackles, like a player glitching briefly. Not continuous. To be written up in
ADR-0005's results.

**Why:** This is the first real listening finding on the Apollo baseline and it is the kind of detail that gets lost
between sessions.

**How to apply:** CORRECTION (2026-08-18): the earlier claim that chunked and unchunked output are bit-identical
outside the crossfade regions is wrong. Measured on a 30 s AN-2 section: they differ *everywhere* at ~-30 dB RMS
(max abs 0.37 outside overlaps, 0.66 inside) — Apollo's output is context-dependent globally, so "match the unchunked
render" is not a correctness target and unchunked full-track is impossible anyway (63 s already OOMs MPS at 16 GB).
The seam mechanism stands: Apollo synthesises the band above the codec cutoff, two chunks realise different phase
there, and the 1 s crossfade blends incompatible realisations. Fix implemented 2026-08-18: discard-mode chunking in
`chunked()` (context thrown away, single realisation everywhere, ~10 ms joins), now `run_apollo`'s default at 12 s
window / 2 s context. **Confirmed by ear 2026-08-18: the crossfade render still has the artifact, the discard render
does not. Closed.** Full findings in ADR-0007. Instrument lesson: the crackle was clearly audible yet invisible to
click detectors and z-score metrics — spectral instruments are weak artifact detectors; the ear decides.
