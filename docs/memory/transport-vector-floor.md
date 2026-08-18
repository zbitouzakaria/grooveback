---
name: transport-vector-floor
description: The latent transport vector looked strong on synthetic pairs and mostly overshoots once a real reference exists
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a40108c-c6aa-4b86-b4a1-81c77b27918c
  modified: 2026-08-18T11:55:35.118Z
---

One 256-dim mean latent offset (`decode(encode(x) + v)`) fitted on clean/MP3 twins. Direction is consistent across
windows (cosine 0.93) but a constant explains only ~31% of magnitude in SAME-S and ~23% in SAME-L — **the better
autoencoder has the less linear damage direction.**

**Corrected 2026-08-18 by the aligned vinyl transfer of AN-2.** The earlier read — "recovers 16–20 kHz to the
autoencoder ceiling" — was measured without a real reference and was too generous. With the vinyl in place:
`an2_60s` at 128k goes −21.9 → −3.1 dB on the round-trip *alone*, then the shift pushes it to +1.5, past the −0.2
ceiling. The autoencoder's hallucination does nearly all the work and the vector adds energy on top. Separately, at
192 kbps the round-trip loses more (−10.5 dB) than the codec did (−2.7 dB), so latent-space methods start behind on
lightly damaged material.

**Why:** it is still the floor any prior must beat, but the honest framing is that it is a cheap brightener, not a
restoration. Do not cite the original ceiling-matching result.

**How to apply:** `scripts/run_experiments.py`, ADR-0007, listening packs in `artifacts/same/transport/`. Fitted on
one track (Aerofunk); AN-2 is the only cross-track evidence. Judge against `ref_decode_encode_input`, not the input
file — see [[same-decoder-invents-hf]].
