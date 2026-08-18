---
name: same-decoder-invents-hf
description: "SAME's decoder hallucinates ~30 dB of top end on bandlimited input; round-trip is never a no-op"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a40108c-c6aa-4b86-b4a1-81c77b27918c
  modified: 2026-08-18T09:00:01.900Z
---

Measured 2026-08-18 (ADR-0007): a 128k rip dead above 16 kHz (−83..−104 dBFS) round-trips through SAME-S with
−63..−73 dBFS of invented, musically plausible content up there. SAME-L invents less (+12 dB at 16–20k vs +23 for S).
Both variants also add energy above 20 kHz on clean input. Alignment is sample-exact (lag 0) in every test.

**Why:** encode→decode of degraded audio is not a no-op, so `decode(encode(degraded))` — not the degraded file — is
the honest reference for any latent-space restoration comparison. The AE is itself a mild blind bandwidth extender,
uncontrolled, and SAME-S (the SA3-small variant) hallucinates more than L.

**How to apply:** Every latent-space eval needs the round-trip as a standing condition. When judging any method's HF
recovery, compare against the AE ceiling `decode(encode(clean))` (−3.3 dB at 16–20k on our material), not against
clean itself. See [[transport-vector-floor]].
