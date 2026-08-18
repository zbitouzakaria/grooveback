# 7. SAME latent space: round-trip cost and the transport-vector floor

Date: 2026-08-18

## Status

Proposed — listening verdicts open

## Context

ADR-0004 puts the prior in the latent space of a pretrained autoencoder, and ADR-0005 requires checking that
autoencoder on real material before anything is built on it. SAME is the one Stable Audio 3 generates into, so the
choice comes with the prior.

Two questions, answered here:

1. What does a round-trip through the space cost?
2. Is MP3 damage a constant offset in that space? If it is, `decode(encode(x) + v)` is the cheapest restoration
   imaginable and becomes the floor every later method must beat.

Material: 12 s excerpts at 1:00 and 3:00 of each source — a fixed rule covering intro and main body. Aerofunk is a
clean master, AN-2 is a real 128 kbps YouTube rip with no clean version, and a 6 s clean asset whose content stops
around 5 kHz is kept as the severe case. Everything is level-matched to −14 LUFS before comparison.

## Round-trip

SAME comes in two variants, S and L; there is no M. Stable Audio 3 small generates into SAME-S.

- **Alignment is sample-exact.** No offset in any test, so residuals are a plain subtraction.
- **Energy-transparent below 16 kHz** on clean material, within ±0.6 dB per band.
- **Waveform residuals are loud** — 10 to 14 dB below signal, worst on the severe case at 5.6 dB. The decoder
  re-realises phase rather than reproducing it, so this is expected and is why the residuals are listening material
  rather than a verdict.
- **SAME-L is better everywhere**, by 1.3 to 3.5 dB, at roughly 500x the compute on this hardware.

### The decoder invents a top end

The AN-2 rip is dead above 16 kHz, at −83 to −104 dBFS. Its round-trip comes back with −63 to −73 dBFS of plausible
content up there. Nothing asked for it. SAME-L invents less than SAME-S, so the variant Stable Audio 3 uses is the
more aggressive of the two.

This has a direct consequence for how everything downstream is measured: **`decode(encode(degraded))` is not a no-op,
so it — not the input file — is the reference any latent-space method is judged against.** Otherwise the
autoencoder's own behaviour gets attributed to the method.

It also means part of the restoration we intended to build already exists inside the decoder, uncontrolled and
unasked-for.

## Transport vector

A clean master through a LAME encode and back gives paired latents. The mean of `z_clean − z_degraded` over 32
windows is a single 256-dim vector. The whole method is one addition before decoding.

The vector is fitted on Aerofunk windows only, with every excerpt below held out.

**How constant is the damage?** Windows agree strongly on direction (cosine 0.93 to the mean, minimum 0.64) but a
constant explains only ~30% of frame-level magnitude; the rest is content-dependent. The 192k vector points the same
way at half the length (0.58 vs 1.37), which is the behaviour you would want if this were measuring something real
about bitrate rather than fitting noise.

**Effect at 128k, 16–20 kHz, relative to the clean master:**

| excerpt | input | round-trip, no shift | **+ shift** | ceiling |
|---|---|---|---|---|
| aerofunk_60s | −12.0 | −4.5 | **−0.1** | −3.3 |
| aerofunk_180s | −11.3 | −10.4 | **−6.2** | −3.4 |
| codec (severe) | −8.7 | −11.1 | **−7.8** | −4.4 |

`ceiling` is `decode(encode(clean))` — the best any latent-space method can reach, since it never leaves the space.
On the first excerpt the shift lands at the ceiling; on the others it closes roughly half the gap. It also removes
the small broadband tilt the codec leaves below 16 kHz.

**On a real rip the vector still does something.** AN-2 has no clean master, so it is scored against its own
unshifted round-trip: adding a vector fitted on synthetic twins of a *different track* raises 16–20 kHz by ~4.8 dB on
both excerpts. That is the first evidence the direction generalises past the material it was fitted on.

**This is now the floor.** A 256-float vector applied with an add. Any prior-based method that cannot audibly beat it
is not earning its complexity.

Caveats: every number here is energy, and energy is not perception — the listening packs decide. The vector is fitted
on one track, so cross-track evidence rests on AN-2 alone. And the pool windows come from the same track as two of
the held-out excerpts, which is a weaker separation than a second source would give.

## Consequences

- The evaluation harness needs `decode(encode(x))` as a standing condition wherever latent-space methods are scored.
- The autoencoder bounds everything: whatever it loses is lost regardless of how good a prior is.
- Any method proposed from here is measured against the transport vector, not against the degraded input.
- SAME-L is not usable interactively on this hardware, so experiments run on SAME-S and L is a spot check.

## Open

Stable Audio 3 small is gated on Hugging Face. The repository documentation says it pairs with SAME-S; confirming that
from the model config needs the licence accepted on the account, which also gates all later work with the prior.

## Revisit triggers

- Listening contradicts the energy story.
- A second clean source changes the cross-track picture.
- The Stable Audio 3 config pairs with something other than SAME-S.
