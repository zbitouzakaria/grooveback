# 7. SAME latent space: round-trip cost and the transport-vector floor

Date: 2026-08-18

## Status

Proposed — listening verdicts open

## Context

ADR-0004 puts the prior in the latent space of a pretrained autoencoder, and ADR-0005 requires checking that
autoencoder on real material first. SAME is the one Stable Audio 3 generates into, so the choice comes with the prior.

Two questions:

1. What does a round-trip through the space cost?
2. Is MP3 damage a constant offset in that space? If it is, `decode(encode(x) + v)` is the cheapest restoration
   imaginable and becomes the floor every later method must beat.

Material: 12 s excerpts at 1:00 and 3:00 of each source, one fixed rule. A clean master (Aerofunk), a real 128 kbps
YouTube rip (AN-2), a vinyl transfer of that same track, and a 6 s clean asset whose content stops near 5 kHz as the
severe case. Everything level-matched to −14 LUFS before comparison.

The vinyl transfer is new and matters: it is the first real clean reference for a real rip. It runs 0.0125% fast, so
it drifts about 45 ms against the rip across the track. `grooveback.align` fits speed and offset together; after
correction the two agree to 12 samples, with 8 samples of residual scatter — a pure speed error, no wow or flutter
worth modelling.

## Round-trip

Both variants; there is no SAME-M. Stable Audio 3 small generates into SAME-S.

- **Alignment is sample-exact**, so residuals are a plain subtraction.
- **Transparent below 12 kHz** on clean material, within a few tenths of a dB.
- **Residuals are loud** — 10 to 14 dB below signal, worst on the severe case at 5.6 dB — because the decoder
  re-realises phase rather than reproducing it. They are listening material, not a score.
- **SAME-L is better everywhere** by 1.3 to 3.5 dB, at roughly 100x the compute. It runs on a rented GPU in seconds
  and takes minutes per clip locally.

### The decoder invents a top end

The AN-2 rip is dead above 16 kHz. Its round-trip comes back with content there: **+22.6 dB for SAME-S, +11.9 dB for
SAME-L** in the 16–20 kHz band. Nothing asked for it, and the variant Stable Audio 3 uses is the more aggressive of
the two.

The vinyl transfer confirms this is invention rather than recovery. Run the vinyl through the same round-trip and the
band barely moves (+1.0 dB, +1.4 dB) — because the vinyl actually has content up there. The decoder only fabricates
where the input is empty.

**Consequence:** `decode(encode(degraded))` is not a no-op, so it is the reference any latent-space method is judged
against. Score against the input file and the autoencoder's own behaviour gets credited to the method.

## Transport vector

One 256-dim vector per bitrate: the mean of `z_clean − z_degraded` over 32 held-out-excluded windows of Aerofunk. The
whole method is `decode(encode(x) + v)`.

| | ‖v‖ | variance explained | direction agreement | worst window |
|---|---|---|---|---|
| same-s 128k | 1.37 | 31% | 0.93 | 0.64 |
| same-s 192k | 0.58 | 28% | 0.94 | 0.75 |
| same-l 128k | 1.61 | 23% | 0.92 | 0.29 |
| same-l 192k | 0.76 | 17% | 0.91 | 0.34 |

Direction is consistent; magnitude is not. The 192k vector points the same way at roughly half the length in both
spaces, which is what you would want if this tracked bitrate rather than fitting noise.

**SAME-L's space is the less linear of the two** — it explains less of the damage with a constant, and its worst
window agrees far less. Reconstruction quality and "damage is a simple direction here" are separate properties, and
the better autoencoder is worse on the second. That is worth remembering when a latent space is chosen for a method
that assumes structure.

### What it does, and the two problems it exposes

16–20 kHz, relative to the clean reference, SAME-S at 128k:

| excerpt | input | round-trip, no shift | + shift | ceiling |
|---|---|---|---|---|
| aerofunk_60s | −12.0 | −8.3 | −4.1 | −5.7 |
| aerofunk_180s | −11.3 | −14.7 | −10.7 | −7.9 |
| codec | −8.7 | −12.6 | −9.9 | −3.1 |
| an2_60s | −21.9 | −3.1 | +1.5 | −0.2 |
| an2_180s | −13.5 | −7.6 | −3.1 | −1.1 |

`ceiling` is `decode(encode(clean))`: where a *faithful* reconstruction lands. It is a calibration point, not an upper
bound — the encoder is not the decoder's inverse, so a better latent may exist, and band energy is not a distance
anyway. Overshooting it means more energy than accuracy would put there.

Two problems are visible here that the earlier synthetic-only run hid:

**On the real rip, the autoencoder does nearly all the work and the shift overshoots.** AN-2 at 60 s goes from −21.9
to −3.1 on the round-trip *alone*, then the shift pushes it to +1.5, past the −0.2 ceiling. The vector is adding
energy, not restoring structure. Without the vinyl reference this looked like the vector closing a gap; it is mostly
the hallucination of the previous section.

**At 192 kbps the autoencoder destroys more than the codec does.** Aerofunk at 192k arrives only 2.7 dB down, and
comes back from the round-trip 10.5 dB down. The ceiling is −7.9. So for lightly damaged material the latent space is
the dominant loss and no method operating inside it can help — it starts behind where it began.

## Consequences

- The evaluation harness needs `decode(encode(x))` as a standing condition wherever latent-space methods are scored.
- The autoencoder bounds everything, and on lightly damaged material it *is* the damage. Latent-space restoration is
  worth pursuing for heavily degraded input and is counterproductive for good input.
- Any method proposed from here is measured against the transport vector, not against the degraded input.
- Experiments needing SAME-L run on a rented GPU. The full transport run took 111 seconds on an L4 and would have
  taken hours locally.

## Open

Stable Audio 3 small is gated on Hugging Face. The repository documentation says it pairs with SAME-S; confirming from
the model config needs the licence accepted, which also gates all later work with the prior.

## Revisit triggers

- Listening contradicts the energy story — particularly whether the shift sounds restorative or merely bright.
- A second clean source changes the transport picture; the vector is fitted on one track.
- The Stable Audio 3 config pairs with something other than SAME-S.
