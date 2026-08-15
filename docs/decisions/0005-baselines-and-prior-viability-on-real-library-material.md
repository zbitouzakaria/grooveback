# 5. Establish baselines and prior viability on real library material

Date: 2026-08-15

## Status

Accepted

Supersedes [3. Run Apollo on target distribution and analyse results](0003-run-apollo-on-target-distribution-and-analyse-results.md)

## Context

Every remaining decision — solver, degradation modelling, evaluation, whether to fine-tune at all — is better made
after hearing what the existing free tools do to actual tracks from the library. Writing those ADRs first would be
speculation.

## Decision

Run the following on real library material, in this order, before writing any further ADR.

1. **Apollo and A2SB**, as two baselines rather than one. Apollo targets codec artifacts; A2SB targets missing
   bandwidth. These rips have both. Apollo needs chunking to fit the local machine; A2SB handles long inputs directly.
2. **The two chained**, in both orders.
3. **A DSP control** — gain match, high shelf, mono below ~120 Hz. This is the floor. From here on, the first question
   asked of any output is whether it beats an EQ, and the answer tells us how much of the damage is tonal balance
   rather than codec loss.
4. **SAME round-trip.** Encode and decode library tracks, listen against the input. Not a comparison between
   autoencoders — SAME is fixed by the Stable Audio 3 pairing. The question is whether it handles this material, and
   the cases to listen to are hats, rides and reverb tails. Restoration never leaves the latent space, so whatever the
   autoencoder loses is lost.
5. **Stable Audio 3 probe**, on the base checkpoint before any fine-tuning. Sample unconditionally, and inpaint a
   masked two-second region of a library track. Inpainting is the cheapest possible test of the prior: the operator is
   exactly known and the answer is audible immediately.

This also settles three things published sources report inconsistently, and which are deliberately not asserted
elsewhere until checked: the checkpoint's parameter count, its sample rate, and whether it decodes through SAME at all.
The last one is what ADR-0004 rests on.

### How it is judged

Level-matched to −14 LUFS before any comparison, without exception. Then by ear on monitors, and by spectrogram.
Objective metrics are not the instrument here.

### Not done yet

No alignment of the real clean/rip pairs, so no paired metrics — the pairs have arbitrary offsets and are often
different masters, and that is second-wave work. No degradation chain, no solvers, no fine-tuning.

### SonicMaster

Tested on electronic material via its Hugging Face demo. It duplicates and stacks kicks, including under an explicit
negative prompt. Not adopted. Recorded so it does not get re-tested later.

## Consequences

- Every later method has something to beat, established before it is designed.
- The two components the design depends on are checked before any GPU budget is spent on them.
- Work stops here until results exist.
- Running other people's checkpoints on a 16 GB machine has its own friction. Accepted; smaller than training
  anything.

## Revisit triggers

- **A baseline is already good enough**, or **the DSP control closes most of the gap.** Either narrows the project's
  value sharply and is worth knowing now.
- **SAME's round-trip is audibly damaging on hats, rides or reverb tails.** This bounds everything downstream, and
  because Stable Audio 3 is tied to SAME it puts the prior source back in question too. The most consequential
  possible outcome here.
- **The probe produces incoherent output, or the checkpoint does not decode through SAME.** Escalates per ADR-0004.
- **The probe succeeds well**, in which case fine-tuning may not be needed for a first attempt.
- **Apollo or A2SB will not run locally**, which moves the baseline to a rented GPU.
