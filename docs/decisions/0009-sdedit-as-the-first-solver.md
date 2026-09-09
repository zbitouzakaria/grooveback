# 9. SDEdit as the first solver

Date: 2026-09-07, concluded 2026-09-09

## Status

Accepted, and concluded: the method is rejected as restoration. This record
absorbs the former ADR-0011 and keeps the findings.

## Context

The baseline phase (ADR-0005 → 0008) measured the existing tools. Apollo is
a network trained on pairs: damaged audio in, repaired audio out, one pass.
Our prior (ADR-0004) is a different kind of model. It has only ever seen
clean music, and it *generates* it: starting from pure noise, it removes
noise over many small steps, following an internal clock that runs from 1
(all noise) to 0 (finished music). It knows nothing about MP3s.

SDEdit is the cheapest way to point such a generator at restoration — no
surrogate model, no fine-tuning — so it went first: to stand the whole
inference chain up, and to find out where the ceiling of this method is.

## The method, in plain terms

**Classic SDEdit.** Mix some noise into the damaged track and drop it into
the generator partway through its process. With noise amount n = 0.25 the
model is told "you are at time 0.25" and runs its usual steps down to 0.
Whatever the noise destroyed, it rebuilds — and it rebuilds it as the clean
music it knows. One number does two jobs here: how much of the track gets
destroyed, and how much rebuilding the model performs. To repair more
damage you must first destroy more of the track. That trade is built in.

**The theta variant.** Split the two jobs. Mix in no noise at all, but
still tell the model "you are at time θ" (say 0.35). It now spends 0.35
worth of rebuilding on a track that contains no added noise — so the only
thing it can treat as noise is whatever already deviates from clean music:
the codec damage itself. And because nothing random is added, the output is
a deterministic function of the input: run it twice, get the same file
(verified bit-identical).

The vendor's code cannot express this — one parameter drives both jobs — so
the project runs a small fork of `stable-audio-3` that adds
`init_mix_level`. One detail cost a day: at time θ the model expects the
track scaled down by (1 − θ), because that is how every training example
looked. The fork's first version skipped that scaling, so every θ render
handed the model a full-volume track at a time where it expected a quieter
one, and the results were incoherent. The fix is one word — scale by the
schedule's value, not the mix's — and with it θ renders are coherent.

## What was run

Two checkpoints of the medium model — "base" (50 careful steps, steerable
by prompts) and "inference" (the 8-step version the paper tuned for sound
quality) — on heavily damaged input: 32 and 64 kbps MP3 twins of two tracks
with known masters. Sweeps of n and θ, a text-prompted arm, an attempt to
fill the dead band by adding noise only above the codec cutoff, and apollo
and a2sb rendered on the same inputs as anchors. Everything ends in
level-matched listening packs behind the demo pages (ADR-0010).

## Findings

- **The prior keeps anything that is plausible music.** A track with no top
  end is plausible music. A track with hiss on top is also plausible music.
  So no setting ever fills the band the codec removed, and noise injected
  into that band comes back as noise. A generative prior only replaces what
  it can recognize as not-music, and almost nothing about codec damage
  qualifies.
- **The damage is large in the model's own units.** Encoding a master and
  its MP3 twin and measuring the distance (σ\*) gives 0.54–0.80 on a scale
  where 1.0 means all noise — far past the noise levels a track survives
  (n above ~0.25 already drifts audibly away from the input).
- **Prompt guidance is far too strong for a shortened run.** cfg 7 is
  calibrated for the full process from pure noise; on SDEdit's short tail
  it wrecks the track (input correlation 0.96 → 0.32) while adding energy
  everywhere. At doses that preserve the track, prompts change nothing.
- **Quality and faithfulness split by damage level.** On heavy damage the
  inference checkpoint sounds clearly better while following the input less
  (correlation ~0.90 against base's ~0.97). On the light damage of the
  first experiment, that same freedom read as hallucinated detail. Which
  prior fits depends on how much of the input deserves keeping.
- **Verdict: rejected.** Compared with apollo or with the untouched input,
  every configuration hallucinates too much to be an improvement.

## Consequences

- Kept: the solver and Hydra harness (a new method is one config group and
  one function), the stable-audio-3 fork, the anchored listening packs,
  and these findings.
- Next: a solver that carries an observation constraint — the prior
  proposes full-band music while being held to "your lowpass must match the
  input" (the DPS/DAPS/LOUDAR line, per ADR-0004). That is the only shape
  in this family that can invent the missing band without permission to
  invent everywhere. Its first design question is posed by the findings:
  the easier model to constrain (base) is not the better-sounding one
  (inference).

## Revisit triggers

- A fine-tuned prior exists → the same sweeps become a cheap re-test.
- Upstream stable-audio-3 moves → rebase the fork.
