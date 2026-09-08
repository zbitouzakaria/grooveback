# 11. SDEdit experiment 2: theta decoupled from the mix, cfg 1, heavy damage

Date: 2026-09-08

## Status

Accepted

## Context

Experiment 1 (ADR-0009's sweeps across all variants) settled several things
by measurement and ear: steps are not a quality knob (the post-trained
family's pingpong sampler re-rolls rather than refines; the base family is
converged at 50 — s100 lands within ~40 dB of s50), the post-trained
family's per-step renoising is a built-in hallucination source at every
noise level, `medium-base` has by far the most faithful encode/decode
anchor, and cfg 7 with an empty prompt extrapolates toward the embedding of
the empty caption — the mean of the model's training music, which is not
this library. At 128/192 kbps the method differences were subtle by ear.

## Decision

One model, heavier damage, three sweeps — replacing experiment 1's renders,
packs and demo content entirely.

- **`medium-base` only**, steps 50. Its euler sampler draws no noise of its
  own, cfg is meaningful there, and six-minute files sit inside its trained
  window.
- **`cfg_scale` 1.0 for unprompted runs** — no steering toward the mean
  caption. A cfg 7 retry on the winning unprompted setting is explicitly
  future work.
- **`theta` decoupled from `noise_level`.** Classic SDEdit ties the
  schedule's starting time to the amount of noise mixed in; the fork adds
  `init_mix_level` (github.com/zbitouzakaria/stable-audio-3, branch
  `init-mix-level`, ~6 lines over the previous pin) so the two separate.
  `noise_level=0` with `theta>0` is the damage-as-noise variant: nothing is
  added, and the model spends θ of denoising budget on what already deviates
  from clean music — the codec damage. Deterministic given the input.
- **Grids** (two sweeps, deliberately not a factorial — cells with
  θ < noise are predictably under-denoised): classic
  n ∈ {0 (round-trip anchor), 0.06, 0.15, 0.25, 0.5}; theta
  t ∈ {0.06, 0.10, 0.15, 0.20} at mix 0. θ is calibratable: encoding a
  master and its twin gives σ\* = RMS(Δlatents), the damage's own size on
  the schedule's scale, measured per bitrate before sweeping.

  Measured (medium-base autoencoder, A100 run of 2026-09-08): σ\* = 0.80 /
  0.54 on aerofunk 32k/64k and 0.71 / 0.57 on codec 32k/64k, against master
  latent RMS 1.15 / 1.02 — the damage sits far above the planned θ ceiling
  of 0.20, so the θ sweep was extended with calibrated points
  {0.35, 0.55, 0.80} in the same run. σ\* reads the encoder's whole latent
  displacement, which includes round-trip idiosyncrasy on top of audible
  damage, so it is an upper bound on the θ that matters; where between 0.20
  and σ\* the useful correction lives is what the extended sweep asks.
- **One prompted variant**, cfg 7 (its trained regime), at n ∈ {0.15, 0.25}:
  positive prompt biased to electronic production and top-end energy,
  negative prompt naming the damage ("muffled, dull, low quality, low
  bitrate mp3, bandlimited") so guidance points from the damage toward the
  target sound. Judged mainly on aerofunk — the codec asset is not
  electronic material.
- **Damage**: 32 and 64 kbps LAME twins of the two sources (codec 6 s,
  aerofunk 360 s cut). The subtle tiers taught little by ear.
- **Anchors in every listening pack**: input, master, apollo, and a2sb
  (cutoff pinned at the measured codec edge, run_xp convention), all
  level-matched in one set.

## Consequences

- The project now depends on its own stable-audio-3 fork; the pin moves with
  it. The patch is additive (defaults reproduce upstream behavior exactly).
- Experiment 1's renders are deleted, not archived — every render is
  regenerable from its seed and config.
- Findings land here or in a successor once the packs are listened to.

## Revisit triggers

- The θ variant beats classic SDEdit → σ\*-calibrated θ becomes the default
  restoration mode and the posterior-sampling work builds on it.
- Upstream stable-audio-3 moves in a way the fork wants → rebase the branch.
