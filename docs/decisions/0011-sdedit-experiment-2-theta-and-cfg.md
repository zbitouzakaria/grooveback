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
- **One prompted variant**, at n 0.25: positive prompt biased to electronic
  production and top-end energy, optionally a negative prompt naming the
  damage. Judged mainly on aerofunk — the codec asset is not electronic
  material.

  The first pass at cfg 7 failed, and the failure is a finding: guidance
  strength is calibrated for a full schedule from pure noise, and on a
  truncated [0.25, 0] tail the same per-step push is a several-fold
  overdose — correlation with the input collapsed to 0.32 (unprompted at
  the same noise: 0.96) while the top band overshot the master. Corrected
  to a cfg sweep {2, 3.5}, with (`p`) and without (`q`) the negative
  prompt. At cfg 2 fidelity returns (0.94–0.96) but the prompt then adds
  almost no top-band energy; cfg 3.5 with the negative prompt still costs
  fidelity (0.75). Prompt guidance does not buy missing high-frequency
  energy at fidelity-safe doses on this material — the ears decide whether
  what little it does is worth keeping.
- **Damage**: 32 and 64 kbps LAME twins of the two sources (codec 6 s,
  aerofunk 360 s cut). The subtle tiers taught little by ear.
- **Anchors in every listening pack**: input, master, apollo, and a2sb
  (cutoff pinned at the measured codec edge, run_xp convention), all
  level-matched in one set.

### Band-noised inputs: tried, failed, removed (2026-09-09)

Listening found that both sweeps preserve the codec cutoff — a bandlimited
track is a plausible clean signal to the prior — so seeded band-limited
noise ({5, 10, 20, 40}% of track RMS above the cutoff) was added to the
inputs to destroy that evidence before the θ variant. The band-energy
numbers looked right only because the injection was calibrated to
master-like levels: by ear the model does not converge — the static
survives on top of the track. Renders removed. Two causes, recorded so
this is not retried naively:

- **A real bug**: the fork's first mixing patch scaled the init signal by
  `(1 - mix_level)` instead of `(1 - sigma_max)`, so every `theta`-set
  render fed the model a full-scale latent at a timestep where it was
  trained on `(1 - θ)`-scaled ones — off-distribution by exactly θ. Fixed
  in fork commit `b0a90b9` (the classic path was and stays bit-identical
  to upstream); every t-family render predates the fix and understates
  the variant.
- **The structural limit**: audio-domain noise reaches the latent as
  structured *content* (hiss is in-distribution music), not as the
  Gaussian latent noise the denoiser was trained to remove. SDEdit has no
  observation operator — no way to keep the observed band and invent the
  missing one. That constraint is the posterior-sampling line's job
  (DPS/DAPS/LOUDAR), which is the planned next solver.

### Listening verdict (2026-09-09, corrected-fork renders)

- **The inference model sounds clearly better than base on this heavy
  damage** — the post-training's perceptual optimization survives partial
  denoising. Measured signature: ~9 dB more mid-band (5.5–11 kHz) energy at
  correlation 0.90–0.92 versus base's 0.97–0.99. Together with
  experiment 1 (where the same renoising read as hallucination on light
  damage), the trade is damage-dependent: faithful prior for light damage,
  quality prior for heavy damage.
- At 32k, high-θ renders drift into coherent *other* tracks — real-sounding
  music loosely anchored to the input — while 64k stays conservative, as
  the surviving structure predicts.
- No SDEdit-family setting fills the dead band; the corrected θ sweep
  confirms it with proper scaling. The missing band remains the
  posterior-sampling solver's job — noting that constraining the preferred
  (8-step, stochastic) inference model is harder than constraining base,
  a tension the next solver design has to face.
- **Final verdict: no SDEdit-family setting is usable as restoration.**
  Compared against apollo or the untouched input, every configuration
  hallucinates too much to be an improvement. SDEdit work stops here; the
  keepers are the solver/harness plumbing, the fork, the listening
  instrument, and these findings. The next method is the
  posterior-sampling solver.

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
