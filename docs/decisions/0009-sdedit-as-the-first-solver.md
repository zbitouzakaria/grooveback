# 9. SDEdit as the first solver

Date: 2026-09-07, concluded 2026-09-09

## Status

Accepted, and concluded: the method is rejected as restoration. This record
absorbs the former ADR-0011.

## Context

The baseline phase (ADR-0005 → 0008) measured the existing tools. Apollo is
a supervised model: trained on pairs of degraded and clean audio, it maps
one to the other in a single pass. The project's prior (ADR-0004) is a
generative model trained on clean music only. It synthesizes audio by
iterative denoising: starting from Gaussian noise in the autoencoder's
latent space, it removes noise over a sequence of steps indexed by a
schedule time running from 1 (pure noise) to 0 (finished audio). It has no
concept of MP3 compression.

SDEdit ([Meng et al., 2022](https://arxiv.org/abs/2108.01073)) is the
minimal way to apply such a generator to restoration: it requires no
surrogate model and no fine-tuning. It was selected as the first solver
with two objectives — establish the full inference chain, and measure the
method's limits on this material.

## Method

**Classic SDEdit.** The degraded track is mixed with Gaussian noise at
level n and handed to the generator as if it were an intermediate sampling
state at time n; the generator then runs its remaining steps down to 0.
Content destroyed by the noise is re-synthesized as the clean music the
prior models. A single parameter therefore controls two quantities at
once: how much of the input is destroyed, and how much re-synthesis the
model performs. Repairing more damage requires destroying more of the
input.

**Decoupled schedule start (theta).** The two quantities are separated: no
noise is mixed in (`noise_level = 0`), but the schedule starts at time θ,
so the model performs θ worth of re-synthesis on an input that contains no
added noise. The working hypothesis is that the degradation, being a
deviation from the clean-music distribution, is what the model will
attribute to noise and remove. This assumption is strong — nothing
guarantees the model treats codec artifacts as noise rather than as signal
content — and the experiments below test it directly. Because no
randomness enters, the output is a deterministic function of the input
(verified bit-identical across repeated runs on the base checkpoint).

**Implementation.** The vendored API drives both quantities from one
parameter, so the project pins a
[fork](https://github.com/zbitouzakaria/stable-audio-3) adding
`init_mix_level`, which sets the mixed-noise amount independently of the
schedule start. The input must additionally be scaled by (1 − θ): the
model's training states have the form (1 − t) · signal + t · noise, and an
unscaled input at a claimed time θ lies outside that family by a factor
growing with θ. An earlier revision of the fork omitted this scaling
(commit `3513afc`); all θ results below use the corrected revision
(`b0a90b9`), which reproduces upstream behavior exactly when
`init_mix_level` is unset. The solver returns output at the input's
integrated loudness and accepts whole files up to 360 s.

## Experimental setup

Two checkpoints of the medium model: *base* (50-step guided sampling,
deterministic given the starting state) and *inference* (the 8-step
post-trained variant, whose sampler draws fresh noise at every step).
Inputs: 32 and 64 kbps LAME twins of two sources with known masters
(codec, 6 s; aerofunk, 360 s), codec edges at 5.5 and 11 kHz. Sweeps of n
and θ; a text-prompted arm with classifier-free guidance; an arm with
band-limited noise injected above the codec edge; apollo and a2sb rendered
on the same inputs as anchors. All outputs are compared as level-matched
listening sets (−14 LUFS, one common gain per set; ADR-0010), and the
damage magnitude σ\* = RMS(Δlatents) between each master and its twin is
measured with the model's own autoencoder.

## Results

- **The prior retains any input that is plausible music.** A bandlimited
  spectrum is plausible music, and so is broadband noise above a cutoff:
  no configuration fills the band the codec removed, and injected band
  noise is returned as noise. The hypothesis behind the θ variant is
  refuted — codec damage is attributed to signal, not to noise.
- **σ\* measures the damage at 0.54–0.80** on the schedule's scale, where
  1.0 corresponds to pure noise. Classic SDEdit drifts audibly from the
  input above n ≈ 0.25, far below the level the damage would require.
- **Classifier-free guidance does not transfer to shortened schedules.**
  cfg 7, calibrated for sampling from pure noise, reduces input
  correlation from 0.96 to 0.32 on the [0.25, 0] tail while adding energy
  across the band. At scales that preserve the input, prompts produce no
  measurable or audible change.
- **Perceptual quality and input fidelity separate by damage level.** On
  heavy damage the inference checkpoint is clearly preferred by ear while
  tracking the input less (correlation ≈ 0.90 against base's ≈ 0.97); on
  the light damage of the first experiment the same behavior presented as
  hallucinated detail.
- **Verdict.** Against apollo and against the untouched input, every
  configuration hallucinates too much to constitute an improvement. The
  method is rejected on this material.

## Reproducibility

The sweeps are Hydra experiment configs (`configs/xp/`); each checkpoint
writes to its own output directory because filenames carry only the swept
value:

    uv run python -m grooveback.cli.run -m xp=sdedit-noise input=data/degraded output_dir=artifacts/sdedit/base
    uv run python -m grooveback.cli.run -m xp=sdedit-theta input=data/degraded output_dir=artifacts/sdedit/base
    uv run python -m grooveback.cli.run -m xp=sdedit-noise model.variant=medium input=data/degraded output_dir=artifacts/sdedit/inference
    uv run python -m grooveback.cli.run -m xp=sdedit-theta model.variant=medium input=data/degraded output_dir=artifacts/sdedit/inference

## Consequences

- Retained: the solver and Hydra harness (a further method is one config
  group and one function), the stable-audio-3 fork, the anchored listening
  packs, and these results.
- The next solver carries an observation constraint: the prior proposes
  full-band audio while each sampling step is held to consistency with the
  observed lowpass (the DPS/DAPS/LOUDAR line, per ADR-0004). Its first
  design question follows from the results above: the checkpoint that is
  easier to constrain (base) is not the one preferred by ear (inference).

## Revisit triggers

- A fine-tuned prior exists → the sweeps above are a cheap re-test.
- Upstream stable-audio-3 moves → rebase the fork.

## References

- Meng et al., *SDEdit: Guided Image Synthesis and Editing with Stochastic
  Differential Equations*, ICLR 2022.
  [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)
