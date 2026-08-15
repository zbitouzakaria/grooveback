# 4. Restoration as a latent inverse problem with a generative prior

Date: 2026-08-15

## Status

Accepted

## Context

grooveback began as a supervised restoration project: train a spectrogram U-Net on clean audio passed through a
synthetic degradation chain, and iterate toward generative methods later. ADR-0002 and the original README encode that
plan.

The goal has changed. What is wanted is an **unconditional generative prior over minimal and electronic music**,
trained on a personal library, living in the latent space of a pretrained audio autoencoder. Restoration of degraded
rips is the first application of that prior, not the object of the project. Unconditional generation falls out of the
prior for free. Conditioning on inspiration samples — a vocal, a loop, a reference texture — is a later and separately
open problem, likely inpainting-shaped at first.

This inverts the dependency structure. A supervised restoration model is trained to invert one specific degradation
chain and is worth nothing when the chain changes. A prior is trained once on clean audio and knows nothing about
degradation at all; the degradation enters only at inference time. That makes the prior reusable across every
application, and makes the degradation chain a swappable component rather than the foundation.

Two facts about the current landscape make this tractable for one person on a few hundred euros of GPU:

1. **Pretrained audio autoencoders are good enough to work inside.** The prior does not operate on waveforms or
   spectrograms; it operates on latents produced by a frozen autoencoder. This cuts the sequence length by three orders
   of magnitude and is what makes a full-length track affordable to model at all.
2. **Stable Audio 3 shipped in May 2026 with open weights**, including a music diffusion transformer that already lives
   in exactly such a latent space, at 44.1 kHz stereo, with LoRA fine-tuning supported.

## Decision

### Restoration is an inverse problem, solved at test time

Model the degraded observation as

```
y = A(x) + n
```

where `x` is the clean audio we want, `A` is the degradation operator (codec loss, bandwidth truncation, analog
transfer colouration), `n` is noise, and `y` is the rip we actually have. Restoration is then sampling from the
posterior

```
p(x | y)  ∝  p(y | x) · p(x)
```

The two factors are sourced completely differently, and this is the central architectural fact of the project:

- `p(x)` — **the prior** — is learned once, from clean audio only, and never sees a degraded example. This is the
  artifact.
- `p(y | x)` — **the likelihood** — is not learned. It is computed at inference time from an assumed or estimated `A`.

Everything downstream follows from that split. Changing what damage we restore means changing `A`, not retraining.
Adding a new application (inpainting, generation, conditioning) means changing the likelihood term or dropping it,
not retraining. The prior is the only expensive thing, and it is trained exactly once.

### The prior is a diffusion model over autoencoder latents

Concretely, "learning `p(x)`" means training a denoising network. For anyone approaching this from a discriminative
background, the operational summary is: a diffusion model is trained to remove noise of a known magnitude from a noisy
input, and it turns out that a network which can do this at every noise level has implicitly learned the gradient of
the log-density of the data — the *score*, `∇ₓ log p(x)`. Sampling is then a walk from pure noise down that gradient.
The reason this matters for inverse problems is that the score is exactly the object needed to do posterior sampling:
add the gradient of the log-likelihood to the gradient of the log-prior and the same machinery now samples `p(x|y)`
instead of `p(x)`.

The prior operates in latent space rather than on waveforms. The autoencoder is frozen, and its latents are
precomputed once for the whole library so it never runs inside a training loop.

### The prior is sourced by adapting Stable Audio 3, not trained from scratch

`stable-audio-3-small-music` — open weights, 44.1 kHz stereo, SAME latents, roughly 0.4–0.6B parameters (sources
disagree; resolve by loading the checkpoint), Stability AI Community License, LoRA supported — is fine-tuned on the
personal library. Sampling with null text conditioning at CFG=1 then gives the unconditional prior.

The reasoning, since this is the decision most likely to be second-guessed:

- **The library is small for from-scratch training.** Roughly 2,000 tracks at a typical 6–7 minutes is about 215 hours,
  which at the SAME latent rate is only ~6,500 distinct two-minute windows. For calibration, LOUDAR trained a 68M
  parameter DiT for 375k iterations on 50 hours of *solo singing voice* — a far lower-entropy distribution than
  full-mix stereo house — and still reports prior-mismatch artifacts, where heavily degraded input acquires
  characteristics of the training corpus rather than of itself. A from-scratch prior on this library would very likely
  be worse than an adapted one at several times the cost.
- **Null conditioning is a genuine unconditional prior, not a trick.** Classifier-free guidance requires training the
  model with the conditioning signal randomly dropped, so the network has an explicitly trained null-conditioned
  branch. That branch *is* an unconditional score estimate; it is half of what CFG combines at sampling time. Using it
  alone with guidance scale 1 is the intended use of the object, not a workaround.
- **Genre specialisation is what fine-tuning does well.** The stated reason genre focus makes this tractable solo is
  that a narrow distribution is easier to model than a broad one. Fine-tuning delivers exactly that specialisation
  while the base checkpoint supplies the generic structure 215 hours cannot teach.
- **The later ambition comes nearly free.** Conditioning on inspiration samples was scoped as a separate, more open
  problem. Stable Audio 3 supports inpainting natively, which is the inpainting-shaped first step already built.
- **It preserves the budget for the part that is actually uncertain.** The open question is whether test-time posterior
  sampling produces convincing restoration, not whether a prior can be trained. Spending the entire GPU budget on the
  part with a known answer would leave nothing for the part without one.

From-scratch training on SAME latents remains a documented fallback, not a committed phase. Its trigger is recorded
under *Revisit triggers* below.

### There will be no supervised spectrogram U-Net baseline

The original plan called for one, to establish a ceiling before generative work. It is cut:

- The ceiling it would establish is already reported in the literature, and Apollo provides it as a pretrained
  checkpoint for the cost of running inference.
- Almost none of the code survives the transition to a latent generative prior — the model, the loss, the training
  loop, and the data pipeline are all discarded.
- What *does* survive is infrastructure: manifests, level matching, evaluation, listening. That infrastructure is
  cheaper to build around a frozen Apollo than around a model we also have to train.

Apollo and A2SB serve as the supervised and bridge-model baselines instead. See ADR-0005.

### The output is a plausible reconstruction, not signal recovery

MP3 psychoacoustic masking discards information by design. Bandwidth truncation and analog transfer noise destroy more.
None of it is recoverable in the signal-processing sense, and no method in this project will recover it.

What the system produces is a sample from the posterior: audio that is **structurally coherent with the input** and
**perceptually convincing on good playback**, but not faithful to the original master. Where the observation is
uninformative, the prior fills in — which is to say, the system invents plausible detail. For the use case (listening
at home to records that cannot be re-sourced) this is the right trade. For anything requiring provenance or fidelity
guarantees, it is not, and the project makes no such claim.

This has a direct methodological consequence: **waveform-level and spectral-magnitude metrics systematically penalise
this class of method**, because a plausible reconstruction differs from the reference in exactly the places the metric
measures. Evaluation must account for that. See ADR-0005 and the evaluation ADR to follow.

## Consequences

- The degradation chain stops being the central research artifact and becomes an evaluation instrument plus a component
  of the inference-time likelihood. The README's Real-ESRGAN framing — correct when the simulator *is* the training
  distribution — no longer applies and is removed. A dedicated ADR will address what the chain must be instead.
- The autoencoder is no longer an independent choice. Adapting Stable Audio 3 fixes the latent space to SAME. This is
  a real loss of optionality, accepted knowingly: the alternative is choosing a latent space freely and then having no
  pretrained prior in it.
- The prior cannot be transplanted between latent spaces. Changing autoencoder means retraining. This is why the
  autoencoder is verified before any adaptation work begins, rather than after.
- Licensing: the Stability AI Community License covers personal and research use, which is the stated and indefinite
  scope. It is not an unencumbered artifact, and this constrains any future commercial use.
- Anyone reading this codebase needs to understand posterior sampling to understand what it does. The inference path is
  not a forward pass. This is a documentation burden the project accepts, and the reason these ADRs explain mechanism
  rather than only recording choices.
- Success is not measurable by a single number. The judge is blinded listening on monitoring hardware; objective
  metrics catch regressions and nothing more.

## Revisit triggers

- **The adapted prior fails the viability probe in ADR-0005** — that is, the base or fine-tuned model cannot inpaint a
  masked region of a library track in a way that is structurally coherent and stylistically near the material. This is
  the trigger for from-scratch training on SAME latents.
- **Test-time posterior sampling proves impractical** — inference cost or instability makes it unusable in practice.
  The branch is then conditional fine-tuning on pairs from the degradation chain, in the StableSR / DiffBIR mould. That
  is a legitimate alternative, not a defeat, and it reuses the same prior and the same infrastructure.
- **Commercial use becomes a possibility**, which makes the Community License binding and forces an unencumbered prior.
- **A stronger open music prior is released.** The field moves monthly; this decision is three months downstream of the
  release that made it possible, and the next release may reset it.
- **The autoencoder round-trip in ADR-0005 proves inadequate on the target material** — specifically on transients and
  reverb tails — which would bound the achievable quality regardless of how good the prior is, since restoration never
  leaves the latent space.
