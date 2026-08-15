# 4. Restoration as a latent inverse problem with a generative prior

Date: 2026-08-15

## Status

Accepted

## Context

The goal is an unconditional generative prior over minimal and electronic music, trained on my own library, living in
the latent space of a pretrained audio autoencoder. The prior is the artifact. Restoration of degraded rips is its
first application. Unconditional generation falls out for free. Conditioning on inspiration samples is a later and
more open problem.

## Decision

### Restoration is an inverse problem, solved at inference time

A degraded rip is an observation of a clean track through some operator:

```
y = A(x) + n
```

Restoration means sampling from `p(x | y) ∝ p(y | x) · p(x)`.

The two halves come from completely different places, and that is the whole design:

- `p(x)`, the prior, is learned once from clean audio only. It never sees a degraded example.
- `p(y | x)` is not learned at all. It is computed at inference time from an assumed or estimated `A`.

So changing what damage gets restored means changing `A`, not retraining. The prior is the only expensive thing, and
it is trained once.

Concretely, learning `p(x)` means training a network to remove noise of a known magnitude. A network that can do this
at every noise level has implicitly learned the gradient of the log-density of the data — the *score*. That gradient is
exactly what posterior sampling needs: add the gradient of the log-likelihood and the same machinery samples `p(x|y)`
instead of `p(x)`.

The prior works on autoencoder latents, not waveforms. The autoencoder stays frozen.

### The prior is adapted from Stable Audio 3, not trained from scratch

`stable-audio-3-small-music` is open weights, stereo, and pairs with the SAME autoencoder. It is fine-tuned on the
library, and sampled with null text conditioning to give the unconditional prior.

Why adapt rather than train:

- The library is about 215 hours, roughly 6,500 two-minute windows. That is thin for training a generative model from
  noise, and ample for specialising one that already knows what music is.
- A model trained for classifier-free guidance is trained with the conditioning randomly dropped, so its
  null-conditioned branch is a real unconditional score. Using it alone is the intended use, not a trick.
- Genre focus is what makes this tractable solo, and specialising a base model is exactly what fine-tuning does well.
- It leaves budget for the part that is actually uncertain — whether test-time posterior sampling produces convincing
  restoration — rather than the part with a known answer.

From-scratch training on SAME latents stays a fallback. It would only make sense at a much smaller model size, since
the library does not get bigger when the fallback is invoked.

### No supervised spectrogram baseline

Apollo already provides that ceiling as a pretrained checkpoint. Training our own would cost weeks, and the model, the
loss and the training loop would all be thrown away in the move to a latent prior. What survives is infrastructure, and
that is cheaper to build around a frozen model.

### The output is plausible, not faithful

MP3 masking, bandwidth truncation and analog transfer noise destroy information irreversibly. Nothing here recovers it.

What comes out is a sample from the posterior: structurally coherent with the input and convincing on good playback,
but not faithful to the original master. Where the observation says nothing, the prior invents. For listening at home
to records that cannot be re-sourced, that is the right trade.

## Consequences

- The degradation chain is an evaluation instrument and part of the inference-time likelihood, not a training
  distribution.
- Adapting Stable Audio 3 fixes the latent space to SAME. Real loss of optionality, accepted: the alternative is
  choosing a latent space freely and having no pretrained prior in it.
- A prior cannot be moved between latent spaces. Changing autoencoder means retraining, which is why the autoencoder
  is checked before any adaptation work.
- Adapted weights inherit the [Stability AI Community License](https://stability.ai/license) — fine for the personal
  and research scope, not unencumbered. Project code stays MIT.
- Inference is posterior sampling, not a forward pass. Anyone reading the codebase needs to know that.

## Revisit triggers

- **The base checkpoint produces incoherent output.** Fatal to this route; fine-tuning on 215 hours will not repair a
  prior that cannot generate. Output that is coherent but stylistically wrong is expected, and is the fine-tuning case
  rather than a trigger.
- **The fine-tuned prior is still stylistically wrong.** Then from-scratch becomes the question, at a smaller size.
- **Test-time sampling proves impractical.** The branch is conditional fine-tuning on degraded pairs, which reuses the
  same prior and infrastructure.
- **Commercial use becomes a possibility**, which makes the licence binding.
- **A stronger open music prior is released.** This decision is three months downstream of the release that enabled it.
