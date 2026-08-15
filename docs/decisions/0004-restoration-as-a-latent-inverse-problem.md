# 4. Restoration with a generative prior

Date: 2026-08-15

## Status

Accepted

## Context

The goal is a generative model of minimal and electronic music, trained on my own library, working in the latent space
of a pretrained audio autoencoder. That model — the prior — is the artifact. Restoring degraded rips is its first
application. Generating new music falls out for free. Conditioning on inspiration samples is a later problem.

## Decision

### The prior is trained on clean audio, and the damage is handled at inference

The prior learns what this music sounds like. It only ever sees clean audio, and knows nothing about MP3s or vinyl
transfers.

Restoration then works by asking it a question at inference time: of all the clean tracks this model finds plausible,
which ones could have produced the damaged file I have? The model proposes clean audio and gets steered toward
candidates that stay consistent with the input.

The useful part of that split is that changing what damage we handle doesn't mean retraining. The prior is the only
expensive thing, and it gets trained once.

### The prior is adapted from Stable Audio 3, not trained from scratch

`stable-audio-3-small-music` is open weights, stereo, and pairs with the SAME autoencoder. It gets fine-tuned on the
library.

- The library is about 215 hours. That is thin for training a model from nothing, and plenty for specialising one that
  already knows what music is.
- I want to focus on minimal and electronic music, and fine-tuning is how you specialise a model on a genre.

From-scratch training stays a fallback.

### No supervised spectrogram baseline

Apollo already gives that ceiling as a pretrained checkpoint. Training our own would cost weeks, and the model, the
loss and the training loop all get thrown away in the move to a latent prior.

### The output is plausible, not faithful

MP3 masking, bandwidth cuts and analog transfer noise destroy information for good. Nothing here recovers it.

What comes out is a plausible reconstruction: structurally coherent with the input, convincing on good playback, but
not faithful to the original master. Where the input says nothing, the prior invents.

## Consequences

- Adapting Stable Audio 3 fixes the latent space to SAME. A real loss of optionality, accepted — the alternative is
  picking a latent space freely and having no pretrained prior in it.
- A prior can't be moved to a different latent space; changing autoencoder means retraining. So the autoencoder gets
  checked before any adaptation work.
- Adapted weights inherit the [Stability AI Community License](https://stability.ai/license) — fine for personal and
  research use. Project code stays MIT.

## Revisit triggers

- **The base checkpoint produces incoherent output.** Fatal to this route. Output that is coherent but sounds wrong for
  the genre is expected, and is what fine-tuning is for.
- **The fine-tuned prior still sounds wrong for the genre.** Then from-scratch becomes the question.
- **A stronger open music prior is released.**
- **Commercial use becomes a possibility**, which makes the licence binding.
