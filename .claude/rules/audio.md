# Audio comparisons

These decide whether a result is real. Breaking any of them produces a confident wrong conclusion rather than an
error.

- **Level-match to −14 LUFS before any comparison**, with one common headroom gain across the set. Loudness
  dominates informal comparison; without this you are measuring gain.
- **Judge a latent-space method against `decode(encode(x))`, never against the input file.** The autoencoder is not
  transparent and invents content where the input is empty, so scoring against the input credits the autoencoder's
  behaviour to the method.
- **Save residuals before level matching**, so they are the honest difference.
- **Energy is not perception.** Band deltas catch regressions. Blinded listening on monitoring hardware decides, and
  it has already overturned an energy-based reading once.

## Correctness gates

Audio breaks silently: no exception, nothing looks wrong, the output is merely worse. Tests target that specifically.

- an identity model through the full inference path must return the input — catches chunking and overlap-add bugs
- every evaluation run also scores the untouched input and the clean reference; if the reference does not win, the
  harness is broken rather than the model
- loudness is asserted before any comparison is scored
- property tests on invariants: sample rate and channel count preserved, no NaN, no clipping introduced

An identity model is coherent by construction, so none of these can catch a model that invents content *differently
on each call*. That class needs a fake model returning random content in a band, then asserting flat band power
across seams — add it before chunking anything generative.

## Findings live in ADRs

`docs/decisions/` holds what we have learned and the evidence for it. Read ADR-0004 (the thesis), ADR-0006
(baselines) and ADR-0007 (the latent space) before proposing a change of method — they record what was already tried
and why it was rejected. Do not restate their conclusions anywhere else; two copies means one goes stale, and the
stale one is the one that gets read.
