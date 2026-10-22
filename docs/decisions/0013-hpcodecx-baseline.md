# 13. HP-codecX: evaluated and excluded

Date: 2026-09-11, concluded 2026-09-12

## Status

Accepted — concluded: excluded from the benchmark. This record exists so the
model is not re-tested; the full integration is preserved in
[PR #21](https://github.com/zbitouzakaria/grooveback/pull/21), closed
unmerged.

## Context

HP-codecX ([Giniès et al., arXiv:2511.21580](https://arxiv.org/abs/2511.21580))
frames bandwidth extension as next-token prediction over a
harmonic/percussive codec, with a framing fixed by construction: the input
is defined as content to 8 kHz at 16 kHz rate (paper §3), the codec's
branches split at 8 kHz (§3.1.1), and the generated band is added to the
input's own upsampled low band (§3.2.1). It was the last open-weights
release in the survey without a "Tested here" mark, and was wired vanilla
into the benchmark on all rungs — the release's `predict.py` unmodified
behind its own venv, with two artifact-level repairs (the Zenodo
`torch.package` checkpoints froze pre-release classes and carry
constructor kwargs from an older signature; both were converted to the
weights format the current code loads, kwargs taken from the release's own
conf) and segmentation past the transformer's measured 5,000-token (50 s)
table.

## What was measured

- **The framing's fingerprint** behaved as §3 predicts: the render is the
  input verbatim below the 8 kHz split (per-band correlation 1.000), the
  codec-dead 5.5–8 kHz band at 32 kbps stays untouched, and at 64 kbps the
  twin's real 8–11 kHz is deleted and regenerated 9 dB quieter. Fed a
  32 kbps band ending at 5.5 kHz where training always supplied 8 kHz,
  generation nearly collapses (~−75 dBFS).
- **Temporal alignment is not at fault**: the generated band's energy
  envelope co-modulates with the music at the same lag structure as the
  master's own top band — no systematic offset anywhere — at roughly half
  the master's envelope coherence on the codec source and a quarter on
  aerofunk.
- **The deciding defect is stereo**: the codec is mono by architecture —
  every encoder branch begins with a 1-channel convolution
  (`hpcodec/model/codec.py`), and nothing models interchannel structure —
  while its nucleus sampler has no seed. Every possible integration
  therefore draws the two channels' top bands independently: measured
  interchannel correlation of the generated band is 0.04–0.14 against the
  masters' 0.69–0.99. The result reads, by ear and on per-channel
  spectrograms, as an incoherent texture cloud detached from the mix.
- On the scoreboard it was the only method sitting at the waveform
  do-nothing floor rather than below it (its low band is the input), with
  the only positive fill spectral SNR among the generative fills — but the
  fill is too quiet to matter, and Apollo dominated every LSD column.

## Decision

Excluded. A mono, seedless generator cannot produce an interchannel-
coherent stereo top band under any vanilla integration, and the target
material is stereo; a mid-summed variant (generate once from the downmix,
share the band across channels) was designed but not pursued — the fill it
would make coherent is negligible on this material to begin with.

## Revisit triggers

- The authors release a stereo-capable or seedable variant.
- A variable-cutoff or 44.1/48 kHz-input variant appears, making the 64 and
  128 kbps rungs meaningful.
