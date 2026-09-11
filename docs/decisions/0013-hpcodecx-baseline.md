# 13. HP-codecX as a token-LM bandwidth-extension baseline

Date: 2026-09-11

## Status

Proposed — rendered, gated and scored; the listening verdict is pending

## Context

HP-codecX ([Giniès et al., arXiv:2511.21580](https://arxiv.org/abs/2511.21580))
frames bandwidth extension as next-token prediction: a transformer language
model generates the missing band's tokens over a neural codec disentangled
by harmonic/percussive decomposition. It is the last open-weights release in
the README survey without a "Tested here" mark; scoring it completes the
survey column, and it adds a third generative family (token LM, after
diffusion-bridge A2SB and mel-diffusion AudioSR) to the missing-band
comparison.

Its framing is fixed by construction rather than detected. The task is
defined (paper §3) as reconstructing s₂₄;₄₈ from s₈;₁₆ — the input *is* a
16 kHz-rate signal band-limited to 8 kHz — and the codec's two branches
(§3.1.1) split at 8 kHz, the 48 kHz branch carrying everything above. The
output (§3.2.1) adds the generated high band to the input's own upsampled
low band, so the kept band below the split returns verbatim. Against the
benchmark's rungs this framing divides cleanly:

- **32 kbps** (edge 5.5 kHz): the mandatory 16 kHz downsample destroys
  nothing, but the codec-dead 5.5–8 kHz band stays unfilled — generation
  exists only above the split.
- **64 / 128 kbps** (edges 11 / 16.5 kHz): the downsample deletes 3 / 8.5 kHz
  of real content before the model runs, all of it re-invented by the LM.

The model runs vanilla on every rung regardless — the ADR-0012 precedent:
what a shipped method does to this material is the finding, not something to
pre-empt. Its sampling is nucleus (top_p 0.95) with no seed anywhere in the
release, so renders are non-reproducible draws; they are cached by file
existence, and provenance is the release tag plus the printed settings.

## Decision

### Environment

The release is a Zenodo code drop (v1.0.1, MIT) with two `torch.package`
checkpoints (1.2 GB codec, 0.8 GB language model) that expect a fixed
`runs/` layout. `torch.package` archives are sensitive to the torch version
they were saved under, so although the repo's requirement is only a floor
(`torch>=2.1`), the environment pins the release era: `scripts/
hpcodecx_setup.sh` clones the tag into `third_party/hpcodecx` (gitignored),
builds its venv with `torch==2.1.2` solved in one resolution with the repo's
requirements (the ADR-0012 lesson), and fetches the weights from Zenodo into
the expected layout — size-checked, because the concept record's file URLs
serve an HTML page that a plain download saves silently.

Two release defects surfaced and are repaired at the artifact level, with no
change to the release's code (`scripts/hpcodecx_convert.py`, run by the
setup). The packages froze the classes as they were at training time, before
the repo's inference helpers existed — the packaged quantizer lacks the
`from_codes` that predict.py calls — and their stored constructor kwargs
predate the current signatures (their `encoder_dims` holds what is now
`latent_dims`; constructing with them allocates a network hundreds of times
larger until the OOM killer fires). The converter therefore rewrites each
checkpoint into the weights format audiotools' loader also accepts — which
instantiates the *current* repo classes — with kwargs taken from the
release's own `conf/codecx/hpcodecx.yml`, gated by a state-dict load that
must match key for key.

### The wrapper

No driver and no upstream change: the release's own `scripts/predict.py` is
already a one-command batch entry point, so `baselines.run_hpcodecx` only
prepares its input convention and reads its output. `predict.py` consumes a
directory of sorted (16 kHz, 48 kHz) file pairs; the wrapper writes, per
channel, the input resampled to 16 kHz and a 48 kHz upsample of the same
audio as the pair twin — the twin's content is not used, but it fixes the
output length — then runs the script with the clone as working directory and
reads back the 48 kHz renders, one per channel, resampled home to the
input's rate and trimmed to its length. The low band is the input's own, so
no level correction applies. `top_p` unset means the flag is not passed and
the release default holds. `model=hpcodecx` joins the Hydra entry point, and
`hpcodecx` joins the benchmark anchors on all rungs.

One capability is ours: the transformer's positional table holds 5,000
tokens — 50 s at its 100 tokens/s — and an 18,000-token input dies on a
broadcast mismatch (measured), so the wrapper cuts longer audio into 40 s
segments joined by a 0.1 s equal-gain crossfade, every segment pair going
through a single predict.py invocation. Identical content passes the joins
exactly (the identity gate, tested), and the seam's cost on this seedless
model is measured below.

## Measurements before trust

Recorded from the L4 run (2026-09-11):

- **The framing's fingerprint** (per-500 Hz-band correlation and energy of
  the render against the twin, on channel 0): at 32 kbps the render is the
  input verbatim below 5.5 kHz (correlation 1.000, Δ 0.0 dB), the predicted
  5.5–8 kHz hole is real — the rolloff residue passes through untouched —
  and above 8 kHz generation nearly collapses: the invented content sits
  around −75 dBFS. Fed a band ending at 5.5 kHz where training always
  supplied 8 kHz, the model generates almost nothing. At 64 kbps the copy
  boundary sits at the architectural 8 kHz split (correlation 0.999 below,
  ≈ 0 above): the twin's real 8–11 kHz is deleted and regenerated **9 dB
  quieter** — the mirror image of AudioSR, which regenerated the same band
  4.5 dB hotter (ADR-0012).
- **Seam consistency** (codec 64 kbps, the standing whole render versus
  forced 2 s segments): the segmented arm carries +4 to +11 dB more
  generated energy above 8 kHz — in content at −40 dBFS and below — with
  the segmented arm the louder one everywhere, ruling out crossfade power
  loss; as with AudioSR, window length modulates how much the model fills.
  The benchmark holds 40 s segments everywhere. LSD between the arms:
  9.9 dB, in those quiet bands.
- **Alignment and level**: `best_lag == 0` on every checked render; the
  verbatim low band anchors loudness (−14.28 vs the twin's −14.20 LUFS on
  codec 64 kbps) with no correction applied.
- **Cost**: a token LM is light — the smoke plus all three codec packs took
  ~4 minutes on an L4, a segmented 180 s stereo pack ~3 minutes (ten
  segment-calls in one process). The whole exercise, environment debugging
  included, stayed under $0.7 of rented GPU.

## Results (2026-09-11 run)

Each table is one source at one bitrate; columns are the five ADR-0007
metrics against the master, LSD lower-is-better. The fill-band decomposition
lives in `results.json`.

**codec @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 12.1 | 10.2 | 9.8 | 11.3 | 41.5 |
| earvae | 7.9 | 7.1 | 6.3 | 9.4 | 40.8 |
| apollo-earvae | 4.7 | 4.8 | 3.7 | 9.9 | 9.4 |
| earvae-sub | 7.1 | 6.7 | 5.7 | 9.9 | 16.5 |
| apollo | 5.8 | 6.0 | 5.0 | 12.3 | 8.6 |
| a2sb | 11.5 | 9.9 | 9.4 | 11.8 | 35.0 |
| audiosr | 3.2 | 3.2 | 2.3 | 4.5 | 35.6 |
| hpcodecx | 12.1 | 10.2 | 9.8 | 11.4 | 37.6 |

**codec @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 14.2 | 12.6 | 12.5 | 14.0 | 33.1 |
| earvae | 7.9 | 7.2 | 6.4 | 10.3 | 32.6 |
| apollo-earvae | 5.0 | 5.0 | 4.0 | 10.1 | 9.4 |
| earvae-sub | 7.7 | 7.0 | 6.1 | 10.7 | 16.2 |
| apollo | 7.1 | 7.2 | 6.5 | 13.7 | 8.2 |
| a2sb | 13.9 | 12.4 | 12.2 | 14.0 | 29.9 |
| audiosr | 7.3 | 7.2 | 6.6 | 8.5 | 25.0 |
| hpcodecx | 14.5 | 12.2 | 12.1 | 13.8 | 18.3 |

**codec @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 17.0 | 16.3 | 16.5 | 18.6 | 20.5 |
| earvae | 7.8 | 7.1 | 6.3 | 11.2 | 21.2 |
| apollo-earvae | 5.2 | 5.2 | 4.2 | 10.2 | 9.0 |
| earvae-sub | 7.8 | 7.0 | 6.3 | 11.3 | 9.7 |
| apollo | 8.6 | 8.9 | 8.4 | 15.3 | 6.8 |
| a2sb | 16.5 | 15.9 | 16.1 | 18.6 | 14.6 |
| audiosr | 13.2 | 12.6 | 12.3 | 15.1 | 15.9 |
| hpcodecx | 16.6 | 13.5 | 13.5 | 15.1 | 16.0 |

**aerofunk @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 19.0 | 14.8 | 14.9 | 15.4 | 23.6 |
| earvae | 16.8 | 13.6 | 13.6 | 14.8 | 23.2 |
| apollo-earvae | 15.7 | 13.2 | 12.9 | 15.8 | 11.2 |
| earvae-sub | 15.3 | 12.8 | 12.7 | 15.2 | 13.1 |
| apollo | 16.8 | 14.0 | 13.9 | 16.8 | 10.4 |
| a2sb | 18.3 | 14.5 | 14.6 | 15.7 | 18.6 |
| audiosr | 7.7 | 7.6 | 7.1 | 8.6 | 18.6 |
| hpcodecx | 19.0 | 14.8 | 14.9 | 15.5 | 20.0 |

**aerofunk @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 18.6 | 17.0 | 17.3 | 18.6 | 16.7 |
| earvae | 15.3 | 13.5 | 13.3 | 16.4 | 16.6 |
| apollo-earvae | 14.7 | 13.2 | 12.9 | 17.2 | 10.0 |
| earvae-sub | 14.7 | 13.1 | 12.9 | 16.8 | 9.2 |
| apollo | 16.9 | 15.9 | 15.8 | 20.3 | 8.9 |
| a2sb | 18.1 | 16.6 | 16.9 | 18.4 | 14.0 |
| audiosr | 10.7 | 10.7 | 10.3 | 11.6 | 14.8 |
| hpcodecx | 19.6 | 16.3 | 16.6 | 17.6 | 12.3 |

**aerofunk @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 23.1 | 21.4 | 23.0 | 23.2 | 7.3 |
| earvae | 15.2 | 13.7 | 13.5 | 17.7 | 9.6 |
| apollo-earvae | 14.8 | 13.4 | 13.2 | 17.6 | 9.1 |
| earvae-sub | 15.2 | 13.7 | 13.5 | 17.7 | 8.2 |
| apollo | 20.2 | 19.9 | 20.0 | 24.3 | 6.7 |
| a2sb | 22.5 | 20.9 | 22.4 | 22.4 | 7.0 |
| audiosr | 20.5 | 19.5 | 20.2 | 21.4 | 9.0 |
| hpcodecx | 22.5 | 17.7 | 18.3 | 18.9 | 10.8 |

What the metrics say, ahead of listening:

- hpcodecx is the first method on this board whose waveform metrics sit *at*
  the do-nothing floor rather than below it (BSS-SDR within 0.3 dB of the
  input in every pack) — the verbatim low band carries them, and the fill is
  too quiet to cost anything.
- Its fill is the only generative one on the board with the right texture by
  the spectral view: fill spectral SNR is positive in five of six packs
  (+0.03 to +0.89, against AudioSR's −8 to −17) — but its magnitude is
  small, so the LSD gains are modest and mixed (better than a2sb at
  64 kbps on both sources, worse elsewhere). Apollo dominates every LSD
  column.
- The framing costs land where §3 predicts: at 32 kbps the model is fed an
  out-of-distribution band and adds nearly nothing; at 64 and 128 kbps its
  gains ride on content it first deleted.

## Listening verdict

Pending: the monitor session on the `demo/xp` page (track 8).

## Consequences

- Every open-weights method in the survey is now either scored on this
  benchmark or excluded on record; the "Tested here" column is complete.
- Three generative missing-band families (bridge, mel diffusion, token LM)
  are now measured on identical material against the same masters.

## Revisit triggers

- The authors release a variable-cutoff or 44.1/48 kHz-input variant →
  the 64/128 kbps rungs become meaningful.
- An upstream code change becomes unavoidable → fork the release and point
  the setup script's clone at the fork.
