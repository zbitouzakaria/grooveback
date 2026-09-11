# 14. SonicMaster under controlled conditions

Date: 2026-09-12

## Status

Proposed — rendered, gated and scored; the listening verdict is pending

Supersedes the SonicMaster note in
[ADR-0005](0005-baselines-and-prior-viability-on-real-library-material.md).

## Context

ADR-0005 excluded SonicMaster
([Melechovsky et al., arXiv:2508.03448](https://arxiv.org/abs/2508.03448))
after one session on its hosted demo, where it duplicated and stacked kick
drums on electronic material. That is the record's only exclusion resting on
an uncontrolled instrument — unknown mode, settings and preprocessing — and
this run replaces it with a benchmark-grade measurement.

The framing is honest about expectations: SonicMaster's five enhancement
groups (equalization, dynamics, reverb, amplitude, stereo) do not include
codec or bandwidth damage, by the paper's own accounting. Its role on this
board is therefore the **learned mastering floor** — the successor to the
DSP control ADR-0005 defined ("the first question asked of any output is
whether it beats an EQ") — and a controlled answer to the kick-stacking
observation, for which the aerofunk source is the natural probe.

## Decision

### Environment

The release (github.com/AMAAI-Lab/SonicMaster, Apache-2.0, clone pinned at
`c4c0869`) pins torch 2.4.0, transformers 4.44.0 and diffusers 0.30.0,
conflicting with the project's torch 2.7.1: it runs from its own venv behind
a subprocess, built by `scripts/sonicmaster_setup.sh` in one resolution.
The README's `python==3.13` cannot hold for torch 2.4.0 (no cp313 wheels);
the venv uses 3.11. The 863 MB `model.safetensors` comes from
`amaai-lab/SonicMaster` on Hugging Face, size-checked (the ADR-0013
lesson); inference additionally pulls the Oobleck VAE from the gated
`stabilityai/stable-audio-open-1.0`, so `HF_TOKEN` with that license
accepted must be in the environment at run time — SonicMaster is a
TangoFlux derivative generating in the Stable Audio Open latent space.

### The wrapper

The release's `infer_single.py` is already a one-command file-in/file-out
entry point and runs unmodified; `baselines.run_sonicmaster` is the leanest
wrapper of the set — the model is natively 44.1 kHz stereo, so there is no
resampling, no per-channel splitting, and no level correction (the output
level is the model's own mastering decision, scored as emitted; listening
packs re-match as always). Long input is chunked by the release itself:
30 s windows with a 10 s overlap, each chunk conditioned on the previous
output's tail and joined by a linear crossfade — and `inference_flow` runs
seeded (default 0), so the model is deterministic given its input and the
crossfade satisfies audio.md's consistency requirement in principle; the
gates below measure it anyway. The release clamps each decoded chunk to
full scale.

**The automatic mode is the empty instruction.** Training dropped 10 % of
prompts to the empty string and replaced another 10 % with generic
mastering sentences ("Master this track for me, please." among four
others), so the unconditioned behavior is trained, not accidental; the
benchmark arm passes the empty prompt, and the generic sentences remain
available for listening-time probes through the Hydra entry point
(`model=sonicmaster model.prompt="..."`). Knobs left unset are not passed,
so the release defaults hold (10 inference steps, guidance 1.0, seed 0).

`sonicmaster` joins the benchmark anchors on all rungs, and
`model=sonicmaster` the Hydra entry point.

## Measurements before trust

Recorded from the L4 run (2026-09-12):

- **Determinism**: the codec 64 kbps twin rendered twice is bit-identical —
  the seeded flow reproduces exactly, which also makes the release's
  crossfaded chunking consistent-given-input by construction (the audio.md
  requirement) rather than by assumption.
- **Memory**: the release's VAE pre-encode batches ten 30 s chunks at once
  and OOMs a 22 GiB L4 on a 180 s file; the wrapper passes
  `--vae_batch_size 2` — the encode is the VAE's deterministic mode, so the
  batch size cannot change the output (the CoDiCodec situation of ADR-0011).
- **Alignment**: the single-chunk codec renders sit at lag 0. The aerofunk
  renders best-fit at ≈ −30 ms (−1,325 samples, constant across bitrates),
  but compensating that shift makes BSS-SDR *worse* (−31.6 against −25.1 at
  64 kbps): the render's waveform does not track the master at any lag. The
  lag is the least-bad alignment of regenerated content, not a repairable
  offset.
- **Level**: it masters loud — output peaks pinned at exactly 0 dBFS (the
  release clamps each decoded chunk) and roughly −10 LUFS on four of six
  packs; the codec source at 128 kbps came back quieter instead
  (−16.5 LUFS). Recorded as the model's mastering decision, not corrected.
- **The kick question, quantified**: in the 30–120 Hz band on aerofunk at
  64 kbps, the twin carries 149 onsets per minute and the render 205 —
  +38 % — with no dominant zero-lag envelope correlation (peak 0.19 near
  +80 ms). The model adds low-frequency transients that are not in the
  music: ADR-0005's kick-stacking observation, reproduced under controlled
  settings and measured.
- **Cost**: the codec packs render in about a minute each on an L4; a 180 s
  stereo pack ≈ 4 minutes. The whole run stayed under $0.5 of rented GPU.

## Results (2026-09-12 run)

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
| sonicmaster | 0.3 | 1.3 | -0.5 | 4.8 | 37.9 |

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
| sonicmaster | 0.1 | -1.0 | -1.2 | 1.7 | 30.5 |

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
| sonicmaster | -0.0 | 2.2 | -1.5 | 5.4 | 22.1 |

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
| sonicmaster | -25.1 | -3.0 | -26.6 | -1.4 | 19.8 |

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
| sonicmaster | -25.0 | -2.6 | -26.6 | -1.1 | 18.9 |

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
| sonicmaster | -24.2 | -1.9 | -25.2 | -0.4 | 15.7 |

What the metrics say, ahead of listening:

- SonicMaster regenerates rather than processes: waveform metrics collapse
  on every pack (BSS-SDR ≈ 0 on the codec source, −24 to −25 on aerofunk,
  where the added transients dominate), while LSD stays near the untouched
  input's — spectral character without waveform structure.
- Against the mastering-floor question it does not earn the floor: no
  column on any pack improves on the untouched input, and Apollo dominates
  it everywhere.
- The controlled run therefore *strengthens* ADR-0005's exclusion rather
  than overturning it: the kick-stacking is real, quantified, and appears
  in the automatic mode with no prompt at all.

## Listening verdict

Pending: the monitor session on the `demo/xp` page (track 8), with the
added kicks as the named check.

## Revisit triggers

- A release trained with codec/bandwidth degradations among its groups.
- The listening verdict overturns the kick-stacking observation → the
  instructed mode becomes worth a systematic pass.
