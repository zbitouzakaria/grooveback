# 6. Baseline findings: Apollo and A2SB on library material

Date: 2026-08-17

## Status

Proposed — listening sections open, one MPS diagnostic pending

## Context

This ADR records what actually happened when the first two baselines of
[ADR-0005](0005-baselines-and-prior-viability-on-real-library-material.md) ran on real material. It exists so later
decisions argue with evidence rather than memory. Everything here is reproducible: the A2SB pipeline was validated
bit-exact against NVIDIA's own inference before any measurement was trusted, and every number below comes from
level-matched audio (−14 LUFS, common headroom gain).

Test material:

- **AN-2 — Moonshine** (7.6 min, 128 kbps MP3 rip, mono-summed for A2SB comparability). Bandwidth knee at 15.75 kHz —
  the typical damage profile of the library.
- **codec_wav** (6 s, Apollo's own test asset, content ends ~5 kHz; also brick-walled at 4 kHz for A2SB). The
  severe-damage case.

## Findings — Apollo

- **Restores strongly across the board**: +10–15 dB above 16 kHz on AN-2; ~+48 dB across 5–22 kHz on the 4 kHz-cut
  codec material. Below the cutoff it changes nothing (< 0.1 dB) — it is honestly surgical.
- **Fast and local**: 1.3–2× faster than realtime on the MacBook (MPS), 16.5M parameters, native stereo. The stereo
  image is preserved (8–16 kHz inter-channel correlation 0.31 vs source 0.33).
- **Conservative exactly at the cutoff seam**: on AN-2 the first synthesized band sits ~3 dB below the natural
  spectral trend, an 8.7 dB step where the real rolloff is ~2.7 dB/kHz. The muffled-regression signature, mildly.
- **It overshoots peaks**: +2.4 to +2.9 dBFS after level matching, on every input tried. Any listening chain needs the
  common-headroom gain (`evaluation.level_matched_set`) or it clips.
- **Unresolved artifact**: faint intermittent crackles ("grésillements") heard on the full stereo AN-2 render.
  Suspected chunk-seam phase incoherence (chunked overlap-add at 9 s hops; measured up to 0.58 absolute sample
  disagreement inside overlap regions between chunked and unchunked passes). Untested hypothesis: render unchunked
  and listen.

## Findings — A2SB

- **It is trained exclusively on brick walls.** `UpsampleMask` zeroes whole FFT bins; real codec rolloffs smear over
  ~1 kHz. Handed a cutoff at the rolloff's *foot*, the model reads the taper as natural spectrum and does nothing
  (measured: cutoff 4750 Hz → silence above; 4000 Hz → full reconstruction, same file). Input must be brick-walled at
  the *knee* — automated in the fork's `restore.py` (`input_prep.py`).
- **The released checkpoints matter audibly.** The paper's headline is an unreleased 4-partition ensemble. Of the
  public weights, the 1-split checkpoint paints a flat ~−45 dB shelf across the extended band; the 2-split ensemble
  rolls off like music. 2-split is the only defensible configuration.
- **At the seam it is smoother than Apollo** (1.3 dB step vs 8.7 dB on AN-2) but rolls off faster above 18 kHz —
  A2SB is closer to a natural continuation, Apollo commits more air. Which is *right* is a listening question.
- **Mono.** `librosa.to_mono` on load. Stereo material comes back mono-in-stereo; running channels independently was
  measured to invent stereo width (correlation 0.99 → 0.56) and was rejected.
- **Cost**: on an RTX 4090 the full-width regime does not fit (OOM at 22 GB even at batch 8). On an A100 80 GB at
  batch 32, a full 7.6-minute track runs in roughly 13 minutes (~$0.35). On the MacBook it is hours, and see below.
- **Stochastic across platforms**: per-step re-noising draws on-device RNG, so CUDA and MPS produce different
  realizations at the same seed (5–10 dB band variance in invented content on codec_wav). Sample-level equality only
  exists within a platform; cross-platform comparisons are shape-level.
- **Trained cutoff range is 2–16 kHz**, so ~16 kHz MP3 truncation sits at the edge of its competence; the severe
  (≤ 8 kHz) cases are its home ground.

### Cross-model comparison (AN-2 full track, mono, level-matched, dB FS per 1 kHz band)

| band | before | Apollo | A2SB 2-split (CUDA, single-pass) |
|---|---|---|---|
| 15–16 kHz | −59.8 | −59.4 | −59.5 |
| 16–17 kHz | −88.5 | −68.1 | −68.0 |
| 17–18 kHz | −124.5 | −70.5 | −72.2 |
| 18–19 kHz | −125.3 | −71.5 | −75.7 |
| 19–20 kHz | −124.3 | −73.6 | −79.8 |
| 20–21 kHz | −122.5 | −73.6 | −92.4 |
| 21–22 kHz | −124.0 | −76.4 | −97.3 |

![codec comparison](assets/0006/codec_comparison.png)

![an2 comparison](assets/0006/an2_comparison.png)

## Finding — the MPS full-width failure

The paper's intended inference (whole file, single pass, internal multidiffusion windowing) **silently corrupts on
Apple Silicon** at full track width, while short inputs are healthy. Two symptoms, one platform: the 2-split ensemble
converges to a no-op in the masked band (output at the input's noise floor); the 1-split paints a decoupled flat
shelf. The same commands on CUDA (A100) are healthy and match the wav-segmented workaround within 1–3 dB per band.

The validation chain that made every comparison trustable:

```
NVIDIA script ══ bit-exact ══ old integration ══ bit-exact ══ fork ══ bit-exact ══ fork-via-wrapper
                                  (codec_wav, 0.00e+00 at every stage, seed-locked)
```

Behaviour by platform and depth (masked-band prediction `|x0|`, full-track width):

| | iter 0 (t=1.0) | iter 1 (t=0.53) | iter 19 output |
|---|---|---|---|
| MPS | 0.144 healthy | 0.102 | **erased to noise floor** |
| CUDA | 0.144 healthy | 0.103 | **0.196, natural rolloff** |
| 30 s control (both platforms) | 0.140 | 0.091 (coarse-stride dip, universal) | healthy at 20 steps |

What eliminated everything else:

| suspect | eliminated by |
|---|---|
| wrapper / config | bit-exact chain: NVIDIA's script == old integration == fork == fork-through-wrapper (0.00e+00) |
| window bookkeeping (unfold/refold) | stub-model test exact to 0.0 at all widths on MPS |
| memory fix (keep-last-prediction) | bit-identical output pre/post fix |
| schedule / stride | coarse-schedule dip reproduced identically on CUDA (0.1441→0.1031) and MPS (0.1436→0.1022) |
| width per se | step-0 prediction healthy at full width on MPS (0.1436 vs healthy 0.1396) |
| model / checkpoints | CUDA full-width 20-step trajectory healthy: 0.144 → 0.196, stable through the late-model handover; 1-split builds hot but coherent (plateau 0.255) where MPS gave a decoupled shelf |

What remains: the corruption **compounds across iterations** on MPS at full width — healthy at iteration 0, identical
to CUDA at iteration 1, erased by iteration 19. Pending: an instrumented 20-step MPS replay to locate the divergence
iteration. Upstream-report material (pytorch MPS), not a grooveback problem to solve.

Two measurement lessons bought during this investigation, both now fixed and regression-tested:

- `band_energy_db` lied by ~60 dB on full-track input (float32 FFT roundoff at 20M samples) — it briefly turned
  "no-op" into "impossible silence". Instruments get validated against a second instrument (Spek) before their
  readings are believed.
- A crash-looping declared-healthy pipeline is worse than a failing one: every silent failure in this campaign
  (missing telemetry, masked exit codes, unfiltered stock listings) cost a run.

## Listening verdicts (Zakaria — to fill)

Level-matched packs in `artifacts/listen/`; CUDA renders in `artifacts/cuda/`.

- **codec set** (2026-08-17, monitoring headphones): **Apollo preferred, and the earlier verdict survives the
  ensemble fix** — "it at least tries to invent things, and is brighter. A2SB fills in missing frequencies minimally
  and it's audible — I hear some hats, but it's very shy in adding anything else." On the instruments themselves:
  "Apollo sounds way more musical — it adds a full hi-hat; A2SB sounds more like a click." The regression model
  reconstructs the *instrument*; the bridge model reconstructs *energy at the transient* and stops.
- **AN-2 set** (2026-08-17, monitoring headphones): **everything sounds the same** — before, Apollo, and both A2SB
  renders indistinguishable on headphones. To retry on the monitoring speakers. If it holds, it matters: at a
  15.75 kHz cutoff, top-octave-only restoration may be barely audible at all, and the audible battleground is the
  severely band-limited material (codec-class, ≤8 kHz) — which reframes how much the typical-library damage profile
  actually needs a restorer, and sharpens the case for the DSP control (ADR-0005 step 3).
- **Apollo crackles** — still audible on the unchunked render? TODO (open)

## Consequences

- **Apollo is the standing baseline** for the library's damage profile: fast, stereo, strong above the cutoff, runs
  locally. Its seam conservatism and peak overshoot are the reference flaws to beat.
- **A2SB is a second opinion, not a workhorse**: mono, ~50–100× Apollo's cost, CUDA required for full tracks — but
  its seam behaviour is the benchmark for "natural continuation", and its brick-wall lesson (models inherit their
  training corruption's *shape*) directly informs the degradation-chain design to come.
- Full-track A2SB inference happens on RunPod (A100, `restore.py`, volume `18v73b8ggl` in US-MO-1); the Mac runs
  everything else. MPS is not trusted for long multidiffusion sampling until the pending diagnostic says otherwise.
- ADR-0005's remaining steps (DSP control, chained baselines, SAME round-trip, SA3 probe) proceed unchanged; nothing
  found here reorders them.

## Revisit triggers

- Zakaria's listening verdicts land — they, not the band tables, decide what "beating the baselines" means.
- The MPS divergence trace localizes the failure — then an upstream issue with a minimal reproducer.
- NVIDIA releases the 4-partition ensemble, which would change A2SB's ceiling.
- The DSP control (ADR-0005 step 3) reframes how much of this restoration was tonal balance all along.
