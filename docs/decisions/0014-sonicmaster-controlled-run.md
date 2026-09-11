# 14. SonicMaster under controlled conditions

Date: 2026-09-12

## Status

Proposed — wired; renders, gates and the listening verdict pending

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

To be recorded from the run:

- **Determinism**: the codec 64 kbps twin rendered twice — the seeded flow
  should make the renders identical; if not, the seam gate carries more
  weight.
- **Alignment and level**: `best_lag` against the twin (a VAE round trip
  may not preserve phase alignment) and the loudness delta — a mastering
  model may legitimately move both; measured and recorded, not corrected.
- **Seam**: aerofunk exercises the release's 30 s / 10 s crossfade path;
  band energies across the chunk boundaries checked for steps.

## Results

Pending: the five ADR-0007 metrics and fill-band decomposition for the six
packs — read against the mastering-floor question — and the monitor
listening verdict, with the kick-stacking check on aerofunk named as the
specific question this run exists to answer.

## Revisit triggers

- A release trained with codec/bandwidth degradations among its groups.
- The listening verdict overturns the kick-stacking observation → the
  instructed mode becomes worth a systematic pass.
