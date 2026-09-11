# 13. HP-codecX as a token-LM bandwidth-extension baseline

Date: 2026-09-11

## Status

Proposed — wired; renders, measurements and the listening verdict pending

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
the expected layout.

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

## Measurements before trust

To be recorded from the run:

- **The framing's fingerprint** (32 kbps render, per-band correlation and
  energy against the twin — the ADR-0012 instrument): verbatim below
  5.5 kHz, silence between 5.5 and 8 kHz, invention above 8 kHz. This
  three-band signature is what a correct integration must show; its absence
  means a wrapper bug, not a finding.
- **Alignment and level**: `best_lag(render, input) == 0`, and the kept
  band's energies against the twin's — the verbatim low band should anchor
  both.
- **Length behavior**: whether the 180 s source passes through the token LM
  whole; no limit is documented. If it fails, the recorded contingency is
  wrapper-side segmentation, gated per audio.md's stochastic-chunking rule
  before any long render is trusted — this model is seedless, so segment
  consistency cannot be assumed.

## Results

Pending: the five ADR-0007 metrics and fill-band decomposition for the six
packs, and the monitor listening verdict on the `demo/xp` page.

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
