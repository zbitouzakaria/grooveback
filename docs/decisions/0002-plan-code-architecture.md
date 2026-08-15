# 2. Plan code architecture

Date: 2026-08-15

## Status

Accepted

## Context

grooveback is starting from near-zero code. This ADR fixes enough structure for the work in
[ADR-0005](0005-baselines-and-prior-viability-on-real-library-material.md) and stops there. Anything past that gets
designed once there are results to design against.

## Decision

### Target tree

```
grooveback/
├── configs/
│   └── config.yaml
├── docs/decisions/
├── src/grooveback/
│   ├── audio.py           # io, resampling, loudness, spectrograms
│   ├── manifest.py        # the track list
│   ├── baselines.py       # Apollo, A2SB, DSP control
│   ├── latents.py         # SAME encode/decode
│   ├── priors.py          # Stable Audio 3
│   ├── evaluation.py      # level matching, comparison output
│   └── cli/
│       ├── ingest.py
│       ├── baseline.py    # run one method or a chain
│       ├── roundtrip.py
│       ├── probe.py
│       └── evaluate.py
├── notebooks/
├── scripts/
├── tests/
├── data/                  # gitignored
└── artifacts/             # gitignored
```

Modules for the degradation chain, solvers, datasets, training and inference are not here. They wait on ADR-0005.

### Grow flat, split later

Every module starts as a single file and becomes a package only when it outgrows one — roughly 300–500 lines. No empty
folders, no stub files: a file appears when it contains working code, so the tree above describes intent rather than
current contents.

### Dependencies

Each one answers why this and not the alternative.

| Package | Why |
|---|---|
| `torch`, `torchaudio` | Tensors plus audio io, resampling and STFT in one stack, so the same code runs on MPS locally and CUDA on a rented box |
| `numpy` | Interchange at the edges |
| `soundfile` | libsndfile bindings, where torchaudio's backend story has churned |
| `einops` | Readable tensor reshapes, which matters most where a silent axis swap is undetectable |
| `pydantic` | Manifest validation — the place a silent schema error corrupts everything downstream. Not for configs; Hydra already owns that |
| `pyloudnorm` | BS.1770 loudness, for the −14 LUFS matching every comparison depends on |
| `pedalboard` | The DSP control, at plugin quality |
| `matplotlib` | Spectrograms |
| `hydra-core`, `omegaconf` | Run configuration |
| `ffmpeg` (system) | Decoding whatever container a library file arrives in |

Not used: `librosa` (numpy-bound and pulls numba; torchaudio covers it and runs on MPS — kept in the `notebooks` group
for analysis), `audiomentations` (redundant with pedalboard), `pesq`/`pystoi` (speech metrics, wrong domain).

### Configuration

`configs/config.yaml` and nothing else for now. There is one run type — take some tracks, run a frozen model, produce a
level-matched comparison — so group composition would be scaffolding for runs that do not exist. Configs appear
alongside the code they configure.

### Testing

Domain logic stays free of infrastructure: `audio`, `manifest` and `evaluation` must be importable and testable with no
GPU, no network and no tracker.

Audio is a domain where a plausible diff silently makes the output worse without raising anything, so the tests target
that specifically:

- **Identity model through the full path** must return the input. Catches chunking and overlap-add bugs, which matters
  immediately because Apollo runs chunked.
- **Every evaluation run scores the untouched input and the clean reference.** If the reference does not win, the
  harness is broken rather than the model.
- **Loudness assertion** before any comparison is scored.
- **SAME round-trip** must clear a fixed threshold on fixtures. Catches wrong channel order, wrong normalisation, stale
  checkpoint.
- **Property tests** on the invariants: sample rate and channel count preserved, no NaN, no clipping introduced,
  deterministic under a fixed seed.

### Compute

Everything in ADR-0005 runs locally on Apple Silicon, with Apollo chunked to fit. Rented GPU starts at fine-tuning.

## Consequences

- The tree only describes work that is about to happen; navigating the repo needs this ADR plus ADR-0004 and ADR-0005.
- Very little Python exists, which is accurate.
- Some refactoring when files become packages. Accepted.

## Revisit triggers

- **ADR-0005's results land** — the deferred modules get designed then.
- **A second run type appears**, which is what justifies config groups.
- **A second model family or audio representation**, which may justify packages instead of files.
- **Any module passes ~500 lines.**
