# 2. Plan code architecture

Date: 2026-08-15

## Status

Accepted

## Context

grooveback is starting from near-zero code. This ADR fixes enough structure for the work in
[ADR-0005](0005-baselines-and-prior-viability-on-real-library-material.md) and stops there.

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

### Grow flat, split later

Every module starts as a single file and becomes a package only when it outgrows one — roughly 300–500 lines. No empty
folders, no stub files: a file appears when it contains working code.

### Configuration

Hydra is the harness. `configs/config.yaml` composes the groups a run needs, and any group can be overridden from the
command line:

```bash
python -m grooveback.cli.baseline model=apollo
```

It starts with one group and grows as the runs do. Config files land alongside the code they configure.

### Dependencies

Added as the code that needs them is written. `pyproject.toml` is the record.

### Testing

Domain logic stays free of infrastructure: `audio`, `manifest` and `evaluation` must be importable and testable with no
GPU and no network.

Audio is a domain where a plausible diff quietly makes the output worse without raising anything, so the tests aim at
that:

- **An identity model through the full path returns the input.** Catches chunking and overlap-add bugs, which matters
  straight away because Apollo runs chunked.
- **Every evaluation run also scores the untouched input and the clean reference.** If the reference doesn't win, the
  harness is broken rather than the model.
- **Loudness is asserted** before any comparison is scored.
- **SAME round-trip clears a fixed threshold** on fixtures. Catches wrong channel order, wrong normalisation, stale
  checkpoint.
- **Property tests** on the invariants: sample rate and channel count preserved, no NaN, no clipping introduced.

### Compute

Everything in ADR-0005 runs locally on Apple Silicon, with Apollo chunked to fit. Rented GPU starts at fine-tuning.

## Consequences

- Some refactoring when files become packages. Accepted.

## Revisit triggers

- **ADR-0005's results land** — the rest gets designed then.
- **A second model family or audio representation**, which may justify packages instead of files.
- **Any module passes ~500 lines.**
