# 2. Plan code architecture

Date: 2026-04-21 · rewritten 2026-08-15

## Status

Accepted

Rewritten in place on 2026-08-15 for the prior-first design of [ADR-0004](0004-restoration-as-a-latent-inverse-problem.md).
The original called itself "a plan, not a contract", and nothing had been implemented against it — every module was a
docstring — so rewriting is more honest than superseding a document no code ever followed.

## Context

The original layout was designed for supervised training: a degradation pipeline as the centrepiece, a spectrogram
U-Net, a loss module, and a training loop as the main run type. ADR-0004 removed all four. The prior is now sourced by
adapting a pretrained model, the degradation chain is an evaluation instrument rather than a training distribution, and
the first real run type is inference against frozen checkpoints.

The **module tree** below is scoped to what [ADR-0005](0005-baselines-and-prior-viability-on-real-library-material.md)
requires and nothing further. ADR-0005 is a deliberate stopping point, and committing to a module for a solver that has
not been chosen yet would be the same mistake this rewrite is correcting.

Two sections deliberately look past that horizon and are marked where they do: the **compute topology**, because it
determines what hardware is rented and when, and parts of the **dependency table**, because a dependency admitted now
should be justified against the use it will actually get. Neither authorises writing the corresponding code.

## Decision

### Target tree

```
grooveback/
├── configs/
│   └── config.yaml
├── docs/
│   └── decisions/
├── src/grooveback/
│   ├── audio.py         # io, resample, LUFS metering and matching, STFT, spectrogram rendering
│   ├── manifest.py      # pydantic schemas + JSONL over the evaluation set
│   ├── baselines.py     # Apollo (chunked), A2SB, DSP control
│   ├── latents.py       # SAME encode/decode, round-trip reporting
│   ├── priors.py        # Stable Audio 3 load, unconditional sample, inpaint
│   ├── evaluation.py    # level matching, null/oracle control, listening packs
│   ├── metrics.py       # minimal set; LSD as a regression tripwire only
│   └── cli/
│       ├── ingest.py        # build the manifest
│       ├── baseline.py      # run Apollo / A2SB / DSP control, singly or chained
│       ├── roundtrip.py     # SAME encode-decode report
│       ├── probe.py         # Stable Audio 3 viability probe
│       └── evaluate.py      # level-matched comparison + listening pack
├── notebooks/
├── scripts/
├── tests/
├── data/                # gitignored
└── artifacts/           # gitignored
```

`manifest.py` holds the schema that ADR-0002's original tree implied lived in `data.py`. It never did — `data.py` was
an empty docstring. Datasets over latent shards will want their own module later; the schema is the part needed now,
and it earns a separate module because it is consumed by every other component.

### Grow flat, split later

Retained from the original. Every module starts as a single file and becomes a package only when it genuinely outgrows
one — roughly 300–500 lines, or when clearly separable concerns accumulate.

**No empty folders, no stub files.** The original tree was created as docstring-only placeholders, which is the failure
mode this rule exists to prevent: a stub asserts a design decision without paying for it, and `degradations.py`
survived four months describing itself as "the core research artifact" after that framing had been abandoned. The
surviving stubs are to be removed, and files in the tree above appear when they contain working code.

### Deliberately not written yet

Named so their absence reads as scheduled rather than overlooked: `degradations.py` (the `A_eval` chain),
`operators.py` (differentiable surrogates for use inside a solver), `solvers.py`, `data.py` (datasets over latent
shards), `training.py`, `inference.py`, `listening.py` (an ABX harness — blinded comparison where the ordering is
randomised and the verdict logged, so listening produces a record rather than a memory). Each waits on an ADR that
waits on ADR-0005's
results.

### Dependencies

One sentence each, per the rule that every dependency justifies itself against its alternative.

| Package | Why this and not the alternative |
|---|---|
| `torch`, `torchaudio` | Tensor ops plus audio IO, resampling and STFT in one stack, so the same code runs on MPS locally and CUDA on a rented box |
| `numpy` | Unavoidable interchange format at the edges |
| `soundfile` | libsndfile bindings; the reliable file-IO layer, where torchaudio's backend story has historically churned |
| `einops` | Makes tensor reshapes in latent code readable and self-documenting, which matters most where a silent axis swap is undetectable |
| `pydantic` | Validation for manifests now, and for latent-cache metadata once that exists — the two places a silent schema error corrupts everything downstream. Not used for configs; Hydra/OmegaConf already owns that and two schema systems fight |
| `pyloudnorm` | ITU-R BS.1770 loudness metering, required for the −14 LUFS matching every comparison depends on; nothing else in the stack implements the standard |
| `pedalboard` | The DSP control condition (gain, high shelf, mono below ~120 Hz) at plugin quality, and it covers what `audiomentations` would have been added for |
| `matplotlib` | Spectrogram rendering for by-eye comparison, which ADR-0005 treats as a primary instrument |
| `hydra-core`, `omegaconf` | Composable run configuration with command-line override of any group |
| `ffmpeg` (system, not a Python dep) | Decoding whatever container and codec a library file arrives in, which `soundfile` alone does not cover. Later it also builds degraded test cases, where an approximation of a codec is not a codec |

Deliberately excluded, with reasons, so they are not re-added by reflex:

- **`librosa`** — numpy-bound and pulls numba; torchaudio covers STFT, mel and resampling and runs on MPS. Latent
  precompute over ~2,000 tracks is IO- and CPU-bound and librosa would be the bottleneck. Available in the `notebooks`
  dependency group for analysis, where its convenience is worth its cost.
- **`audiomentations`** — largely redundant with pedalboard for this use case.
- **`pesq`, `pystoi`** — speech-intelligibility metrics, wrong domain for full-mix music.

Development tooling lives in PEP 735 `[dependency-groups]` (`dev`, `notebooks`) rather than
`[project.optional-dependencies]`, since these are workflow groups rather than installable extras.

### Configuration

Hydra stays minimal at this stage. ADR-0005 has effectively one run type — run a frozen model over a set of tracks and
produce a comparison — so the original four-group composition (`data`, `degradation`, `model`, `training`) would be
scaffolding for runs that do not exist.

The restructure is recorded as **pending, not built**: the real axes of variation in the prior-first design are
`data / autoencoder / prior / solver / degradation / eval`, `model` splits into `autoencoder` and `prior`, and
restoration runs are a distinct run type from training runs. That lands when there is a second run type to justify it.

Config files continue to appear alongside the code they configure — a config for a component that does not exist yet is
a stub by another name.

### Domain logic stays untangled from infrastructure

`audio`, `manifest`, `metrics` and `evaluation` must be importable and testable with no GPU, no cloud, no experiment
tracker and no network. Tracking, Hydra and device management live at the CLI and training edges. This is enforced
mechanically rather than by discipline — see the gates below.

### Compute topology

Forward-looking: this section covers work beyond ADR-0005, because it determines what hardware gets rented and when.

- **Local (Apple Silicon, 16 GB unified, ~10–11 GB usable)**: manifests, level matching, evaluation, spectrograms,
  listening, the DSP control, Apollo (chunked), A2SB, SAME round-trip, and — because SAME is small and fast —
  plausibly the full-library latent precompute as an overnight job. At SAME's reported ~10.8 latent frames per second
  across 256 channels (see ADR-0004), 215 hours of audio is roughly 4 GB in fp16, so the cache is not the constraint
  and fits in RAM.
- **Rented GPU (4090-class, 24 GB)**: fine-tuning and, later, solver sweeps. Budget for solver sweeps separately:
  posterior sampling at hundreds of steps across a hyperparameter grid will outspend the fine-tune.
- **Latents are precomputed once and never encoded inside a training loop.** A hard rule for when training exists, not
  an optimisation.

### Correctness gates

Audio is a domain where a plausible diff silently breaks the output — no exception is raised, nothing looks wrong, and
the result is merely worse. These gates target that class of error specifically, and all of them run on CPU with no
network.

| Gate | Catches |
|---|---|
| **No-op model test** — run the full inference path with an identity model; output must be sample-close to input | Chunking, overlap-add and windowing bugs. Directly load-bearing now: Apollo runs chunked |
| **Null/oracle control in every evaluation run** — always score the untouched degraded input and the clean reference | Metric wiring errors. If clean does not win and degraded does not lose, the harness is broken, not the model |
| **Loudness assertion** — every comparison asserts all items within ±0.1 LU of −14 LUFS before scoring | The single most common cause of confidently wrong audio comparisons |
| **SAME round-trip threshold** — encode→decode must clear a fixed SI-SDR / mel-L1 bound on fixtures | Wrong channel order, wrong normalisation, stale or mismatched checkpoint |
| **Golden-file spectral regression** with perceptual tolerance (mel-spectrogram L1 within bound), never bit-exact | Unintended processing changes, without the false positives that make bit-exact audio tests get muted after the first library upgrade |
| **Property tests (hypothesis)** — sample rate preserved, channel count preserved, length contract held, no NaN/Inf, no clipping introduced beyond a declared gain stage, determinism under fixed seed | The invariants that hold across every transform in the codebase |
| **Import-linter architecture test** — `audio`, `manifest`, `metrics`, `evaluation` may not import hydra, any experiment tracker, or CUDA-only paths | Infrastructure leaking into domain logic, which is what makes tests need a GPU six months later. The tracker is named in the contract when one is added; no tracker is a dependency yet |

Tests come first where correctness is checkable. Where it is not — whether a restoration sounds good — the judge is
blinded listening on monitoring hardware, and no gate substitutes for it.

## Consequences

- The tree is smaller than the original and describes only work that is about to happen. Navigating the repo requires
  this ADR plus ADR-0004 and ADR-0005.
- Deleting the stubs means the repo briefly contains almost no Python. That is accurate: almost no Python has been
  written.
- Some refactoring when files become packages — accepted, unchanged from the original decision.
- The Hydra restructure is deferred, so the first configs will be thin and the composition story arrives later than a
  config-first project would have it.
- Excluding librosa from core costs some convenience in exchange for a precompute path that runs on MPS.

## Revisit triggers

- **ADR-0005's results land** — the deferred modules and the Hydra restructure are designed then, from evidence.
- **A second run type appears** (training, or a solver sweep) — that is what justifies the config group restructure.
- **A second audio representation** (e.g. CQT) — may justify a `representations/` module.
- **A second prior or autoencoder** — may justify `priors/` and `latents/` as packages behind a common interface.
- **Latent precompute proves infeasible locally** — moves the autoencoder to rented GPU and changes the compute split.
- **Any module passes ~500 lines** — split it.
