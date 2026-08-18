---
name: operate-grooveback
description: How to run grooveback — environment, the third-party model clones behind subprocess boundaries, the experiment scripts, and the audio conventions every comparison depends on. Use before running or modifying anything in this project.
---

# Operating grooveback

Restoring degraded electronic music with a generative prior. **What we have learned lives in `docs/decisions/`** —
read ADR-0004 (the thesis), ADR-0006 (baselines) and ADR-0007 (the latent space) before proposing a change of method.
This file is only how to run things.

## Environment

uv, Python 3.12.

```bash
uv sync --all-groups
uv run pytest
```

## Third-party models live behind subprocess boundaries

Three external models are used, none of them importable alongside each other — their torch pins conflict. Each is
cloned into `third_party/` (gitignored, not submodules except Apollo) and called by shelling out.

| model | location | how it is called |
|---|---|---|
| Apollo | `third_party/apollo` (submodule) | imported directly; only needs torch, numpy, huggingface_hub |
| A2SB | `third_party/a2sb` — fork, branch `runnable-anywhere` | `run_a2sb` shells out to its `restore.py` |
| SAME | `third_party/stable-audio-3` | `grooveback.latents` shells out to `scripts/same_codec.py` |

```bash
git submodule update --init --recursive                     # Apollo
git clone -b runnable-anywhere git@github.com:zbitouzakaria/diffusion-audio-restoration.git third_party/a2sb
cd third_party/a2sb && ./setup.sh
git clone https://github.com/Stability-AI/stable-audio-3.git third_party/stable-audio-3
cd third_party/stable-audio-3 && uv sync --frozen
```

**Do not try to import these into the project environment.** The subprocess boundary is the design, not a workaround.

## Running things

```bash
uv run python -m grooveback.cli.baseline --method apollo <track>    # baselines
uv run python scripts/run_experiments.py mps same-s same-l         # SAME round-trip + transport
```

`run_experiments.py` skips any step whose output exists, so it is safe to re-run and can pick up renders produced
elsewhere — SAME-L is generated on a rented GPU, since it is far too slow locally (see the `runpod` skill).

Outputs land in `artifacts/` (gitignored). Listening sets are named for the operation that produced them:
`1_input_*` is before, `2_output_*` is after, `ref_*` are references.

## Audio conventions that decide whether a result is real

- **Level-match to −14 LUFS before any comparison**, with one common headroom gain across the set. Loudness
  differences dominate informal comparisons and produce confident wrong conclusions.
- **Judge a latent-space method against `decode(encode(x))`, never against the input file.** The autoencoder is not
  transparent and invents content on bandlimited input, so scoring against the input credits its behaviour to the
  method.
- **Energy is not perception.** Band deltas catch regressions; blinded listening on monitoring hardware decides.
- **Residuals are saved before level matching**, so they are the honest difference.

## Correctness gates

Audio breaks silently — no exception, nothing looks wrong, the result is merely worse. The tests target that:

- an identity model through the full inference path must return the input (catches chunking and overlap-add bugs)
- every evaluation run also scores the untouched input and the clean reference; if the reference does not win, the
  harness is broken rather than the model
- loudness is asserted before any comparison is scored
- property tests on the invariants: sample rate and channel count preserved, no NaN, no clipping introduced

An identity model is coherent by construction, so these cannot catch a model that *invents* content differently on
each call. That class needs a fake model returning random content in a band, then asserting flat band power across
seams — worth adding before chunking anything generative.
