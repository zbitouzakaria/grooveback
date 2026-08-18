# Environment

uv, Python 3.12. `uv sync --all-groups`, `uv run pytest`.

## Third-party models run behind subprocess boundaries

Three external models are used and their torch pins conflict, so none of them can be imported alongside the others.
Each is cloned into `third_party/` and called by shelling out. **The subprocess boundary is the design, not a
workaround** — do not try to import them into the project environment.

| model | location | called by |
|---|---|---|
| Apollo | `third_party/apollo` (git submodule) | imported directly; only needs torch, numpy, huggingface_hub |
| A2SB | `third_party/a2sb` (gitignored clone, fork branch `runnable-anywhere`) | `baselines.run_a2sb` → its `restore.py` |
| SAME | `third_party/stable-audio-3` (gitignored clone) | `grooveback.latents` → `scripts/same_codec.py` |

```bash
git submodule update --init --recursive
git clone -b runnable-anywhere git@github.com:zbitouzakaria/diffusion-audio-restoration.git third_party/a2sb
cd third_party/a2sb && ./setup.sh
git clone https://github.com/Stability-AI/stable-audio-3.git third_party/stable-audio-3
cd third_party/stable-audio-3 && uv sync --frozen
```

## Running things

```bash
uv run python -m grooveback.cli.baseline --method apollo <track>
uv run python scripts/run_experiments.py mps same-s same-l
```

`run_experiments.py` skips any step whose output already exists, so it is safe to re-run and can pick up renders
produced elsewhere — which is how GPU-rendered results get folded back in.

## What does not run locally

The MacBook is 16 GB unified memory. These need a rented GPU — use the `runpod` skill without asking, then delete the
pod:

| | local | rented |
|---|---|---|
| SAME-L encode/decode | ~10x slower than realtime | the full 140-window transport run took **111 s on an L4** |
| A2SB, full track | does not fit | A100 |
| Any fine-tuning | no | yes |

Apollo, SAME-S, the degradation chain, evaluation and listening all run locally and should stay there.

Outputs land in `artifacts/` (gitignored). Listening sets are named for the operation that produced them: `1_input_*`
is before, `2_output_*` is after, `ref_*` are references.
