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
produced elsewhere. SAME-L is generated on a rented GPU — it is far too slow locally. See the `runpod` skill.

Outputs land in `artifacts/` (gitignored). Listening sets are named for the operation that produced them: `1_input_*`
is before, `2_output_*` is after, `ref_*` are references.
