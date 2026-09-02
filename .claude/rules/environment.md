# Environment

uv, Python 3.12. `uv sync --all-groups`, `uv run pytest`.

## Third-party models

Two are vendored, one is a dependency. **torch is pinned to 2.7.1 project-wide
by stable-audio-3** — see ADR-0007.

| model | location | called by |
|---|---|---|
| Apollo | `third_party/apollo` (git submodule) | imported directly; chunking and its tests live in the fork |
| A2SB | `third_party/a2sb` (gitignored clone, fork branch `runnable-anywhere`, own venv) | `baselines.run_a2sb` → its `restore.py`, behind a subprocess — its pins genuinely conflict |
| SAME | `stable-audio-3`, a git dependency pinned by commit in `pyproject.toml` | `grooveback.latents`, in-process |

```bash
git submodule update --init --recursive
git clone -b runnable-anywhere git@github.com:zbitouzakaria/diffusion-audio-restoration.git third_party/a2sb
cd third_party/a2sb && ./setup.sh
```

## Running things

```bash
uv run python scripts/run_xp.py cuda     # the whole benchmark, one pass
uv run python -m grooveback.cli.baseline --method apollo <track>
```

`run_xp.py` skips any step whose output already exists, so it is safe to
re-run and picks up renders produced elsewhere — which is how GPU-rendered
results get folded back in.

## RunPod by default

Model inference and anything GPU-shaped runs on a rented GPU by default — use
the `runpod` skill without asking, then delete the pod. MPS hangs the machine
while it runs, which blocks working; run locally only what is demonstrably
trivial. The MacBook is 16 GB unified memory.

| | local | rented |
|---|---|---|
| SAME-S / SAME-L | ~10x slower than realtime | seconds per clip on an L4 |
| A2SB, full track | does not fit | A100 |
| Any fine-tuning | no | yes |

Past local timings are sizing data, not a recommendation.

Outputs land in `artifacts/` (gitignored). The benchmark writes
`artifacts/xp/{source}/{bitrate}/`: `input.wav` is the untouched MP3 twin,
`{method}.wav` the renders, `listen/` the level-matched copies to A/B.
