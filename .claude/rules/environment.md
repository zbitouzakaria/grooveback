# Environment

uv, Python 3.12. `uv sync --all-groups`, `uv run pytest`.

## Third-party models

Some are vendored, some are dependencies, and two run in their own environments. **torch is pinned to 2.7.1 project-wide
by stable-audio-3** — see ADR-0007.

| model | location | called by |
|---|---|---|
| Apollo | `third_party/apollo` (git submodule) | imported directly; chunking and its tests live in the fork |
| A2SB | `third_party/a2sb` (gitignored clone, fork branch `runnable-anywhere`, own venv) | `baselines.run_a2sb` → its `restore.py`, behind a subprocess — its pins genuinely conflict; `scripts/a2sb_restore.py` restores a directory of clips in one run |
| AudioSR | pip package `audiosr` 0.0.7 in its own venv at `third_party/audiosr/.venv` (gitignored, built by `scripts/audiosr_setup.sh`) | `scripts/audiosr_restore.py`, run with that venv's python — its pins conflict too |
| SAME and the Stable Audio 3 priors | `stable-audio-3`, a git dependency pinned by commit in `pyproject.toml` | `grooveback.latents` and `grooveback.priors`, in-process |
| εar-VAE | `third_party/earvae` (git submodule; deps in group `aes`) | `grooveback.latents` registry, in-process |
| εar-VAE2 | `third_party/earvae2` (git submodule) | `grooveback.latents` registry, in-process |
| CoDiCodec | pip package `codicodec` (group `aes`) | `grooveback.latents` registry, in-process |

```bash
git submodule update --init --recursive
git clone -b runnable-anywhere git@github.com:zbitouzakaria/diffusion-audio-restoration.git third_party/a2sb
cd third_party/a2sb && ./setup.sh
```

## Running things

```bash
uv run python scripts/run_xp.py device=cuda     # the whole benchmark, one pass
uv run python -m grooveback.cli.baseline --method apollo <track> <output.wav>
uv run python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps 24   # ADR-0015, ADR-0016
uv run --group notebooks python scripts/build_demo.py   # listening pages from artifacts/*/listen
uv run --group notebooks python scripts/publish_benchmark.py [--page-only]   # the benchmark page, docs/benchmark/
python3 -m http.server 8880 -d demo      # then open /base/, /sdedit/ or /roundtrip/ (ADR-0010)
```

`run_xp.py` skips any render whose file already exists and re-scores
everything on every run, so it is safe to re-run, and renders made on a GPU
pod are scored once they are copied back.

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

Torch on a pod sees all of the host's cores (128–256 measured), while the
container is allowed far fewer (13–27). When several processes share a pod,
set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` so that processes × threads fits
the container's cores.

Outputs land in `artifacts/` (gitignored). The benchmark writes
`artifacts/xp/{source}/{bitrate}/`: `input.wav` is the untouched MP3 twin,
`{method}.wav` the renders, `listen/` the level-matched copies to A/B.
