# Environment

uv, Python 3.12. `uv sync --all-groups`, `uv run pytest`.

## Third-party models

Two are vendored, one is a dependency. **torch is pinned to 2.7.1 project-wide
by stable-audio-3** — see ADR-0007.

| model | location | called by |
|---|---|---|
| Apollo | `third_party/apollo` (git submodule) | imported directly; chunking and its tests live in the fork |
| A2SB | `third_party/a2sb` (gitignored clone, fork branch `runnable-anywhere`, own venv) | `baselines.run_a2sb` → its `restore.py`, behind a subprocess — its pins genuinely conflict |
| AudioSR | pip `audiosr` in its own venv at `third_party/audiosr/.venv` (gitignored, no fork) | `baselines.run_audiosr` → `scripts/audiosr_restore.py`, behind a subprocess — its torch 2.0.1 pins genuinely conflict (ADR-0012) |
| HP-codecX | `third_party/hpcodecx` (gitignored release clone at v1.0.1, own venv + Zenodo weights) | `baselines.run_hpcodecx` → its unmodified `predict.py`, behind a subprocess — its torch.package checkpoints need the torch 2.1 era (ADR-0013) |
| SAME | `stable-audio-3`, a git dependency pinned by commit in `pyproject.toml` | `grooveback.latents`, in-process |
| εar-VAE | `third_party/earvae` (git submodule; deps in group `aes`) | `grooveback.latents` registry, in-process |
| εar-VAE2 | `third_party/earvae2` (git submodule) | `grooveback.latents` registry, in-process |
| CoDiCodec | pip package `codicodec` (group `aes`) | `grooveback.latents` registry, in-process |

```bash
git submodule update --init --recursive
git clone -b runnable-anywhere git@github.com:zbitouzakaria/diffusion-audio-restoration.git third_party/a2sb
cd third_party/a2sb && ./setup.sh
scripts/audiosr_setup.sh
scripts/hpcodecx_setup.sh
```

## Running things

```bash
uv run python scripts/run_xp.py cuda     # the whole benchmark, one pass
uv run python -m grooveback.cli.baseline --method apollo <track>
uv run python -m grooveback.cli.run model=audiosr input=<file-or-dir> output_dir=artifacts/audiosr
uv run --group notebooks python scripts/build_demo.py   # listening pages from artifacts/*/listen
python3 -m http.server 8880 -d demo      # then open /base/ or /sdedit/ (ADR-0010)
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
| AudioSR (50 steps, the pinned setting) | no | ~1 min per 10 s chunk on an L4; ~8 s on an A100 — 180 s stereo ≈ 5 min |
| Any fine-tuning | no | yes |

Past local timings are sizing data, not a recommendation.

Outputs land in `artifacts/` (gitignored). The benchmark writes
`artifacts/xp/{source}/{bitrate}/`: `input.wav` is the untouched MP3 twin,
`{method}.wav` the renders, `listen/` the level-matched copies to A/B.
