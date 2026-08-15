# 2. Plan code architecture

Date: 2026-04-21

## Status

Accepted

## Context
 
grooveback is starting from near-zero code. This ADR fixes the target layout so early commits stay minimal without losing sight of where the structure is heading. It is a plan, not a contract — revisited when the reality diverges.
 
## Decision
 
### Target tree
 
```
grooveback/
├── configs/
│   ├── config.yaml
│   ├── data/
│   ├── degradation/           # named recipes: mp3_128, yt_rip, vinyl_rip, full_chain
│   ├── model/
│   ├── training/
│   └── experiment/            # end-to-end presets
├── docs/
│   ├── roadmap.md
│   └── decisions/
├── src/grooveback/
│   ├── audio.py               # io, resampling, stft
│   ├── degradations.py        # the core research artifact
│   ├── data.py                # manifests, datasets
│   ├── models.py              # restoration models
│   ├── losses.py
│   ├── metrics.py
│   ├── training.py
│   ├── inference.py           # chunked restoration
│   ├── evaluation.py
│   └── cli/
│       ├── train.py
│       ├── infer.py
│       ├── evaluate.py
│       └── validate_degradation.py   # synthetic vs real YT rip comparison
├── notebooks/
├── scripts/
├── tests/
├── demo/                      # static A/B player site (later)
├── data/                      # gitignored
└── artifacts/                 # gitignored
```
 
### Grow flat, split later
 
Every module above starts as a single file. A file becomes a package (`audio.py` → `audio/`) only when it genuinely outgrows one file — roughly 300–500 lines or when clearly separable concerns accumulate. No empty folders, no stub files.
 
### Hydra composition
 
`configs/config.yaml` composes defaults from four groups: `data`, `degradation`, `model`, `training`. Any run can override any group from the command line:
 
```bash
python -m grooveback.cli.train degradation=yt_rip model=unet_spec
```
 
Named end-to-end experiments live in `configs/experiment/` and pin all four groups plus any run-specific overrides:
 
```bash
python -m grooveback.cli.train +experiment=baseline_mp3
```
 
Config files are added alongside the code they configure — a `configs/degradation/vinyl_rip.yaml` appears when the vinyl pipeline code exists, not before.
 
### Degradation pipeline is the centerpiece
 
Following Real-ESRGAN's thesis that the degradation model is the artifact in real-world restoration, `degradations.py` gets disproportionate care: dedicated tests, named recipes mirroring the Hydra configs, and a `validate_degradation.py` CLI that compares synthetic output against real YouTube rips for ear-based validation.
 
### Evaluation: metrics catch regressions, listening is the judge
 
Numeric metrics (LSD, ViSQOL, FAD) are implemented in `metrics.py` and tracked every run. The authoritative evaluation is a curated set of tracks from a personal library, played back on monitoring hardware. `evaluation.py` produces both numeric reports and A/B listening packs.
 
## Consequences
 
- Commits stay small and truthful; structure follows code.
- Some refactoring when files become packages — accepted cost.
- Navigating the repo requires this ADR plus the README. Acceptable if both stay current.
## Revisit triggers
 
- A second audio representation (e.g. CQT) — may justify a `representations/` module.
- A second model family (diffusion, flow-matching) — may justify `models/` as a package.
- Commercialization — introduces licensing and API-surface concerns not covered here.
