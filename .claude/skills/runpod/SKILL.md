---
name: runpod
description: Run GPU experiments on RunPod with runpodctl — auth via RUNPOD_API_KEY in a gitignored .env, creating and killing pods, when a network volume is and is not worth it, moving files, and the cost rules. Use when a job is too slow locally and needs a rented GPU.
---

# RunPod

The rented-GPU tier. Prepaid credits on Zakaria's account, so a forgotten pod eats the budget directly.

`runpodctl` via `brew install runpod/runpodctl/runpodctl` (v2.9.0+).

## Auth

The key lives in `.env` at the repo root, which is gitignored. One variable:

```bash
RUNPOD_API_KEY=rpa_...        # runpod.io -> Settings -> API Keys
```

Load it before any `runpodctl` call. Keep the value out of commands, commits and files — reading it from the
environment is the whole point:

```bash
set -a; source .env; set +a
```

`runpodctl doctor` prompts for the key and stores it in its own config instead, which also works — the `.env` exists
so the value has one home rather than being re-pasted from shell history.

SSH uses `~/.ssh/id_ed25519`, whose public half is registered in the RunPod UI.

## Commands

```bash
runpodctl user                     # balance and current spend
runpodctl gpu list                 # ids, prices, per-datacenter stock
runpodctl datacenter list
runpodctl pod list                 # ALWAYS check this before finishing
runpodctl pod get <id>
runpodctl pod create --image <img> --gpu-id "NVIDIA L4" --min-cuda-version 12.8 \
    --container-disk-in-gb 40 --ports 22/tcp --name <name>
runpodctl pod delete <id>
runpodctl network-volume create --name <n> --size <gb> --data-center-id <dc>
```

Files: `scp`/`rsync` over the pod's SSH is the reliable path. `runpodctl send` / `receive` (croc) also works.

## Setting up a pod

**Always make a venv, even here** — the image's Python is externally managed and plain `pip install` fails silently,
with no error in quiet mode. Inherit the preinstalled torch rather than downloading it again.

Use uv — it is a single binary and much faster than pip:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --system-site-packages /workspace/venv

# example only — install whatever the job actually imports and the image lacks
uv pip install --python /workspace/venv/bin/python \
    numpy soundfile einops safetensors huggingface-hub
```

`--system-site-packages` is the load-bearing flag: it lets the venv see the image's torch so only the missing
packages are fetched.

Avoid `uv sync --frozen` on a pod. It follows the project lockfile, which pins its own torch, so it re-downloads
~4 GB of torch and CUDA wheels the image already has — over half an hour, against 5 seconds for inheriting. Install
the handful of missing packages explicitly instead of syncing an environment.

**Exception: sync when the lockfile's torch is the point.** If the project pins a torch version the image does not
provide, inheriting gives you the wrong one and results will not match local runs. Then pay the download and sync
properly. Check before assuming — compare the image's `torch.__version__` against the project's pin; only sync when
they genuinely differ in a way the job cares about.

Run installers without `-q` so a refusal is visible.

Note that `runpod/pytorch:*` images ship torch **without numpy**, so a missing-numpy error looks like a broken image
when it is really a blocked installer.

## Gotchas

- **Pass `--ports 22/tcp` at creation.** SSH is not published by default, so `--wait` times out on a perfectly
  healthy pod — while it bills. Fixing it afterwards with `pod update --ports` restarts the container.
- **`--wait` hangs rather than failing** when a container crash-loops. If it stalls, check `pod get` in another
  shell rather than waiting.
- **A failed or crash-looping pod still reports RUNNING and still bills.** Delete it; do not wait for it to recover.
- **Match CUDA to the host driver** with `--min-cuda-version`, or an image can crash-loop on an older driver.
- **Network volumes are pinned to a datacenter and cannot move.** A volume therefore constrains which GPUs you can
  reach — see *Volumes* below before assuming you need one.
- Stock shown as "available" does not guarantee a schedulable host; `pod create` can still fail.

## Volumes: default to none

A volume is pinned to one datacenter, and datacenters do not carry every GPU. US-MO-1 has A100s and no 4090s, so
attaching that volume rules out the cheap consumer cards. Reaching both classes with persistence would mean a second
volume in a second region — but that is the wrong conclusion, because **most jobs need no volume at all**.

Decide by setup cost, not by habit:

| | volume | why |
|---|---|---|
| A2SB | yes — the existing `grooveback-US-MO-1` (`18v73b8ggl`, 14 GB) | fork clone, its own venv, and checkpoints; rebuilding each time is real work, and it needs an A100 anyway |
| SAME-L, and anything else | **no** | setup is a git clone plus a venv that inherits the image's torch. The whole 140-window transport run, setup included, was 111 s on a fresh $0.49/h L4 |

So: one volume, where the heavy setup lives, and everything else runs volume-less on whatever GPU is cheapest in any
region. Do not add a second volume to chase a GPU class.

**Keep `grooveback-US-MO-1`. Do not delete it to save money.** It bills ~$1/month whether used or not, and that cost
is accepted: it exists so returning to an A100 does not mean rebuilding the A2SB fork, its venv and its checkpoints
from scratch. Deleting it trades a known dollar for an unknown hour.

That $1/month is also the whole budget, about 14 GB at $0.07/GB/mo, so a second volume either halves the space or
doubles the bill. Neither is worth it when the alternative is a five-minute setup on a fresh pod.

## Cost rules

- Pods run experiments only. Results come back to the Mac for analysis.
- **`pod delete` when the run finishes, not `pod stop`** — stopped pods still bill for disk.
- Check `runpodctl pod list` returns `[]` before considering a job done.
- Pick the cheapest GPU that fits. Most jobs here are small.
- Watch `runpodctl user` for the balance.

Pair every pod setup with the audit loop in `CLAUDE.md` — provisioning is exactly the kind of outside effect that
hangs silently.
