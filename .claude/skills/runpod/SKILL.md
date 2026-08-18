---
name: runpod
description: Run GPU experiments on RunPod with runpodctl — auth, creating and killing pods, network volumes, moving files, and the cost rules. Use when a job is too slow locally and needs a rented GPU.
---

# RunPod

The rented-GPU tier. Prepaid credits on Zakaria's account, so a forgotten pod eats the budget directly.

`runpodctl` via `brew install runpod/runpodctl/runpodctl` (v2.9.0+).

## Auth

Read the key from the environment; it is never committed.

```bash
export RUNPOD_API_KEY=...   # runpod.io -> Settings -> API Keys
```

`runpodctl doctor` prompts for it and saves it instead. SSH uses `~/.ssh/id_ed25519`, whose public half is
registered in the RunPod UI.

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
with no error in quiet mode. Inherit the preinstalled torch rather than downloading it again:

```bash
python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install numpy soundfile einops safetensors huggingface-hub
```

Never `uv sync --frozen` a torch project on a pod: the lockfile pulls ~4 GB of torch and CUDA wheels the image
already has. Inheriting took 5 seconds where syncing took over half an hour. Run pip without `-q` so a refusal is
visible.

Note that `runpod/pytorch:*` images ship torch **without numpy**, so a missing-numpy error looks like a broken image
when it is really a blocked installer.

## Gotchas

- **Pass `--ports 22/tcp` at creation.** SSH is not published by default, so `--wait` times out on a perfectly
  healthy pod — while it bills. Fixing it afterwards with `pod update --ports` restarts the container.
- **`--wait` hangs rather than failing** when a container crash-loops. If it stalls, check `pod get` in another
  shell rather than waiting.
- **A failed or crash-looping pod still reports RUNNING and still bills.** Delete it; do not wait for it to recover.
- **Match CUDA to the host driver** with `--min-cuda-version`, or an image can crash-loop on an older driver.
- **Network volumes are pinned to a datacenter and cannot move.** That constrains which GPUs are reachable, so a
  volume can force an expensive GPU class. For one-off jobs, skip the volume entirely and pick on price.
- Stock shown as "available" does not guarantee a schedulable host; `pod create` can still fail.

## Cost rules

- Pods run experiments only. Results come back to the Mac for analysis.
- **`pod delete` when the run finishes, not `pod stop`** — stopped pods still bill for disk.
- Check `runpodctl pod list` returns `[]` before considering a job done.
- Pick the cheapest GPU that fits. Most jobs are small: 140 SAME-L encodes plus decodes took 111 s on a $0.49/h L4.
- Watch `runpodctl user` for the balance.

Pair every pod setup with the audit loop in `CLAUDE.md` — provisioning is exactly the kind of outside effect that
hangs silently.
