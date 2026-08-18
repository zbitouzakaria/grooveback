---
name: runpod-cli
description: "How to run GPU experiments on RunPod with runpodctl — auth, pods, network volume, transfer, and the house rules"
metadata: 
  node_type: memory
  type: reference
  originSessionId: d5c93125-9700-44bb-967a-f3423d0ebd90
  modified: 2026-08-17T15:01:18.634Z
---

RunPod is the rented-GPU tier for grooveback (RTX 4090 @ ~$0.74/h, Zakaria's account, prepaid credits). CLI is
`runpodctl` (installed via `brew install runpod/runpodctl/runpodctl`, v2.9.0+). Verified command map:

**Auth** (needed once per machine): `runpodctl doctor` prompts for the API key and saves it, or
`export RUNPOD_API_KEY=...`. Key comes from runpod.io → Settings → API Keys.
The key is read from `RUNPOD_API_KEY` and is never committed; get its value from runpod.io -> Settings -> API Keys. SSH uses Zakaria's `~/.ssh/id_ed25519`,
whose public half is registered in the RunPod UI.

**Discovery**: `runpodctl gpu list` (GPU ids), `runpodctl datacenter list` (availability), `runpodctl user` (balance).

**Network volume** (persistent, per-datacenter):
`runpodctl network-volume create --name grooveback --size <GB> --data-center-id <DC>` · `list` · `delete`.

**Pods**:
`runpodctl pod create --image runpod/pytorch:<tag> --gpu-id "NVIDIA GeForce RTX 4090" --network-volume-id <id>
--container-disk-in-gb 20 --name <name> --wait` — `--wait` blocks until SSH is reachable and prints it.
`runpodctl pod list` · `pod get <id>` · `pod stop <id>` · `pod delete <id>`.
Volume mounts at `/workspace` (default). Pods must be created in the volume's datacenter.

**Files**: `runpodctl send <file>` / `runpodctl receive <code>` (croc-based, no key needed), or plain
`scp`/`rsync` over the pod's SSH (preferred for wavs).

**House rules (Zakaria, 2026-08-17):**
- Pods are for running experiments only; wavs come back to the Mac for analysis (never analyse on the pod).
- One network volume, kept across runs, holding checkpoints + code. **Budget cap $1/month → ≤14 GB** at $0.07/GB/mo.
- Kill the pod (`pod delete`, not just stop — stopped pods still bill disk) as soon as a full experiment set
  finishes. The volume persists.
- Watch balance with `runpodctl user`; prepaid credits, so a forgotten pod eats the budget.

**Pod setup is uv-first (Zakaria, 2026-08-17):** bootstrap uv before anything else
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv venv --system-site-packages /workspace/venv` and
`uv pip install --python /workspace/venv/bin/python ...`. Never bare pip — it is slow and its quiet mode looks hung.

**Standard grooveback experiment cycle**: create pod on the volume's DC → `git clone -b runnable-anywhere
git@github.com:zbitouzakaria/diffusion-audio-restoration.git` (or pull) into `/workspace` → `./setup.sh` (venv on the
volume survives pod deletion; checkpoints land in the HF cache — set `HF_HOME=/workspace/hf` so they persist) →
run `restore.py` experiments → `scp` the wavs back → `runpodctl pod delete`.

**Gotchas learned 2026-08-17:**
- **Match the image's CUDA to the host driver.** EUR-NO-1's 4090 hosts run driver 550.163.01 = CUDA 12.4; a cu12.8
  image crash-loops with `nvidia-container-cli: requirement error` while the pod BILLS the whole time. Use
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-cudnn-devel-ubuntu22.04` there, or pass `--min-cuda-version` to constrain
  scheduling. `pod create --wait` hangs forever on a crash-looping container — check the GUI/`pod get` if it stalls.
- **Volumes are datacenter-pinned and cannot move.** Changing GPU class can force a DC change, which means a NEW
  empty volume and a full one-time setup rebuild — avoid by choosing the DC for the volume based on the GPUs the
  project will actually need. A2SB full-width needs >24 GB VRAM, so the volume lives with the A100s.
  **The grooveback volume is `18v73b8ggl`, 14 GB, US-MO-1** (A100 80GB PCIe available there; no 4090s).
  Everything persistent goes under `/workspace`: `a2sb/` (fork clone), `venv/`, `hf/` (HF_HOME), `inputs/`, `out/`.
  A new pod in US-MO-1 mounts it and runs with zero setup — setup must never be repeated while this volume lives.
- "available" stock in `datacenter list` does not guarantee a volume-compatible host; pod create can still return
  "no instances available". Walk candidate DCs programmatically.
- A failed/crash-looping pod still shows RUNNING and bills — delete it, don't wait.

Refine this file after each real use; anything that surprised the workflow gets written down here.
