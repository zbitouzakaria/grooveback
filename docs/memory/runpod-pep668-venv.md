---
name: runpod-pep668-venv
description: RunPod pytorch images block pip via PEP 668; make a venv with --system-site-packages to reuse their torch
metadata: 
  node_type: memory
  type: reference
  originSessionId: 7a40108c-c6aa-4b86-b4a1-81c77b27918c
  modified: 2026-08-18T11:55:04.403Z
---

`runpod/pytorch:*` images ship an externally-managed Python (PEP 668). Plain `pip install` **fails silently in quiet
mode** — no packages, no visible error. They also ship torch without numpy, so the missing-numpy symptom looks like a
broken image rather than a blocked installer.

```bash
python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install numpy soundfile einops einops-exts safetensors huggingface-hub
```

**Why:** this cost about 50 minutes and one wasted pod on 2026-08-18. It presents as a slow network — `uv sync
--frozen` is separately trying to pull ~4 GB of torch and CUDA wheels, so both look like bandwidth. Inheriting the
image's preinstalled torch instead took **5 seconds**.

**How to apply:** never `uv sync --frozen` a torch project on a pod; the image already has torch. Make the venv with
`--system-site-packages` and install only what is missing. Always run pip without `-q` on a pod so a refusal is
visible. Other gotchas from the same session:

- `pod create` does not publish SSH by default — pass `--ports 22/tcp` at creation, or `--wait` times out on a healthy
  pod while it bills. Fixing it afterwards with `pod update --ports` restarts the container and resets uptime.
- `--min-cuda-version` constrains scheduling to hosts whose driver matches the image.
- For one-off jobs, skip the network volume: it pins the datacenter (US-MO-1, A100-priced) when a $0.49/h L4 elsewhere
  is plenty. SAME-L transport over 140 windows took 111 s on an L4.

See [[runpod-cli]] for the general workflow and house rules, and [[audit-long-running-commands]] for how to watch pod
setup without hanging.
