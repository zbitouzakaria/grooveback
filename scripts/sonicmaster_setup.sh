#!/bin/sh
# Build the SonicMaster environment at third_party/sonicmaster (gitignored):
# the release clone, its own venv, and the checkpoint.
#
# Its requirements pin torch 2.4.0 (+ transformers 4.44, diffusers 0.30),
# which conflict with the project's torch 2.7.1 stack, so it runs in its own
# venv behind a subprocess -- the established arrangement. Everything
# installs in one resolution (the ADR-0012 lesson). The README's
# "python==3.13" cannot be right for torch 2.4.0 (no cp313 wheels); 3.11 is
# used (ADR-0014).
#
# Runtime note: inference additionally downloads the Oobleck VAE from the
# gated stabilityai/stable-audio-open-1.0 repo, so HF_TOKEN (with that
# license accepted) must be in the environment when the model runs.
set -eu

cd "$(dirname "$0")/.."

if [ ! -f third_party/sonicmaster/infer_single.py ]; then
    git clone --depth 1 https://github.com/AMAAI-Lab/SonicMaster.git \
        third_party/sonicmaster
fi

uv venv --clear --python 3.11 third_party/sonicmaster/.venv
PYTHON=third_party/sonicmaster/.venv/bin/python
uv pip install --python "$PYTHON" \
    -r third_party/sonicmaster/requirements_sonic.txt soundfile

WEIGHTS=third_party/sonicmaster/weights/model.safetensors
if [ ! -f "$WEIGHTS" ] || [ "$(wc -c < "$WEIGHTS")" -lt 500000000 ]; then
    mkdir -p third_party/sonicmaster/weights
    "$PYTHON" -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('amaai-lab/SonicMaster', 'model.safetensors',
                       local_dir='third_party/sonicmaster/weights')
print(path)
"
fi
if [ "$(wc -c < "$WEIGHTS")" -lt 500000000 ]; then
    echo "$WEIGHTS came back tiny — not the checkpoint" >&2
    exit 1
fi

"$PYTHON" -c "import torch, diffusers, transformers, laion_clap" \
    && echo "sonicmaster venv ready"
