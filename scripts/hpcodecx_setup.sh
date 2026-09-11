#!/bin/sh
# Build the HP-codecX environment at third_party/hpcodecx (gitignored):
# the release clone at tag v1.0.1, its own venv, and the two Zenodo weight
# files in the runs/ layout its loaders expect.
#
# The checkpoints are torch.package archives, which are sensitive to the
# torch version they were saved under, so torch is pinned to the release era
# (2.1.x) even though the repo's requirements floor is only torch>=2.1.
# Everything installs in one resolution -- the ADR-0012 lesson: a second
# install command lets the resolver quietly upgrade an already-pinned torch.
set -eu

cd "$(dirname "$0")/.."

if [ ! -d third_party/hpcodecx/scripts ]; then
    git clone --depth 1 --branch v1.0.1 \
        https://github.com/Harmonic-Percussive-Bandwidth-Extension/HP-codecX.git \
        third_party/hpcodecx
fi

uv venv --clear --python 3.11 third_party/hpcodecx/.venv
PYTHON=third_party/hpcodecx/.venv/bin/python
uv pip install --python "$PYTHON" \
    torch==2.1.2 torchaudio==2.1.2 "numpy<2" \
    -r third_party/hpcodecx/requirements.txt

WEIGHTS=third_party/hpcodecx/runs
mkdir -p "$WEIGHTS/hp-codec/best_finetuning/dac" \
    "$WEIGHTS/hp-codecx/best/transformermodel"
if [ ! -f "$WEIGHTS/hp-codec/best_finetuning/dac/package.pth" ]; then
    curl -L -C - -o "$WEIGHTS/hp-codec/best_finetuning/dac/package.pth" \
        "https://zenodo.org/records/22144249/files/hp-codec.package.pth?download=1"
fi
if [ ! -f "$WEIGHTS/hp-codecx/best/transformermodel/package.pth" ]; then
    curl -L -C - -o "$WEIGHTS/hp-codecx/best/transformermodel/package.pth" \
        "https://zenodo.org/records/22144249/files/hp-codecx.package.pth?download=1"
fi

"$PYTHON" -c "import audiotools, argbind, torch" && echo "hpcodecx venv ready"
