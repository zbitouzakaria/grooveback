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

# The weight files live on the version-specific record 22144250; the concept
# record's /records/<id>/files/... path serves an HTML landing page that curl
# happily saves, so every download is size-checked afterwards.
WEIGHTS=third_party/hpcodecx/runs
ZENODO=https://zenodo.org/api/records/22144250/files
mkdir -p "$WEIGHTS/hp-codec/best_finetuning/dac" \
    "$WEIGHTS/hp-codecx/best/transformermodel"
fetch() {
    if [ ! -f "$1" ] || [ "$(wc -c < "$1")" -lt 100000000 ]; then
        curl -L -C - -o "$1" "$ZENODO/$2/content"
    fi
    if [ "$(wc -c < "$1")" -lt 100000000 ]; then
        echo "$1 came back tiny — an error page, not a checkpoint" >&2
        exit 1
    fi
}
fetch "$WEIGHTS/hp-codec/best_finetuning/dac/package.pth" hp-codec.package.pth
fetch "$WEIGHTS/hp-codecx/best/transformermodel/package.pth" hp-codecx.package.pth

# The Zenodo packages froze train-time classes that predate the repo's
# inference helpers; convert them to the weights format audiotools also
# loads, which instantiates the current repo code instead (ADR-0013).
( cd third_party/hpcodecx && .venv/bin/python ../../scripts/hpcodecx_convert.py )

"$PYTHON" -c "import audiotools, argbind, torch" && echo "hpcodecx venv ready"
