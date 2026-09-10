#!/bin/sh
# Build the AudioSR environment at third_party/audiosr/.venv (gitignored).
#
# AudioSR (pip `audiosr`) pins torch 2.0.1 and transformers 4.30.2, which
# conflict with the project's torch 2.7.1 stack (ADR-0007), so it runs in its
# own venv behind a subprocess -- the A2SB arrangement, without the fork: the
# package needs no code changes, only install-time pins that keep a 2023-era
# release installable today:
#   - numpy<1.24: the librosa version audiosr pulls still uses the
#     np.float alias, removed in numpy 1.24 (torch 2.0.1 also needs <2)
#   - huggingface_hub<0.26: the diffusers version audiosr pulls imports
#     cached_download, removed from the hub client in 0.26
#   - setuptools<81: the librosa version audiosr pulls imports pkg_resources,
#     which uv venvs do not ship and setuptools 81 removed
#   - matplotlib, progressbar, unidecode: imported by the package but
#     missing from its pip metadata (they are in its repo requirements)
#   - soundfile, soxr: used by our driver (scripts/audiosr_restore.py)
set -eu

cd "$(dirname "$0")/.."
mkdir -p third_party/audiosr
uv venv --clear --python 3.11 third_party/audiosr/.venv

# One resolution for everything: installing audiosr in a second command lets
# the resolver quietly upgrade the already-installed torch (measured: 2.0.1
# became 2.14.0), so the torch pins and audiosr must be solved together.
PYTHON=third_party/audiosr/.venv/bin/python
case "$(uname -s)-$(uname -m)" in
Linux-x86_64)
    uv pip install --python "$PYTHON" \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1+cu118 torchaudio==2.0.2+cu118 audiosr==0.0.7 \
        "numpy<1.24" "huggingface_hub<0.26" "setuptools<81" \
        matplotlib progressbar unidecode soundfile soxr ;;
*)
    uv pip install --python "$PYTHON" \
        torch==2.0.1 torchaudio==2.0.2 audiosr==0.0.7 \
        "numpy<1.24" "huggingface_hub<0.26" "setuptools<81" \
        matplotlib progressbar unidecode soundfile soxr ;;
esac

"$PYTHON" -c "import audiosr" && echo "audiosr venv ready"
