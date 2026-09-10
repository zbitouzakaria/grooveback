#!/bin/sh
# Build the AudioSR environment at third_party/audiosr/.venv (gitignored).
#
# AudioSR (pip `audiosr`) pins torch 2.0.1 and transformers 4.30.2, which
# conflict with the project's torch 2.7.1 stack (ADR-0007), so it runs in its
# own venv behind a subprocess -- the A2SB arrangement, without the fork: the
# package needs no code changes, only install-time pins that keep a 2023-era
# release installable today:
#   - numpy<2: torch 2.0.1 was built against the numpy 1 ABI
#   - huggingface_hub<0.26: the diffusers version audiosr pulls imports
#     cached_download, removed from the hub client in 0.26
#   - setuptools<81: the librosa version audiosr pulls imports pkg_resources,
#     which uv venvs do not ship and setuptools 81 removed
#   - soundfile, soxr: used by our driver (scripts/audiosr_restore.py)
set -eu

cd "$(dirname "$0")/.."
mkdir -p third_party/audiosr
uv venv --clear --python 3.11 third_party/audiosr/.venv

PYTHON=third_party/audiosr/.venv/bin/python
case "$(uname -s)-$(uname -m)" in
Linux-x86_64)
    uv pip install --python "$PYTHON" \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1+cu118 torchaudio==2.0.2+cu118 ;;
*)
    uv pip install --python "$PYTHON" torch==2.0.1 torchaudio==2.0.2 ;;
esac
uv pip install --python "$PYTHON" \
    audiosr==0.0.7 "numpy<2" "huggingface_hub<0.26" "setuptools<81" soundfile soxr

"$PYTHON" -c "import audiosr" && echo "audiosr venv ready"
