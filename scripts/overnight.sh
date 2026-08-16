#!/usr/bin/env bash
# Sequenced overnight work. Nothing runs concurrently: two A2SB processes do not
# fit in 16 GB of unified memory.
#
#   1. wait for the full AN-2 run already in flight
#   2. vanilla A2SB via their own dataset script, on mono_codec_wav_cut4k
#   3. our wrapper on the same input with matching settings
#   4. compare the two, and analyse AN-2
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
A2SB_PY="$REPO/.venvs/a2sb/bin/python"
LOG="$REPO/artifacts/overnight.log"
mkdir -p "$REPO/artifacts"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

AN2_PID="${1:-}"
if [ -n "$AN2_PID" ]; then
  say "waiting for AN-2 run (pid $AN2_PID)"
  while kill -0 "$AN2_PID" 2>/dev/null; do sleep 60; done
  say "AN-2 finished"
fi

# --- vanilla, through their dataset script -------------------------------
# Their config ships placeholder checkpoint paths and a gpu/ddp/SLURM trainer,
# so gb_vanilla.yaml is the same file with only paths and device changed.
say "vanilla A2SB starting"
cd "$REPO/third_party/a2sb/inference"
PYTHONPATH="$REPO/third_party/shims" "$A2SB_PY" A2SB_upsample_dataset.py \
  -dn gbcodec -exp gb_vanilla -cf 4000 >> "$LOG" 2>&1
cd "$REPO"
VANILLA=$(find "$REPO/third_party/a2sb/inference/exp" -name "recon.wav" | head -1)
say "vanilla output: ${VANILLA:-NONE}"

# --- ours, matching vanilla's settings: 2-split, 50 steps, cutoff 4000 ---
say "our wrapper starting"
uv run python -m grooveback.cli.baseline \
  data/mono_codec_wav_cut4k.wav artifacts/a2sb/mono_codec_ours_2split.wav \
  --method a2sb --steps 50 --cutoff-hz 4000 --ensemble >> "$LOG" 2>&1
say "our wrapper done"

# --- compare -------------------------------------------------------------
VANILLA="$VANILLA" uv run python scripts/compare_overnight.py >> "$LOG" 2>&1
say "ALL DONE"
