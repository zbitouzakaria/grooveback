---
name: apollo-beats-a2sb
description: "Apollo beat A2SB by ear on a 4 kHz brick-walled test, plus the A2SB gotchas found getting it to run"
metadata: 
  node_type: memory
  type: project
  originSessionId: d5c93125-9700-44bb-967a-f3423d0ebd90
  modified: 2026-08-16T18:18:34.187Z
---

On 2026-08-15, listening to `codec_wav` brick-walled at 4 kHz and reconstructed by both, Apollo was clearly better
than A2SB. That is on A2SB's home turf — a sharp low cutoff is what it is built for, and bandwidth extension is not
what Apollo is for. Apollo is also ~110x faster (3.2s vs ~350s for 6s of audio), stereo-native, and measured 5-9 dB
hotter across 4-22 kHz.

**Why:** Apollo is the baseline to beat for ADR-0005, and A2SB has not earned its cost. Re-testing A2SB needs a
specific reason, because each run is expensive on a 16 GB Air.

**How to apply:** A2SB now lives in a fork — github.com/zbitouzakaria/diffusion-audio-restoration, branch
`runnable-anywhere` — cloned gitignored at `third_party/a2sb` with its own venv (`setup.sh`), entry point `restore.py`.
Grooveback's `run_a2sb` is a thin subprocess wrapper that logs the fork SHA. The fork handles all three gotchas below
internally (knee detection, brick-walling, mono). Its memory fix (keep only the final diffusion step) makes single-pass
full-track inference fit in ~7.3 GB, so there is no segmentation anywhere. Validated bit-exact: upstream vanilla ==
old embedded integration == fork == fork-through-wrapper on codec_wav.

The original integration-era gotchas, still true of the model itself:

1. **Feed it a brick wall.** It is trained only on `UpsampleMask`, which zeroes whole FFT bins. Given a real smeared
   codec rolloff it reads the taper as natural and extends nothing. Cut sharply below the knee first.
2. **Give the cutoff explicitly, detected on the whole file.** A quiet few seconds shows no cliff and reads as full
   bandwidth. Trained range is 2000-16000 Hz.
3. It is **mono** — `librosa.to_mono` on load — so a stereo file silently comes back mono.

Ruled out as causes during debugging: sampling steps (20 vs 50 identical), `predict_batch_size` (output-neutral, only
sets chunk count in `get_multidiffusion_vf` — but must be capped around 2 or MPS runs out of memory).

**The checkpoint split DOES matter** (earlier "identical" finding was measured in the do-nothing regime and was wrong):
the 1-split checkpoint paints a flat ~-45 dB shelf across the extended band; the 2-split ensemble rolls off naturally
(-46 → -92 dB) and matches upstream's own inference bit-exactly. `a2sb_checkpoints()` now defaults to the ensemble.
The wrapper was validated 0.00 dB in every band against a vanilla clone run — divergences since are settings, not bugs.

The 4 kHz listening verdict (Apollo way better) predates the ensemble fix and compared against the flat-shelf 1-split
output, so A2SB's fair listening comparison is still open. The speed and mono conclusions stand regardless.

Open: Apollo's output peaked at +2.4 dBFS after level matching, which clips in any integer playback path. See
[[apollo-crackle-on-an2]].
