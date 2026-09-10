# 12. AudioSR as a bandwidth-extension baseline

Date: 2026-09-10

## Status

Proposed — renders and the listening verdict pending

## Context

AudioSR ([Liu et al., arXiv:2309.07314](https://arxiv.org/abs/2309.07314))
performs diffusion super-resolution in the mel domain with a vocoder, taking
any input bandwidth between 2 and 16 kHz to a 48 kHz output. It is the last
open-weights restoration method in the README survey without a "Tested here"
mark, and the second bandwidth-extension model on the board after A2SB
(ADR-0006), which fills the removed band conservatively. Whether a mel-domain
diffusion model invents a more convincing top end than a Schrödinger bridge
is a question the benchmark can answer directly.

Two properties of the paper's inference shape the expectation. First, the
paper prescribes no sampling values and ablates none, so the released
defaults are the authors' operating point; the benchmark runs them
unchanged. Second, inference replaces the spectrogram below the detected
0.99 roll-off frequency with the input's own content, in both the latent
output and the vocoder output — the kept band should therefore track the
degraded input, and the columns that can differ are the fill band and the
listening verdict. ADR-0007's decomposition bounds what any fill is worth on
the waveform metrics (at most ~0.5 dB); the phase-blind fill columns and the
ear are where this method can distinguish itself.

## Decision

### Environment

The pip release `audiosr==0.0.7` pins torch 2.0.1 and transformers 4.30.2,
which conflict with the project-wide torch 2.7.1 pin (ADR-0007). It
therefore runs in its own virtual environment behind a subprocess — the
A2SB arrangement — but without a fork: the package needs no code changes,
so `scripts/audiosr_setup.sh` builds `third_party/audiosr/.venv` from pip
with the install-time pins that keep a 2023 release installable (`numpy<2`,
`huggingface_hub<0.26`). Should an upstream code change ever become
unavoidable, the recorded escape hatch is a fork whose git URL replaces the
pip source in the setup script.

### The driver

`scripts/audiosr_restore.py` is ours and lives in this repository, so its
tests run in CI. It accepts any sample rate and returns the input's rate,
length and channel count; the model's native format (48 kHz mono) is private
to the driver, which resamples once with soxr VHQ before chunking, restores
each channel separately, and resamples back. The release has no long-audio
path, so the driver chunks: 10.24 s chunks (two of the model's 5.12 s
grains) cut on input boundaries, each render trimmed to its chunk before the
model's padding can accumulate, consecutive chunks joined by a 0.1 s linear
crossfade. The fade is equal-gain rather than equal-power so that identical
content passes through exactly — the identity gate of ADR-0002 — and its
cost on independently invented content is measured below rather than
assumed, per the chunking rule in `.claude/rules/audio.md`.

Sampling knobs left unset are not passed to the model at all, so the pinned
package's own defaults apply; `configs/model/audiosr.yaml` exposes them as
nulls behind the Hydra entry point (`model=audiosr`), which also gains
`model=apollo` so every waveform baseline is launchable the same way.

### The benchmark

`audiosr` joins `apollo` and `a2sb` as an anchor in `scripts/run_xp.py`,
with no wall at the codec edge: A2SB is walled because its own cutoff
detection provably misreads sharp synthetic edges (ADR-0007); AudioSR runs
as released, and what its roll-off detection does with a LAME edge is a
finding of this experiment. Concurrently the matrix narrows to the ADR-0011
kept set — sources codec and aerofunk at 32/64/128 kbps, εar-VAE arms and
the three anchors — and the losing ADR-0011 renders are deleted from
`artifacts/xp`; they remain reproducible from that ADR's configuration, and
its tables remain the record. The local comparison page `demo/xp` (formerly
`demo/roundtrip`) carries the kept set plus AudioSR in the published
solo-key order.

## Measurements before trust

To be recorded here from the run:

- **Installed-source read** (0.0.7): the actual sampling defaults, the
  presence of the paper's low-band replacement, and whether the short
  inference path clips or peak-rescales input past full scale — the twins
  carry MP3-decode overshoot above ±1.0, and a clip would call for a
  down-only pre-gain in the driver (a contingency, not a default).
- **Seam consistency**: the 6 s codec source at 64 kbps rendered whole
  versus forced-chunked (`--chunk-seconds 2.56`), same seed; per-1 kHz band
  energy above the codec edge and LSD between the two renders. Agreement
  within ~1 dB per band keeps the fade; a real power loss switches the join
  to an unfaded cut at the boundary.
- **Alignment and level**: `best_lag(render, input) == 0` (a vocoder frame
  shift would need compensation before any waveform metric is meaningful),
  render peaks not pinned at ±1.0, and the render's integrated loudness
  against the input's — a systematic drift would add a gain match in the
  wrapper before scoring.

## Results

Pending: the five ADR-0007 metrics and fill-band decomposition for the six
packs, and the monitor listening verdict against apollo, a2sb and the
εar-VAE arms.

## Consequences

- Every open-weights method in the survey that this benchmark can run is now
  either scored on it or excluded on record; HP-codecX (fixed 16→48 kHz,
  applicable only at 32 kbps) is the remaining open release, deferred.
- The benchmark's standing matrix is the kept set; the full autoencoder
  comparison lives in ADR-0011.

## Revisit triggers

- An upstream code change becomes unavoidable → fork, and point the venv
  install at the fork.
- Sampling settings become a question → sweep through the Hydra entry point
  (`cli.run -m model=audiosr model.ddim_steps=…`), the ADR-0009 pattern.
- AudioSR releases a stereo or 44.1 kHz-native checkpoint.
