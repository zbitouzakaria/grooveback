# 11. Benchmark autoencoder round-trips and mean damage subtraction

Date: 2026-09-10

## Status

Proposed — metrics in; the listening verdict is pending

## Context

The decoder-as-restorer hypothesis (README, Future work): an autoencoder
trained on clean music outputs clean-sounding audio, so a degraded input may
come back cleaner from a bare `decode(encode(x))` — and better latents than
`encode(x)` exist to be found, since the encoder is not the decoder's inverse.
ADR-0007 measured the round-trip once, with SAME only, at 64/128/192 kbps.
Three more open music autoencoders now exist with released weights (εar-VAE,
εar-VAE2, CoDiCodec), and the bitrates that matter for the library sit lower.

## Decision

### The benchmark

- **Sources** (chunks at 44.1 kHz): `codec` (the 6 s severe asset, whole),
  `aerofunk` (180 s from 0:00), `nesta` (Nesta — James Bande (Edit), 180 s
  from 0:00, same artist as the damage donor). The Nesta masters are 48 kHz
  PCM_24 and are resampled once to the 44.1 kHz benchmark grid at ingestion
  (soxr VHQ), identically for every method downstream.
- **Bitrates**: 32/64/128 kbps LAME twins, the ADR-0007 alignment gate
  unchanged (`best_lag == 0`). Measured edges: 5.5 / 11 / 16.5 kHz on codec.
- **Autoencoders**: SAME-L, εar-VAE (44.1 kHz checkpoint), εar-VAE2
  (natively 48 kHz — its wrapper resamples 44.1↔48, the one model paying a
  double resample, ~−140 dBFS in soxr artifacts), CoDiCodec (the shipped
  checkpoint is 48 kHz too, same treatment; its open weights process joint
  stereo). All four run in-process: their dependencies resolve under the
  stable-audio-3 torch pin (dependency group `aes`); εar-VAE and εar-VAE2 are
  vendored as submodules, CoDiCodec is a pip package.
- **Anchors**: Apollo and A2SB exactly as in ADR-0007 (A2SB walled at the
  measured edge − 250 Hz, 50 steps).
- **Storage**: renders, twins and originals are float32 WAV; only the
  listening packs (pulled under −1 dBFS by one common gain) are 24-bit FLAC.
  A first pass stored everything as FLAC, which silently hard-clips at ±1.0:
  decoder overshoot (εar-VAE2 peaks at 2.1 on a 0.5-peak tone), loudness
  matching, MP3 decode ringing, and even soxr's intersample overshoot on the
  0 dBFS 48 kHz masters all pushed past full scale, so every file class
  touched the ceiling and the whole set was re-rendered. `ga.save` now
  refuses audio past full scale in any integer subtype, so this failure is
  loud instead of silent.

### The chains are solvers

`grooveback.solvers.roundtrip` and `grooveback.solvers.latent_sub` sit next
to `sdedit` behind the Hydra entry point (`model=roundtrip`,
`model=latent-sub`), per ADR-0009's extension pattern. Both return output at
the input's length and integrated loudness (gain only), like every solver.
`scripts/run_xp.py` is Hydra-configured (`configs/benchmark.yaml`) and calls
the same solver functions.

### Mean damage subtraction

Per autoencoder M and bitrate b, on the donor track (Nesta — Bad Hoe Running
(edit), 180 s from 0:00, same artist as the `nesta` source):

    d_b = mean_t( encode_M(mp3_b(donor))[:, t] − encode_M(donor)[:, t] )    # one (channels,) vector
    output = decode_M( encode_M(x) − d_b[:, None] )                          # broadcast over frames

d_b points clean → damaged, so subtraction moves toward clean. A per-frame
variant was considered and dropped: unrelated tracks share no beat grid, so
frame t of the donor says nothing about frame t of the input — only the
average direction can transfer. The damage encodes are deterministic by
construction (SAME's bottleneck is affine; εar-VAE takes the posterior mean;
εar-VAE2 runs `deterministic=True`; CoDiCodec's encoder is deterministic).
The same-artist pairing (nesta restored with the donor's direction) is the
favourable case on record; codec and aerofunk test transfer across artists.

### Chained renders

Each autoencoder also renders `apollo-{ae}` — the round-trip run on Apollo's
render instead of the raw twin: `ae(apollo(input))`. Apollo repairs in-band
damage but stays a regression model; the chain asks whether a decoder adds
plausible detail on top of its cleaner output.

### Sampled-encode variant: tried once, removed

A first pass also rendered `{ae}-sample` (a sampled posterior encode where
the model has one — εar-VAE, εar-VAE2). Over every pack and metric the
sampled and mean encodes differed by at most 0.01 dB: both posteriors are
effectively deterministic (collapsed scales). The variant is removed from
the code and the matrix; this note is why it does not come back.

## Results

*(re-measuring: the first pass's numbers and listening packs were corrupted
by the FLAC clipping described under Storage; the full re-render is in
progress and the tables land here)*

## Consequences

- `grooveback.latents` grew a uniform registry (`load_ae` / `ae_encode` /
  `ae_decode`) over in-process autoencoders, plus the damage arithmetic
  (`mean_damage`, `subtract_damage`); every wrapper speaks 44.1 kHz
  `(channels, samples)` audio and `(channels, frames)` latents.
- CoDiCodec's import flips global torch backend flags (TF32, cudnn
  benchmark); its loader snapshots and restores them so other models'
  numerics stay untouched.
- The old benchmark tree (64/128/192 kbps, WAV) was deleted with the SDEdit
  renders; `results.json` regenerates from whatever renders exist.

## Revisit triggers

- εar-VAE2's authors release weights trained on the paper's full corpus →
  re-render its rows.
- A fine-tuned prior exists → its renders join this scoreboard (ADR-0007's
  original purpose).
