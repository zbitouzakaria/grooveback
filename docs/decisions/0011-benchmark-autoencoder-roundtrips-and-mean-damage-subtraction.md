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

## Results (2026-09-10 run, float-WAV storage)

Rendered on an A100 80 GB (all four autoencoders, their subtraction and
apollo-chained arms, and apollo itself; a2sb kept from the first pass — the
least ceiling-touched render class at ≤0.02 % of samples, re-rendering it
costs an A100 afternoon). Scored locally against the verified lag-0 twins.
Each table is one source at one bitrate; columns are the five metrics
against the master, LSD lower-is-better, **bold green** best in column,
**bold red** worst. The fill-band decomposition lives in `results.json`.

Findings, before listening:

- **The ADR-0007 pattern holds everywhere**: in all 9 packs, no method beats
  the untouched input on any of the three waveform metrics. Only the
  phase-blind columns reward restoration.
- **Chaining an autoencoder after apollo helps almost always**: `apollo-{ae}`
  beats the plain round-trip's LSD in 35 of 36 cases (the exception is
  same-l on aerofunk @ 128k). On aerofunk at 32 and 64 kbps,
  `apollo-earvae2` edges out apollo itself (10.35 vs 10.41 and 8.86 vs
  8.91 dB) — the first rows in this benchmark where anything passes apollo,
  though by margins listening may not confirm. Apollo alone keeps the best
  LSD in the other 7 packs.
- **Mean damage subtraction works, on the spectral metrics.** The `-sub`
  render beats its own round-trip's LSD in 34 of 36 cases and the
  *untouched input's* full-band LSD in 32 of 36; the gain concentrates in
  the codec-dead band (nesta @ 128k fill-LSD: input 26.2, εar-VAE
  round-trip 26.1, εar-VAE-sub **8.7**). Every failure sits on
  aerofunk @ 128k (plus same-l on aerofunk @ 64k against its own
  round-trip): the lightest damage on a cross-artist source, where the
  donor's direction overcorrects. Waveform columns are essentially
  unchanged by the subtraction — it adds spectral plausibility, not
  waveform fidelity.
- **The damage direction is a coherent measurement**: its norm grows
  monotonically with compression in every latent space (εar-VAE
  6.2 / 5.2 / 3.7, SAME-L 8.2 / 5.4 / 2.3, εar-VAE2 3.5 / 2.7 / 1.7,
  CoDiCodec 14.5 / 9.5 / 4.5 at 32/64/128 kbps).
- **Round-trips differ in what they do to the codec-dead band**: SAME-L
  invents top-band content, εar-VAE leaves the band essentially empty
  (fill-LSD within ±3 dB of the input's silence), εar-VAE2 and CoDiCodec
  sit between.
- **CoDiCodec's waveform scores collapse** (5–8 dB BSS-SDR, worst cell in
  every pack) while its LSD stays VAE-like: its diffusion decoder
  re-invents phase wholesale. Whether that costs anything is a listening
  question.
- The same-artist pairing (nesta) shows the largest subtraction gains at
  32k, but the direction also transfers across artists — the transfer
  fails only where there is little damage left to subtract.

Listening on monitors decides what these numbers mean; `demo/roundtrip`
holds the nine 16-track level-matched packs.

**aerofunk @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **19.0** 🟢 | **14.8** 🟢 | **14.9** 🟢 | 15.4 | **23.6** 🔴 |
| same-l | 16.5 | 13.7 | 13.6 | 15.0 | 18.0 |
| same-l-sub | 14.4 | 12.5 | 12.4 | 15.2 | 17.2 |
| apollo-same-l | 15.4 | 13.1 | 12.9 | 16.1 | 10.6 |
| earvae | 16.8 | 13.6 | 13.6 | 14.8 | 23.2 |
| earvae-sub | 15.3 | 12.8 | 12.7 | 15.2 | 13.1 |
| apollo-earvae | 15.7 | 13.2 | 12.9 | 15.8 | 11.2 |
| earvae2 | 13.5 | 11.5 | 11.3 | 13.9 | 23.5 |
| earvae2-sub | 12.3 | 10.8 | 10.5 | 13.9 | 15.0 |
| apollo-earvae2 | 12.7 | 11.1 | 10.8 | 14.6 | **10.3** 🟢 |
| codicodec | **5.5** 🔴 | **5.9** 🔴 | **4.7** 🔴 | **9.6** 🔴 | 23.6 |
| codicodec-sub | 5.8 | 6.0 | 4.8 | 9.9 | 13.4 |
| apollo-codicodec | 5.7 | 6.0 | 4.9 | 10.1 | 10.6 |
| apollo | 16.8 | 14.0 | 13.9 | **16.8** 🟢 | 10.4 |
| a2sb | 18.3 | 14.5 | 14.6 | 15.7 | 18.6 |

**aerofunk @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **18.6** 🟢 | **17.0** 🟢 | **17.3** 🟢 | 18.6 | 16.7 |
| same-l | 14.4 | 13.3 | 13.2 | 16.8 | 11.4 |
| same-l-sub | 13.9 | 12.9 | 12.8 | 16.7 | 13.8 |
| apollo-same-l | 13.9 | 12.9 | 12.7 | 17.4 | 9.8 |
| earvae | 15.3 | 13.5 | 13.3 | 16.4 | 16.6 |
| earvae-sub | 14.7 | 13.1 | 12.9 | 16.8 | 9.2 |
| apollo-earvae | 14.7 | 13.2 | 12.9 | 17.2 | 10.0 |
| earvae2 | 12.5 | 10.9 | 10.6 | 15.0 | 17.1 |
| earvae2-sub | 11.6 | 10.4 | 10.0 | 14.7 | 12.0 |
| apollo-earvae2 | 11.8 | 10.4 | 10.0 | 15.3 | **8.9** 🟢 |
| codicodec | **5.2** 🔴 | 5.7 | **4.5** 🔴 | **9.8** 🔴 | **17.2** 🔴 |
| codicodec-sub | 5.8 | 6.0 | 5.0 | 10.3 | 11.4 |
| apollo-codicodec | 5.3 | **5.6** 🔴 | 4.5 | 10.2 | 9.6 |
| apollo | 16.9 | 15.9 | 15.8 | **20.3** 🟢 | 8.9 |
| a2sb | 18.1 | 16.6 | 16.9 | 18.4 | 14.0 |

**aerofunk @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **23.1** 🟢 | **21.4** 🟢 | **23.0** 🟢 | 23.2 | 7.3 |
| same-l | 14.1 | 13.2 | 13.1 | 17.5 | 8.2 |
| same-l-sub | 14.1 | 13.2 | 13.1 | 17.5 | **9.9** 🔴 |
| apollo-same-l | 13.8 | 13.1 | 12.8 | 17.8 | 8.9 |
| earvae | 15.2 | 13.7 | 13.5 | 17.7 | 9.6 |
| earvae-sub | 15.2 | 13.7 | 13.5 | 17.7 | 8.2 |
| apollo-earvae | 14.8 | 13.4 | 13.2 | 17.6 | 9.1 |
| earvae2 | 12.1 | 10.6 | 10.2 | 15.8 | 9.6 |
| earvae2-sub | 11.9 | 10.5 | 10.1 | 15.6 | 8.1 |
| apollo-earvae2 | 11.7 | 10.3 | 9.9 | 15.5 | 8.1 |
| codicodec | 5.3 | 5.7 | 4.6 | 10.2 | 9.9 |
| codicodec-sub | 5.5 | 5.9 | 4.7 | 10.3 | 8.5 |
| apollo-codicodec | **5.2** 🔴 | **5.6** 🔴 | **4.5** 🔴 | **10.2** 🔴 | 8.7 |
| apollo | 20.2 | 19.9 | 20.0 | **24.3** 🟢 | **6.7** 🟢 |
| a2sb | 22.5 | 20.9 | 22.4 | 22.4 | 7.0 |

**codec @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **12.1** 🟢 | **10.2** 🟢 | **9.8** 🟢 | 11.3 | **41.5** 🔴 |
| same-l | 7.8 | 7.4 | 6.5 | 9.9 | 27.7 |
| same-l-sub | 6.5 | 6.5 | 5.4 | 10.0 | 11.2 |
| apollo-same-l | 4.3 | 4.8 | 3.5 | 10.2 | 9.2 |
| earvae | 7.9 | 7.1 | 6.3 | 9.4 | 40.8 |
| earvae-sub | 7.1 | 6.7 | 5.7 | 9.9 | 16.5 |
| apollo-earvae | 4.7 | 4.8 | 3.7 | 9.9 | 9.4 |
| earvae2 | 5.8 | 5.5 | 4.3 | 9.0 | 41.2 |
| earvae2-sub | 5.3 | 5.2 | 3.9 | 9.4 | 18.1 |
| apollo-earvae2 | 3.3 | 3.8 | 2.3 | 9.5 | 12.2 |
| codicodec | -6.8 | -0.7 | -7.9 | **4.0** 🔴 | 41.1 |
| codicodec-sub | -6.6 | -0.6 | -7.8 | 4.2 | 18.6 |
| apollo-codicodec | **-6.9** 🔴 | **-0.9** 🔴 | **-8.0** 🔴 | 4.2 | 11.9 |
| apollo | 5.8 | 6.0 | 5.0 | **12.3** 🟢 | **8.6** 🟢 |
| a2sb | 11.5 | 9.9 | 9.4 | 11.8 | 35.0 |

**codec @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **14.2** 🟢 | **12.6** 🟢 | **12.5** 🟢 | **14.0** 🟢 | **33.1** 🔴 |
| same-l | 7.3 | 7.2 | 6.3 | 10.8 | 19.9 |
| same-l-sub | 7.0 | 6.9 | 6.0 | 10.8 | 10.8 |
| apollo-same-l | 4.5 | 5.0 | 3.8 | 10.4 | 9.2 |
| earvae | 7.9 | 7.2 | 6.4 | 10.3 | 32.6 |
| earvae-sub | 7.7 | 7.0 | 6.1 | 10.7 | 16.2 |
| apollo-earvae | 5.0 | 5.0 | 4.0 | 10.1 | 9.4 |
| earvae2 | 5.8 | 5.5 | 4.3 | 9.9 | 32.9 |
| earvae2-sub | 5.4 | 5.3 | 4.0 | 10.0 | 17.6 |
| apollo-earvae2 | 3.5 | 3.9 | 2.4 | 9.6 | 12.3 |
| codicodec | -6.7 | -0.7 | -7.7 | 4.1 | 32.7 |
| codicodec-sub | -6.4 | -0.7 | -7.5 | 4.2 | 16.9 |
| apollo-codicodec | **-7.2** 🔴 | **-1.1** 🔴 | **-8.3** 🔴 | **4.1** 🔴 | 12.1 |
| apollo | 7.1 | 7.2 | 6.5 | 13.7 | **8.2** 🟢 |
| a2sb | 13.9 | 12.4 | 12.2 | 14.0 | 29.9 |

**codec @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **17.0** 🟢 | **16.3** 🟢 | **16.5** 🟢 | **18.6** 🟢 | 20.5 |
| same-l | 6.9 | 6.8 | 5.9 | 11.3 | 13.5 |
| same-l-sub | 7.0 | 6.9 | 6.0 | 11.4 | 10.0 |
| apollo-same-l | 4.8 | 5.2 | 4.1 | 10.6 | 8.9 |
| earvae | 7.8 | 7.1 | 6.3 | 11.2 | **21.2** 🔴 |
| earvae-sub | 7.8 | 7.0 | 6.3 | 11.3 | 9.7 |
| apollo-earvae | 5.2 | 5.2 | 4.2 | 10.2 | 9.0 |
| earvae2 | 5.5 | 5.3 | 4.1 | 10.6 | 20.8 |
| earvae2-sub | 5.5 | 5.3 | 4.1 | 10.6 | 13.4 |
| apollo-earvae2 | 3.6 | 4.0 | 2.5 | 9.8 | 11.7 |
| codicodec | -6.5 | -0.7 | -7.5 | 4.4 | 21.2 |
| codicodec-sub | -6.4 | -0.7 | -7.5 | 4.4 | 13.7 |
| apollo-codicodec | **-7.1** 🔴 | **-1.1** 🔴 | **-8.2** 🔴 | **4.2** 🔴 | 11.7 |
| apollo | 8.6 | 8.9 | 8.4 | 15.3 | **6.8** 🟢 |
| a2sb | 16.5 | 15.9 | 16.1 | 18.6 | 14.6 |

**nesta @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **17.2** 🟢 | **14.7** 🟢 | **14.8** 🟢 | 15.7 | **28.5** 🔴 |
| same-l | 15.0 | 13.2 | 13.1 | 15.2 | 21.4 |
| same-l-sub | 13.6 | 12.3 | 12.2 | 15.5 | 13.0 |
| apollo-same-l | 13.4 | 12.2 | 11.9 | 16.5 | 10.1 |
| earvae | 15.2 | 13.2 | 13.0 | 15.0 | 27.9 |
| earvae-sub | 14.2 | 12.5 | 12.4 | 15.5 | 12.4 |
| apollo-earvae | 14.0 | 12.4 | 12.2 | 16.4 | 10.7 |
| earvae2 | 12.6 | 11.1 | 10.7 | 14.4 | 28.2 |
| earvae2-sub | 11.8 | 10.6 | 10.2 | 14.6 | 13.7 |
| apollo-earvae2 | 11.6 | 10.4 | 10.1 | 15.6 | 10.3 |
| codicodec | **5.6** 🔴 | **6.0** 🔴 | **4.9** 🔴 | **10.1** 🔴 | 27.8 |
| codicodec-sub | 5.7 | 6.0 | 4.9 | 10.3 | 12.9 |
| apollo-codicodec | 5.8 | 6.1 | 5.0 | 10.8 | 10.2 |
| apollo | 14.7 | 13.2 | 13.1 | **17.7** 🟢 | **10.0** 🟢 |
| a2sb | 16.2 | 14.1 | 14.1 | 16.3 | 22.3 |

**nesta @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **18.8** 🟢 | **17.5** 🟢 | **17.9** 🟢 | 19.3 | 21.0 |
| same-l | 13.8 | 12.9 | 12.8 | 16.8 | 14.2 |
| same-l-sub | 13.4 | 12.6 | 12.4 | 16.7 | 10.3 |
| apollo-same-l | 13.4 | 12.6 | 12.3 | 17.4 | 9.1 |
| earvae | 14.7 | 13.3 | 13.2 | 17.0 | 20.9 |
| earvae-sub | 14.4 | 13.0 | 12.8 | 17.3 | 10.3 |
| apollo-earvae | 14.1 | 12.9 | 12.7 | 17.6 | 9.7 |
| earvae2 | 11.8 | 10.6 | 10.2 | 15.9 | **21.1** 🔴 |
| earvae2-sub | 11.2 | 10.2 | 9.8 | 15.7 | 11.9 |
| apollo-earvae2 | 11.3 | 10.3 | 9.9 | 16.3 | 9.2 |
| codicodec | **5.5** 🔴 | 6.0 | **4.8** 🔴 | **10.6** 🔴 | 20.8 |
| codicodec-sub | 6.0 | 6.2 | 5.2 | 10.9 | 11.0 |
| apollo-codicodec | 5.6 | **5.9** 🔴 | 4.9 | 10.8 | 9.3 |
| apollo | 16.7 | 16.0 | 15.9 | **20.8** 🟢 | **8.6** 🟢 |
| a2sb | 18.8 | 17.4 | 17.8 | 19.4 | 18.5 |

**nesta @ 128k** (edge 16.25 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **23.9** 🟢 | **21.9** 🟢 | **23.7** 🟢 | 23.4 | 11.0 |
| same-l | 13.8 | 13.0 | 12.8 | 17.4 | 10.0 |
| same-l-sub | 13.8 | 13.1 | 12.9 | 17.4 | 9.0 |
| apollo-same-l | 13.4 | 12.8 | 12.5 | 17.7 | 8.6 |
| earvae | 14.8 | 13.5 | 13.4 | 17.9 | **12.6** 🔴 |
| earvae-sub | 14.7 | 13.5 | 13.3 | 17.9 | 8.9 |
| apollo-earvae | 14.3 | 13.2 | 13.0 | 17.9 | 9.2 |
| earvae2 | 11.6 | 10.5 | 10.1 | 16.6 | 12.5 |
| earvae2-sub | 11.5 | 10.4 | 10.0 | 16.4 | 9.3 |
| apollo-earvae2 | 11.3 | 10.2 | 9.8 | 16.5 | 8.7 |
| codicodec | 5.7 | 6.1 | 5.0 | 10.9 | 12.6 |
| codicodec-sub | 5.8 | 6.1 | 5.1 | 11.0 | 9.5 |
| apollo-codicodec | **5.5** 🔴 | **5.9** 🔴 | **4.8** 🔴 | **10.8** 🔴 | 9.1 |
| apollo | 20.2 | 19.9 | 20.1 | **24.6** 🟢 | **7.0** 🟢 |
| a2sb | 23.8 | 21.9 | 23.7 | 23.4 | 9.8 |

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
