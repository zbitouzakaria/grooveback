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
- **Storage**: every benchmark artifact is 24-bit FLAC. The quantization
  floor sits ~130 dB under program level; the ADR-0007 codec rows reproduce
  to the second decimal through the new storage, which pins the change as
  metric-transparent.

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

### Sampled-encode variant

Where an encoder can sample its posterior (εar-VAE, εar-VAE2), one extra
seeded round-trip render `{ae}-sample` uses a sampled encode — another data
point on how much of the round-trip's character is the posterior's spread.
Models without the knob refuse rather than silently returning the mean.

## Results (2026-09-10 run)

Rendered overnight on an L4 (SAME-L, εar-VAE2, Apollo) and an A100 80 GB
(εar-VAE and CoDiCodec, whose 180 s passes exceed 22 GiB; A2SB as always),
scored locally. Each table is one source at one bitrate; columns are the five
metrics against the master, LSD lower-is-better, **bold green** best in
column, **bold red** worst. The fill-band decomposition lives in
`results.json`.

Findings, before listening:

- **The ADR-0007 pattern holds everywhere**: in all 9 packs, no method beats
  the untouched input on any of the three waveform metrics. Only the
  phase-blind columns reward restoration.
- **Apollo keeps the best LSD in 9 of 9 packs.** A2SB again changes little
  outside the band it fills.
- **Round-trips differ in what they do to the codec-dead band**: SAME-L
  invents top-band content (fill-LSD well below the input's), εar-VAE leaves
  the band essentially empty (fill-LSD within ±3 dB of the input's silence),
  εar-VAE2 and CoDiCodec sit between. Full-band LSD ranks the round-trips
  accordingly.
- **CoDiCodec's waveform scores collapse** (5–8 dB BSS-SDR, worst cell in
  every pack) while its LSD stays VAE-like: its diffusion decoder re-invents
  phase wholesale. Whether that costs anything is a listening question.
- **Mean damage subtraction works, on the spectral metrics.** The `-sub`
  render beats its own round-trip's LSD in 34 of 36 cases, and beats the
  *untouched input's* full-band LSD in 32 of 36. The gain concentrates
  exactly where the codec removed content — e.g. nesta @ 128k fill-LSD:
  input 25.8, εar-VAE round-trip 26.0, εar-VAE-sub **7.6**. All four
  failures are aerofunk @ 128k: the lightest damage on a cross-artist
  source, where the donor's direction overcorrects. Waveform columns are
  unchanged by the subtraction — it adds spectral plausibility, not
  waveform fidelity.
- **The damage direction is a coherent measurement**: its norm grows
  monotonically with compression in every latent space (e.g. εar-VAE
  6.09 / 4.94 / 3.52 at 32/64/128 kbps).
- **The sampled-encode variant is redundant**: over every pack and metric,
  sampled and mean encodes differ by at most 0.01 dB — both εar posteriors
  are effectively deterministic (collapsed scales). Recorded here so the
  variant is not re-run.
- The same-artist pairing (nesta) shows the largest subtraction gains at
  32k, but the direction also transfers across artists (aerofunk, codec)
  at 32/64k — the transfer fails only where there is little damage left to
  subtract.

Listening on monitors decides what these numbers mean;
`demo/roundtrip` holds the nine level-matched packs.

**aerofunk @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **19.0** 🟢 | **14.8** 🟢 | **14.9** 🟢 | 15.4 | **23.6** 🔴 |
| same-l | 16.5 | 13.7 | 13.6 | 15.0 | 18.0 |
| same-l-sub | 14.4 | 12.5 | 12.4 | 15.2 | 17.2 |
| earvae | 16.8 | 13.6 | 13.6 | 14.8 | 23.2 |
| earvae-sample | 16.8 | 13.6 | 13.6 | 14.8 | 23.2 |
| earvae-sub | 15.2 | 12.8 | 12.7 | 15.2 | 13.1 |
| earvae2 | 13.5 | 11.5 | 11.3 | 13.9 | 23.4 |
| earvae2-sample | 13.5 | 11.5 | 11.3 | 13.9 | 23.4 |
| earvae2-sub | 12.3 | 10.8 | 10.5 | 13.9 | 14.9 |
| codicodec | **5.5** 🔴 | **5.9** 🔴 | **4.7** 🔴 | **9.6** 🔴 | 23.6 |
| codicodec-sub | 5.8 | 6.0 | 4.8 | 9.9 | 13.2 |
| apollo | 16.9 | 14.0 | 13.9 | **16.9** 🟢 | **10.4** 🟢 |
| a2sb | 18.3 | 14.5 | 14.6 | 15.7 | 18.6 |

**aerofunk @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **18.6** 🟢 | **17.0** 🟢 | **17.3** 🟢 | 18.6 | 16.7 |
| same-l | 14.4 | 13.3 | 13.2 | 16.8 | 11.4 |
| same-l-sub | 13.9 | 12.9 | 12.8 | 16.7 | 13.7 |
| earvae | 15.2 | 13.5 | 13.3 | 16.4 | 16.6 |
| earvae-sample | 15.2 | 13.5 | 13.3 | 16.4 | 16.6 |
| earvae-sub | 14.7 | 13.1 | 12.9 | 16.8 | 9.0 |
| earvae2 | 12.5 | 10.9 | 10.6 | 15.0 | 16.9 |
| earvae2-sample | 12.5 | 10.9 | 10.6 | 15.0 | 16.9 |
| earvae2-sub | 11.6 | 10.4 | 10.0 | 14.7 | 11.9 |
| codicodec | **5.2** 🔴 | **5.7** 🔴 | **4.5** 🔴 | **9.8** 🔴 | **17.1** 🔴 |
| codicodec-sub | 5.9 | 6.1 | 5.0 | 10.3 | 11.5 |
| apollo | 16.9 | 15.9 | 15.8 | **20.3** 🟢 | **8.9** 🟢 |
| a2sb | 18.1 | 16.6 | 16.9 | 18.4 | 14.0 |

**aerofunk @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **23.1** 🟢 | **21.4** 🟢 | **23.0** 🟢 | 23.2 | 7.3 |
| same-l | 14.1 | 13.2 | 13.1 | 17.5 | 8.2 |
| same-l-sub | 14.1 | 13.2 | 13.1 | 17.5 | **9.9** 🔴 |
| earvae | 15.2 | 13.7 | 13.5 | 17.7 | 9.6 |
| earvae-sample | 15.2 | 13.7 | 13.5 | 17.7 | 9.6 |
| earvae-sub | 15.2 | 13.7 | 13.5 | 17.7 | 8.0 |
| earvae2 | 12.1 | 10.6 | 10.2 | 15.8 | 9.5 |
| earvae2-sample | 12.1 | 10.6 | 10.2 | 15.8 | 9.5 |
| earvae2-sub | 11.9 | 10.5 | 10.1 | 15.6 | 8.2 |
| codicodec | **5.3** 🔴 | **5.7** 🔴 | **4.6** 🔴 | **10.2** 🔴 | 9.8 |
| codicodec-sub | 5.5 | 5.9 | 4.7 | 10.3 | 8.4 |
| apollo | 20.2 | 19.9 | 20.0 | **24.3** 🟢 | **6.7** 🟢 |
| a2sb | 22.5 | 20.9 | 22.4 | 22.4 | 7.0 |

**codec @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **12.1** 🟢 | **10.2** 🟢 | **9.8** 🟢 | 11.3 | **41.5** 🔴 |
| same-l | 7.8 | 7.4 | 6.5 | 9.9 | 27.7 |
| same-l-sub | 6.5 | 6.5 | 5.4 | 10.0 | 11.2 |
| earvae | 7.9 | 7.1 | 6.3 | 9.4 | 40.8 |
| earvae-sample | 7.9 | 7.1 | 6.3 | 9.4 | 40.8 |
| earvae-sub | 7.1 | 6.7 | 5.7 | 9.9 | 17.0 |
| earvae2 | 5.9 | 5.5 | 4.3 | 9.0 | 41.2 |
| earvae2-sample | 5.9 | 5.5 | 4.3 | 9.0 | 41.2 |
| earvae2-sub | 5.3 | 5.3 | 3.9 | 9.4 | 18.4 |
| codicodec | **-6.8** 🔴 | **-0.7** 🔴 | **-7.9** 🔴 | **4.0** 🔴 | 41.1 |
| codicodec-sub | -6.6 | -0.6 | -7.8 | 4.2 | 21.0 |
| apollo | 5.8 | 6.0 | 5.0 | **12.3** 🟢 | **8.6** 🟢 |
| a2sb | 11.5 | 9.9 | 9.4 | 11.8 | 35.0 |

**codec @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **14.2** 🟢 | **12.6** 🟢 | **12.5** 🟢 | **14.0** 🟢 | **33.1** 🔴 |
| same-l | 7.3 | 7.2 | 6.3 | 10.7 | 19.9 |
| same-l-sub | 7.0 | 6.9 | 6.0 | 10.8 | 10.9 |
| earvae | 7.9 | 7.2 | 6.4 | 10.3 | 32.6 |
| earvae-sample | 7.9 | 7.2 | 6.4 | 10.3 | 32.6 |
| earvae-sub | 7.7 | 7.0 | 6.1 | 10.7 | 17.5 |
| earvae2 | 5.8 | 5.5 | 4.3 | 9.9 | 32.9 |
| earvae2-sample | 5.8 | 5.5 | 4.3 | 9.9 | 32.9 |
| earvae2-sub | 5.5 | 5.3 | 4.0 | 10.0 | 18.6 |
| codicodec | **-6.7** 🔴 | **-0.7** 🔴 | **-7.7** 🔴 | **4.1** 🔴 | 32.7 |
| codicodec-sub | -6.3 | -0.7 | -7.4 | 4.2 | 21.1 |
| apollo | 7.1 | 7.2 | 6.5 | 13.7 | **8.2** 🟢 |
| a2sb | 13.9 | 12.4 | 12.2 | 14.0 | 29.9 |

**codec @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **17.0** 🟢 | **16.3** 🟢 | **16.5** 🟢 | **18.6** 🟢 | 20.5 |
| same-l | 6.9 | 6.8 | 5.9 | 11.3 | 13.5 |
| same-l-sub | 7.0 | 6.9 | 6.0 | 11.4 | 10.0 |
| earvae | 7.8 | 7.1 | 6.3 | 11.2 | **21.2** 🔴 |
| earvae-sample | 7.8 | 7.1 | 6.3 | 11.2 | 21.2 |
| earvae-sub | 7.8 | 7.0 | 6.3 | 11.3 | 10.5 |
| earvae2 | 5.5 | 5.3 | 4.1 | 10.6 | 20.8 |
| earvae2-sample | 5.5 | 5.3 | 4.1 | 10.6 | 20.8 |
| earvae2-sub | 5.5 | 5.3 | 4.1 | 10.6 | 13.7 |
| codicodec | **-6.5** 🔴 | -0.7 | **-7.5** 🔴 | 4.4 | 21.2 |
| codicodec-sub | -6.4 | **-0.7** 🔴 | -7.4 | **4.4** 🔴 | 15.7 |
| apollo | 8.6 | 8.9 | 8.4 | 15.3 | **6.8** 🟢 |
| a2sb | 16.5 | 15.9 | 16.1 | 18.6 | 14.6 |

**nesta @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **17.2** 🟢 | **14.7** 🟢 | **14.8** 🟢 | 15.7 | **28.2** 🔴 |
| same-l | 15.0 | 13.2 | 13.1 | 15.2 | 21.4 |
| same-l-sub | 13.6 | 12.3 | 12.2 | 15.5 | 13.0 |
| earvae | 15.2 | 13.2 | 13.0 | 15.0 | 27.7 |
| earvae-sample | 15.2 | 13.2 | 13.0 | 15.0 | 27.7 |
| earvae-sub | 14.2 | 12.5 | 12.4 | 15.4 | 12.8 |
| earvae2 | 12.6 | 11.1 | 10.7 | 14.4 | 27.7 |
| earvae2-sample | 12.6 | 11.1 | 10.7 | 14.4 | 27.7 |
| earvae2-sub | 11.8 | 10.5 | 10.2 | 14.5 | 13.6 |
| codicodec | **5.6** 🔴 | **6.0** 🔴 | **4.9** 🔴 | **10.1** 🔴 | 27.4 |
| codicodec-sub | 5.7 | 6.0 | 4.9 | 10.3 | 12.9 |
| apollo | 14.7 | 13.3 | 13.1 | **17.7** 🟢 | **10.0** 🟢 |
| a2sb | 16.2 | 14.1 | 14.1 | 16.3 | 22.3 |

**nesta @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **18.8** 🟢 | **17.5** 🟢 | **17.9** 🟢 | 19.3 | **20.8** 🔴 |
| same-l | 13.8 | 12.9 | 12.8 | 16.8 | 14.2 |
| same-l-sub | 13.4 | 12.6 | 12.4 | 16.7 | 10.2 |
| earvae | 14.7 | 13.3 | 13.1 | 17.0 | 20.8 |
| earvae-sample | 14.7 | 13.3 | 13.1 | 17.0 | 20.8 |
| earvae-sub | 14.3 | 13.0 | 12.8 | 17.2 | 10.8 |
| earvae2 | 11.8 | 10.6 | 10.2 | 15.9 | 20.6 |
| earvae2-sample | 11.8 | 10.6 | 10.2 | 15.9 | 20.6 |
| earvae2-sub | 11.2 | 10.2 | 9.8 | 15.7 | 12.2 |
| codicodec | **5.5** 🔴 | **5.9** 🔴 | **4.8** 🔴 | **10.6** 🔴 | 20.5 |
| codicodec-sub | 6.0 | 6.2 | 5.2 | 10.9 | 11.8 |
| apollo | 16.7 | 16.0 | 15.9 | **20.8** 🟢 | **8.5** 🟢 |
| a2sb | 18.7 | 17.4 | 17.8 | 19.4 | 18.5 |

**nesta @ 128k** (edge 16.25 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **23.8** 🟢 | **21.9** 🟢 | **23.7** 🟢 | 23.4 | 10.9 |
| same-l | 13.8 | 13.0 | 12.9 | 17.4 | 10.0 |
| same-l-sub | 13.8 | 13.1 | 12.9 | 17.5 | 9.0 |
| earvae | 14.8 | 13.5 | 13.4 | 17.9 | **12.6** 🔴 |
| earvae-sample | 14.8 | 13.5 | 13.4 | 17.9 | 12.6 |
| earvae-sub | 14.7 | 13.5 | 13.3 | 17.9 | 8.9 |
| earvae2 | 11.6 | 10.5 | 10.1 | 16.6 | 12.2 |
| earvae2-sample | 11.6 | 10.5 | 10.1 | 16.6 | 12.2 |
| earvae2-sub | 11.5 | 10.4 | 10.0 | 16.4 | 9.3 |
| codicodec | **5.7** 🔴 | **6.1** 🔴 | **5.0** 🔴 | **10.9** 🔴 | 12.5 |
| codicodec-sub | 5.8 | 6.1 | 5.1 | 11.0 | 9.9 |
| apollo | 20.2 | 19.9 | 20.0 | **24.6** 🟢 | **6.9** 🟢 |
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
