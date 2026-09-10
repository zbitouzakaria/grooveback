# 11. Benchmark autoencoder round-trips and mean damage subtraction

Date: 2026-09-10

## Status

Accepted — concluded: apollo → εar-VAE restores heavily damaged records best;
apollo alone stays closer to the master above 64 kbps. The kept set is
published.

## Context

An autoencoder trained on clean music outputs clean-sounding audio, so a
degraded input may come back cleaner from a bare `decode(encode(x))` — and
better latents than `encode(x)` exist to be found, because the encoder is not
the decoder's inverse. ADR-0007 measured this once, with SAME only, at
64/128/192 kbps. Three further open music autoencoders have since released
weights (εar-VAE, εar-VAE2, CoDiCodec), and the bitrates that matter for the
library sit lower. This experiment widens the round-trip comparison to all
four, moves the ladder to 32/64/128 kbps, and adds two inference-time
constructions on top of the round-trip: a mean damage subtraction in latent
space, and the round-trip applied to a supervised restorer's output.

## Decision

### The benchmark

We cut three sources at 44.1 kHz: the 6 s severe codec asset (whole), a 180 s
excerpt of the aerofunk track, and a 180 s excerpt of a third track whose
artist also provides the damage donor below. The
two 48 kHz masters are resampled once to the benchmark grid at ingestion
(soxr VHQ), identically for every method downstream. LAME twins are built at
32, 64 and 128 kbps under the ADR-0007 alignment gate (`best_lag == 0`,
verified at the start, middle and end of every twin); the measured codec
edges on the codec source are 5.5, 11 and 16.5 kHz.

Four autoencoders run in-process, since their dependencies resolve under the
stable-audio-3 torch pin (dependency group `aes`): SAME-L, εar-VAE (the
44.1 kHz checkpoint), εar-VAE2 and CoDiCodec. The εar repositories are
vendored as pinned submodules and CoDiCodec is a pip package. εar-VAE2 and
CoDiCodec are natively 48 kHz, so their wrappers resample 44.1↔48 privately —
the two models that pay a double resample, worth ~−140 dBFS in soxr
artifacts. Apollo and A2SB anchor the comparison exactly as in ADR-0007,
with A2SB walled at the measured edge − 250 Hz at 50 steps.

Renders, twins and originals are stored as float32 WAV, and only the
listening packs — pulled under −1 dBFS by one common gain per pack — are
24-bit FLAC. A first pass stored everything as FLAC, which hard-clips at
±1.0 silently: decoder overshoot (εar-VAE2 reaches a 2.1 peak on a 0.5-peak
tone), loudness matching, MP3 decode ringing, and soxr's intersample
overshoot on 0 dBFS masters all push past full scale, so every file class
touched the ceiling and the set was re-rendered. `ga.save` now refuses audio
past full scale in any integer subtype, which makes this failure loud
instead of silent.

### The chains are solvers

`grooveback.solvers.roundtrip` and `grooveback.solvers.latent_sub` sit next
to `sdedit` behind the Hydra entry point (`model=roundtrip`,
`model=latent-sub`), following ADR-0009's extension pattern. Both return
output at the input's length and integrated loudness, like every solver.
`scripts/run_xp.py` is configured by `configs/benchmark.yaml` and calls the
same solver functions the entry point dispatches to.

### Mean damage subtraction

For each autoencoder M and bitrate b we measure one damage direction on a
donor track — a track by the same artist as the third source, 180 s, never
itself scored:

    d_b = mean_t( encode_M(mp3_b(donor))[:, t] − encode_M(donor)[:, t] )    # one (channels,) vector
    output = decode_M( encode_M(x) − d_b[:, None] )                          # broadcast over frames

The direction points clean → damaged, so subtraction moves toward clean. A
per-frame variant was considered and dropped: unrelated tracks share no beat
grid, so frame t of the donor says nothing about frame t of the input, and
only the average direction can transfer. The damage encodes are
deterministic by construction — SAME's bottleneck is affine, εar-VAE takes
the posterior mean, εar-VAE2 runs its deterministic mode, and CoDiCodec's
encoder draws no noise. The same-artist pairing is the favourable case on
record; the codec and aerofunk sources test transfer across artists.

### Chained renders

Each autoencoder also renders `apollo-{ae}`, the round-trip applied to
Apollo's render rather than to the raw twin: `ae(apollo(x))`. Apollo repairs
in-band damage but remains a regression model, and the chain asks whether a
decoder adds plausible detail on top of its cleaner output.

### A sampled-encode variant, tried once and removed

A first pass also rendered a sampled-posterior encode where the model offers
one (εar-VAE, εar-VAE2). Over every pack and metric the sampled and mean
encodes differed by at most 0.01 dB: both posteriors are effectively
deterministic. The variant is removed from the code and the matrix, and this
note records why it does not come back.

## Results (2026-09-10 run, float-WAV storage)

The renders were produced on an A100 80 GB — the four autoencoders, their
subtraction and chained arms, and apollo; A2SB was kept from the first pass,
whose renders were the least ceiling-touched class at ≤0.02 % of samples —
and scored locally against the verified lag-0 twins. Each table below is one
source at one bitrate; columns are the five metrics against the master, LSD
lower-is-better, **bold green** best in column, **bold red** worst. The
fill-band decomposition lives in `results.json`.

The metrics, ahead of listening, say the following. The ADR-0007 pattern
holds everywhere: in all nine packs, no method beats the untouched input on
any of the three waveform metrics, and only the phase-blind columns reward
restoration. Chaining an autoencoder after apollo helps almost always — the
chained render beats the plain round-trip's LSD in 35 of 36 cases, and on
aerofunk at 32 and 64 kbps `apollo-earvae2` edges out apollo itself (10.35
vs 10.41 and 8.86 vs 8.91 dB), the first rows in this benchmark where
anything passes apollo; apollo keeps the best LSD in the other seven packs.
The mean damage subtraction works on the spectral metrics: it beats its own
round-trip's LSD in 34 of 36 cases and the untouched input's full-band LSD
in 32 of 36, with the gain concentrated in the codec-dead band (on the
same-artist source at 128 kbps, fill-LSD falls from 26.2 on the input to 8.7
after subtraction, where the plain round-trip leaves it at 26.1). Every
failure sits on aerofunk at 128 kbps — the lightest damage on a
cross-artist source, where the donor's direction overcorrects — and the
subtraction leaves the waveform columns essentially unchanged: it adds
spectral plausibility, not waveform fidelity. The damage direction behaves
like a measurement, its norm growing monotonically with compression in every
latent space (εar-VAE 6.2/5.2/3.7, SAME-L 8.2/5.4/2.3, εar-VAE2
3.5/2.7/1.7, CoDiCodec 14.5/9.5/4.5 at 32/64/128 kbps). The round-trips
differ in what they do to the codec-dead band — SAME-L invents top-band
content, εar-VAE leaves the band essentially empty, εar-VAE2 and CoDiCodec
sit between — and CoDiCodec's waveform scores collapse to 5–8 dB BSS-SDR
while its LSD stays VAE-like, because its diffusion decoder re-invents phase
wholesale.

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

**same-artist @ 32k** (edge 5.5 kHz)

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

**same-artist @ 64k** (edge 11 kHz)

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

**same-artist @ 128k** (edge 16.25 kHz)

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

## Listening verdict (2026-09-10, monitors)

Apollo → εar-VAE is the method for really damaged records: it beats apollo
alone on 32 kbps material, splits with it at 64 kbps, and above 64 kbps
apollo alone sounds closer to the master. The mechanism as heard is that
apollo guesses the missing content well, and εar-VAE's decode smooths the
result — which removes the artifacts apollo itself introduces when the
compression is severe. At lighter compression apollo adds few artifacts, and
the same smoothing then moves the sound away from the master.

The mean damage subtraction through εar-VAE stays in the promising set.
Everything else tried — the other autoencoders' round-trips, chains and
subtractions — is worse by ear. A²SB is kept as a comparison point rather
than a contender. The released εar-VAE2 sounds clearly worse than εar-VAE,
consistent with its public-data retrain (its model card shows the open
weights losing to the paper's proprietary ones) and with its 50 Hz STFT bins
in the bass.

The kept set, in solo-key order: ground truth, degraded input, Apollo, A²SB,
Apollo → εar-VAE, εar-VAE − MP3 damage.

### Published set

`docs/listen/` carries the kept set for the codec and aerofunk sources at
32/64/128 kbps on GitHub Pages, as 24-bit FLAC listening copies at −14 LUFS
under one common −1 dBFS gain per pack — about 556 MB of audio committed to
the repository, accepted over the initial 500 MB budget to keep aerofunk at
its full three minutes. The same-artist pair stays out of the published set.
`scripts/publish_listen.py` rebuilds the page, and the audio URLs carry each
file's mtime so a republished pack defeats any cached copy.

## Consequences

- `grooveback.latents` now holds a uniform registry (`load_ae` / `ae_encode`
  / `ae_decode`) over in-process autoencoders and the damage arithmetic
  (`mean_damage`, `subtract_damage`); every wrapper speaks 44.1 kHz
  `(channels, samples)` audio and `(channels, frames)` latents.
- CoDiCodec's import flips global torch backend flags (TF32, cudnn
  benchmark); its loader snapshots and restores them so the other models'
  numerics stay untouched. Its batch size is capped for 22 GiB cards, and
  the benchmark empties the CUDA cache between autoencoders.
- εar-VAE's unchunked 180 s encode exceeds an L4's 22 GiB, and CoDiCodec's
  parallel decode accumulates ~21 GiB regardless of batch size; both render
  on 80 GB cards.
- The old benchmark tree (64/128/192 kbps, WAV) was deleted together with
  the SDEdit renders; `results.json` regenerates from whatever renders
  exist.

## Revisit triggers

- εar-VAE2's authors release weights trained on the paper's full corpus →
  re-render its rows.
- A fine-tuned prior exists → its renders join this scoreboard, which is
  ADR-0007's original purpose.
- NVIDIA releases A2SB's 4-split ensemble → re-render its rows.
