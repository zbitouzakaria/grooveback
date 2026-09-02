# 7. Benchmark codec restoration on MP3 twins

Date: 2026-09-01

## Status

Accepted

## Context

The project's goal is to restore old MP3 rips with a generative model trained
on clean music of the same genre (ADR-0004). That model would live inside the
SAME autoencoder's latent space, so it can never sound better than what
survives a SAME round-trip — and before any model work, ADR-0005 requires
knowing what the existing tools already achieve and what that round-trip
costs.

All of that needs one scoreboard. Real rips have no clean reference to score
against, so the benchmark makes its own damage: compress clean chunks to MP3
and score every method against the original.

## Decision

One script (`scripts/run_xp.py`) runs everything; metrics live in
`grooveback.evaluation`; one notebook (`notebooks/xp.ipynb`) shows one source
at a time.

- **Sources**: one chunk each of two clean files — aerofunk (12 s at 1:00)
  and the codec asset (6 s whole).
- **Damage**: LAME MP3 at 64, 128 and 192 kbps, decoded back to wav. A twin
  even one sample out of alignment is refused (`best_lag == 0`) — that alone
  ruins a waveform metric.
- **Methods**: the SAME round-trip (`decode(encode(x))`, S and L variants),
  Apollo, and A2SB. The untouched MP3 is scored too, as the degraded input —
  the do-nothing floor every method must beat.
- **Metrics**, all against the master: BSS-eval SDR (via fast_bss_eval, what
  papers publish — a 512-tap filter of the master is fitted first, so gain,
  EQ and small delays are forgiven), plain SDR (every waveform difference
  counts), SI-SNR (gain forgiven), spectral SNR (spectrogram magnitudes, so
  phase costs nothing), and log-spectral distance (lower is better — the gap
  between the two log spectrograms, the standard in bandwidth-extension
  papers). Plus spectrograms and level-matched listening sets.
- **The band the codec removed is also scored on its own**, above the
  measured codec edge, in waveform SDR, spectral SNR and LSD. The degraded
  input is silence there — 0 dB on both SNRs — so these columns show directly
  whether an invented top end carries information. The two spectral views
  punish opposite sins: spectral SNR charges filling too much, LSD charges
  filling too little.
- **A2SB is told the measured codec edge** rather than using its own cutoff
  detection, which is built for the blurry rolloffs of real rips; on a sharp
  synthetic edge it cuts below the edge and deletes real content.
- **SAME comes from the `stable-audio-3` dependency**, pinned by commit,
  which pins torch to 2.7.1 for the whole project. A2SB keeps its own
  environment behind a subprocess — its dependency pins conflict with the
  project's.
- Everything renders in one pass, on a GPU pod by default; a step whose
  output already exists is skipped.

## Results (2026-09-02 run)

Each table is one source at one bitrate; columns are the metrics, all in dB against the master. LSD is lower-is-better; the rest higher. **Bold green** = best in column, **bold red** = worst.

**aerofunk @ 64k**

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **19.4** 🟢 | **17.6** 🟢 | **18.1** 🟢 | 19.2 | **17.1** 🔴 |
| a2sb | 18.8 | 17.2 | 17.6 | 18.9 | 14.2 |
| apollo | 15.9 | 15.3 | 15.2 | **20.1** 🟢 | **9.2** 🟢 |
| same-l | 14.8 | 13.7 | 13.6 | 17.3 | 11.5 |
| same-s | **13.4** 🔴 | **12.2** 🔴 | **12.0** 🔴 | **15.9** 🔴 | 11.1 |

**aerofunk @ 128k**

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **24.2** 🟢 | **22.1** 🟢 | **24.1** 🟢 | 23.7 | 7.4 |
| a2sb | 23.4 | 21.5 | 23.3 | 22.7 | 7.2 |
| apollo | 21.1 | 20.6 | 20.8 | **24.6** 🟢 | **6.7** 🟢 |
| same-l | 14.6 | 13.7 | 13.6 | 17.9 | **8.3** 🔴 |
| same-s | **13.2** 🔴 | **12.2** 🔴 | **11.9** 🔴 | **16.2** 🔴 | 8.2 |

**aerofunk @ 192k**

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **31.8** 🟢 | **28.1** 🟢 | **31.8** 🟢 | 29.0 | 5.2 |
| a2sb | 28.5 | 26.0 | 28.3 | 26.4 | **5.1** 🟢 |
| apollo | 27.9 | 27.7 | 27.8 | **29.8** 🟢 | 5.8 |
| same-l | 14.8 | 13.9 | 13.8 | 18.5 | 8.1 |
| same-s | **13.3** 🔴 | **12.3** 🔴 | **12.1** 🔴 | **16.5** 🔴 | **8.2** 🔴 |

**codec @ 64k**

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **14.2** 🟢 | **12.6** 🟢 | **12.5** 🟢 | **14.0** 🟢 | **33.1** 🔴 |
| a2sb | 13.9 | 12.4 | 12.2 | 14.0 | 29.9 |
| apollo | 7.1 | 7.2 | 6.5 | 13.7 | **8.2** 🟢 |
| same-l | 7.3 | 7.2 | 6.3 | 10.8 | 19.9 |
| same-s | **5.9** 🔴 | **6.0** 🔴 | **4.9** 🔴 | **9.6** 🔴 | 21.6 |

**codec @ 128k**

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **17.0** 🟢 | **16.3** 🟢 | **16.5** 🟢 | **18.6** 🟢 | **20.5** 🔴 |
| a2sb | 16.5 | 15.9 | 16.1 | 18.6 | 14.6 |
| apollo | 8.6 | 8.9 | 8.4 | 15.3 | **6.8** 🟢 |
| same-l | 6.9 | 6.8 | 5.9 | 11.3 | 13.5 |
| same-s | **5.5** 🔴 | **5.7** 🔴 | **4.5** 🔴 | **9.9** 🔴 | 13.9 |

**codec @ 192k**

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | **20.9** 🟢 | **20.1** 🟢 | **20.4** 🟢 | **22.5** 🟢 | **15.9** 🔴 |
| a2sb | 20.0 | 19.4 | 19.6 | 22.1 | 12.3 |
| apollo | 19.5 | 17.7 | 18.5 | 20.6 | **5.9** 🟢 |
| same-l | 7.0 | 6.9 | 6.0 | 11.5 | 10.7 |
| same-s | **5.5** 🔴 | **5.7** 🔴 | **4.6** 🔴 | **9.9** 🔴 | 11.5 |

The fill-band decomposition lives in `results.json` and the notebook.

## Consequences

- On the three waveform metrics nothing beats the do-nothing floor: restored
  audio is never closer to the master, sample by sample, than the MP3 it
  started from. Only the phase-blind metrics reward restoration. The tables
  rank fidelity; listening judges how it sounds (audio.md).
- Apollo sitting under the floor matches its own paper, which reports
  absolute output scores and never the unprocessed input's. On the authors'
  released demo files their outputs also score at or below their inputs
  under their own metric (mean −0.7 dB over 15 samples), and this pipeline
  reproduces their renders to 75–81 dB.
- The torch pin follows stable-audio-3. Apollo runs on 2.7.1 — verified by
  the suite and a render — and anything that needs a newer torch conflicts
  with the prior's stack.
- Synthetic twins are cleaner than real rips: no re-encodes, no unknown
  encoder chain. Results bound the easy case, not the library case.

## Revisit triggers

- Later work needs a torch newer than the stable-audio-3 pin.
