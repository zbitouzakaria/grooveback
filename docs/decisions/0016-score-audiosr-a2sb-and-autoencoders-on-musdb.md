# 16. Score AudioSR, A2SB, εar-VAE and εar-VAE2 on the MUSDB18-HQ test mixtures

Date: 2026-10-24

## Status

Accepted — on 50 clips of the MUSDB18-HQ test songs, none of the four scores above the untouched MP3 on SDR or SI-SNR
at any bitrate. A2SB comes closest, and scores about 0.5 dB above Apollo at 32 to 64 kbps.

## Context

ADR-0015 scores Apollo, and Apollo followed by εar-VAE, on MP3 copies of the 50 MUSDB18-HQ test songs, with the authors'
own protocol. We had scored three more open methods against a master only on a few excerpts, ours and Apollo's test
clip: A2SB (ADR-0006, ADR-0007), and the εar-VAE and εar-VAE2 round trips of the MP3 (ADR-0011). AudioSR, an open model
from the README's survey that fills in the band above a cut-off, had never run in this repository. We scored all four
with ADR-0015's harness, at the same six bitrates.

## Results (2026-10-23)

![SDR, SI-SNR, spectral SNR and LSD per MP3 bitrate on 50 MUSDB18-HQ test clips, for six restorations and the untouched MP3](assets/0016/methods.png)

Every line is a mean over the same 50 clips, drawn at random from ADR-0015's 3,677 (see below). We did not run Apollo
and Apollo → εar-VAE again: their lines are ADR-0015's scores of those 50 clips.

### No method scores above the untouched MP3 on SDR

The untouched MP3 scores 13.2 dB SDR at 24 kbps and 19.4 dB at 128 kbps. All four methods score below it at every
bitrate, on SI-SNR too, by more than the draw's noise. A2SB comes closest, 0.1 to 0.7 dB under the MP3, and about
0.5 dB above Apollo at 32 to 64 kbps. AudioSR scores 2.0 to 5.4 dB under the MP3. A2SB fills the band above a cut-off
250 Hz under the MP3's, as in ADR-0011, with one cut-off per bitrate: the most common among the 50 clips.

The spectral scores rank the methods differently from SDR. Apollo has the highest spectral SNR and the lowest LSD at
every bitrate. A2SB and AudioSR have a lower LSD than the MP3, but not a higher spectral SNR: AudioSR's is below the
MP3's at every bitrate, and A2SB's within 0.35 dB of it.

### The εar-VAE and εar-VAE2 round trips do not improve with the bitrate

While the MP3 rises from 13.2 to 19.4 dB SDR between 24 and 128 kbps, the round trip of the MP3 through εar-VAE stays
between 10.4 and 10.7 dB, and through εar-VAE2 between 8.1 and 8.5 dB. Apollo → εar-VAE stays between 10.6 and
10.9 dB. The autoencoder sets the score, not the damage. Their LSD stays within 1.4 dB of the MP3's, so neither fills
the band the MP3 removed.

## We scored 50 clips, because the diffusion models are slow

AudioSR and A2SB are diffusion models. On one L40S GPU, AudioSR takes 7.3 s per 6 s stereo clip and A2SB 37 s. For
A2SB that is about 3 s per second of audio and channel, the same order as in ADR-0006, where it took 35 min for a
7.6 min mono track on an A100. Running three A2SB processes at once on the GPU did not render more clips per hour. All
3,677 clips at six bitrates hold 37 hours of stereo audio, which would take about 45 hours of AudioSR and 230 hours of
A2SB. So we drew 50 clips at random, from 32 of the 50 songs, and scored every method on the same 50 at every bitrate.

ADR-0015 scored the MP3, Apollo and Apollo → εar-VAE on all 3,677 clips, so for them we can see what the draw does. It
moves their mean SDR by up to 0.4 dB: Apollo's SDR at 24 kbps is 12.2 dB on the 50 clips and 11.9 dB on all. It moves
their gap to the MP3 less. Apollo's gap is −0.93 dB on the 50 and −0.86 dB on all, and the two agree within 0.07 dB at
every bitrate; Apollo → εar-VAE's gap moves by up to 0.34 dB. Over 50 clips, the standard error of a method's SDR gap
to the MP3 is at most 0.5 dB, and under 0.07 dB for A2SB. So the figure compares methods, and ADR-0015 keeps the
reference numbers. Our MP3 scores of the 50 clips match ADR-0015's to 1e-10 dB, so its Apollo scores are of the same
clips and the same MP3s.

`musdb_eval --clips 50` takes the first 50 clips of one fixed random order of all 3,677. A later `--clips 100` keeps
these 50 and adds 50 more.

## Consequences

- AudioSR, A2SB and every autoencoder in the registry can be scored on any number of these clips, or on all of them.
  `musdb_eval` writes the MP3s it scores and scores renders made in another environment, and `scripts/` holds a driver
  for each of the two diffusion models.
- This run took about 4 hours of one L40S at $1.09 per hour, about $4.40. A2SB alone took 3.2 hours.

## Revisit triggers

- Two methods are closer than the draw's noise and the difference matters → run `--clips 100`, which keeps these 50.
- Blind listening is done (ADR-0011's verdicts were not blind) → compare it with these scores.
- A method of this project is ready → score it on this draw and on all clips.
