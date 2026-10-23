# 16. Score AudioSR, A2SB, εar-VAE and εar-VAE2 on the MUSDB18-HQ test mixtures

Date: 2026-10-24

## Status

Accepted — on 50 clips of the MUSDB18-HQ test songs, none of the four scores above the untouched MP3 on SDR or SI-SNR
at any bitrate. A2SB comes closest, and scores about 0.5 dB above Apollo at 32 to 64 kbps. AudioSR rewrites part of
the band the MP3 kept.

## Context

ADR-0015 scores Apollo, and Apollo followed by εar-VAE, on MP3 copies of the 50 MUSDB18-HQ test songs, with the authors'
own protocol. We had scored three more open methods against a master only on a few excerpts, ours and Apollo's test
clip: A2SB (ADR-0006, ADR-0007), and the εar-VAE and εar-VAE2 round trips of the MP3 (ADR-0011). AudioSR, an open model
from the README's survey that fills in the band above a cut-off, had never run in this repository. We scored all four
with ADR-0015's harness, at the same six bitrates.

## Results (2026-10-23)

![SDR, SI-SNR, spectral SNR and LSD per MP3 bitrate on 50 MUSDB18-HQ test clips, for six restorations and the untouched MP3](assets/0016/methods.png)

Every line is a mean over the same 50 clips, drawn at random from ADR-0015's 3,677 (see below). Apollo and Apollo →
εar-VAE are ADR-0015's scores of those clips.

### No method scores above the untouched MP3 on SDR

The untouched MP3 scores 13.2 dB SDR at 24 kbps and 19.4 dB at 128 kbps. All four methods score below it at every
bitrate, on SI-SNR too, by more than the draw's noise. The spectral scores rank the methods differently from SDR.
Apollo has the highest spectral SNR and the lowest LSD at every bitrate. A2SB and AudioSR have a lower LSD than the
MP3, but not a higher spectral SNR: AudioSR's is below the MP3's at every bitrate, and A2SB's within 0.35 dB of it.

### A2SB stays closest to the MP3

A2SB keeps the band below its cut-off, which we set 250 Hz under the MP3's, and fills the band above it. Its SDR is
0.1 to 0.7 dB under the MP3's at every bitrate. To score the band it keeps, we low-passed the clean clip, the MP3 and
the render at that cut-off and took a plain SDR, without the 512-tap filter of the authors' SDR. There A2SB is 0.06 to
0.23 dB under the MP3. The band it adds lowers the LSD most at 24 kbps, from 33.1 dB for the MP3 to 23.0 dB. On SDR,
A2SB scores about 0.5 dB above Apollo at 32 to 64 kbps, and less than 0.25 dB from it at the other bitrates.

### AudioSR rewrites part of the band the MP3 kept

AudioSR scores 2.0 to 5.4 dB under the MP3 on SDR, most at 48 kbps. It detects a cut-off on its input and replaces the
spectrum above it with its own. On these MP3s, the cut-off it detects is lower than the codec's. We compared each
render with its MP3 in 250 Hz bands: at 64 kbps, on the median clip, the render first differs from its MP3 by more than
1 dB at 8 kHz, while the MP3's cut-off is at 11 kHz. Below the MP3's cut-off, on the same plain SDR as above, AudioSR
is 0.7 to 2.4 dB under the MP3. It still has a lower LSD than the MP3, 26.7 against 33.1 dB at 24 kbps, because it
fills the band the MP3 removed.

### The εar-VAE and εar-VAE2 round trips do not improve with the bitrate

While the MP3 rises from 13.2 to 19.4 dB SDR between 24 and 128 kbps, the round trip of the MP3 through εar-VAE stays
between 10.4 and 10.7 dB, and through εar-VAE2 between 8.1 and 8.5 dB. Apollo → εar-VAE stays between 10.6 and
10.9 dB. The autoencoder sets the score, not the damage. Their LSD stays within 1.4 dB of the MP3's, so neither fills
the band the MP3 removed.

## We scored 50 clips, because the diffusion models are slow

AudioSR and A2SB are diffusion models. On one L40S GPU, AudioSR takes 7.3 s per 6 s stereo clip and A2SB 37 s.
Running three A2SB processes at once on the GPU did not render more clips per hour. All 3,677 clips at six bitrates
would take about 45 hours of AudioSR and 230 hours of A2SB. So we drew 50 clips at random, from 32 of the 50 songs,
and scored every method on the same 50 at every bitrate.

The draw moves each method's SDR by up to 0.4 dB. For the two methods we also scored on all clips, it moves their SDR
gap to the MP3 by less. Apollo's SDR at 24 kbps is 12.2 dB on the 50 clips and 11.9 dB on all 3,677, but its gap to
the MP3 is −0.93 dB on the 50 and −0.86 dB on all, and the two gaps agree within 0.07 dB at every bitrate. Apollo →
εar-VAE's gap moves by up to 0.34 dB. Over 50 clips,
the standard error of a method's SDR gap to the MP3 is at most 0.5 dB, and under 0.07 dB for A2SB. So the figure
compares methods, and ADR-0015 keeps the reference numbers. The MP3 scores of the 50 clips match ADR-0015's run to
1e-10 dB: both runs saw the same clips and the same MP3s.

`musdb_eval --clips 50` takes the first 50 clips of one fixed random order of all 3,677. A later `--clips 100` keeps
these 50 and adds 50 more.

## How each method runs

AudioSR and A2SB pin dependencies that conflict with this project's, so each runs in its own environment, on the MP3s
that `musdb_eval` writes. `musdb_eval` then scores their renders, and runs the autoencoders itself. For each bitrate K,
with C the A2SB cut-off below:

```bash
python -m grooveback.cli.musdb_eval musdb18hq/test mp3-K.csv --kbps K --no-apollo --clips 50 --mp3-dir mp3/K
third_party/audiosr/.venv/bin/python scripts/audiosr_restore.py mp3/K audiosr/K
third_party/a2sb/.venv/bin/python scripts/a2sb_restore.py mp3/K a2sb/K --cutoff-hz C
python -m grooveback.cli.musdb_eval musdb18hq/test scores-K.csv --kbps K --no-apollo --clips 50 \
    --render audiosr=audiosr/K --render a2sb=a2sb/K --ae earvae --ae earvae2
```

- **AudioSR** 0.0.7, the `basic` checkpoint, at 50 DDIM steps, the default of its command-line tool (its Python
  function defaults to 200), with the package's seed and guidance scale. The model is mono and 48 kHz, so each
  channel is resampled to 48 kHz and back. The package scales each input to a peak of 0.5, so the driver scales each
  render back to its input's level. `scripts/audiosr_setup.sh` builds its environment.
- **A2SB**, the 2-split checkpoints at 50 steps. As in ADR-0011, it keeps everything below 250 Hz under the MP3's
  cut-off and fills the rest. For each bitrate we took the most common MP3 cut-off among the 50 clips, 4.25, 5.5, 7.5,
  11, 15.25 and 16.5 kHz from 24 to 128 kbps, so C was 4, 5.25, 7.25, 10.75, 15 and 16.25 kHz. At 48 kbps, 22 of the
  50 clips cut off at 7.75 kHz instead. The fork's
  inverse STFT returns a whole number of 512-sample hops, which left the last 408 samples of each 6 s clip silent, so
  the driver pads each clip to whole hops first.
- **εar-VAE and εar-VAE2**: the round trip of the MP3, with the deterministic encode, as in ADR-0011.

## Consequences

- AudioSR, A2SB and every autoencoder in the registry can be scored on any number of these clips, or on all of them,
  with the commands above.
- This run took about 4 hours of one L40S at $1.09 per hour, about $4.40. A2SB alone took 3.2 hours.

## Revisit triggers

- Two methods are closer than the draw's noise and the difference matters → run `--clips 100`, which keeps these 50.
- Blind listening is done (ADR-0011's verdicts were not blind) → compare it with these scores.
- A method of this project is ready → score it on this draw and on all clips.
