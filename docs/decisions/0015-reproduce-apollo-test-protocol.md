# 15. Reproduce Apollo's test protocol on the MUSDB18-HQ test mixtures

Date: 2026-10-20

## Status

Accepted — Apollo, and Apollo followed by εar-VAE, are scored at six bitrates on the 50 MUSDB18-HQ test mixtures,
which the released checkpoint was not trained on, with the authors' own preprocessing, codec and metrics. One change to
the codec keeps every MP3 aligned with its clean clip. With the codec as released, the authors' own `test.py` gives the
same scores.

## Context

Apollo (Li & Luo, ICASSP 2025) is this project's strongest baseline (ADR-0006). The paper reports it on "the combined
test set of MUSDB18-HQ and MoisesDB (with 5000 samples for each case)", MP3-compressed at 24, 32, 48, 64, 96 and
128 kbps. That set was never released. The authors' dataset `JusperLee/Apollo-val-test-data` on Hugging Face contains
only a README, and the paper never identifies its MoisesDB songs, so the paper's set cannot be rebuilt.

The MUSDB18-HQ test split can be used instead. The maintainer states that it "was not used to train or retrain the
released checkpoint" ([JusperLee/Apollo#46](https://github.com/JusperLee/Apollo/issues/46#issuecomment-6281774888)),
so its 50 songs are a held-out test for Apollo, and later for this project's own methods.

## Decision

### The test set

`python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps K --earvae` scores the untouched MP3, Apollo and
Apollo → εar-VAE on these songs. Each step is the authors' released code, ported, with one change:

| step | source |
|---|---|
| 6 s sessions of each test mixture, 3 s hop, kept when over half of their 10 ms frames exceed the track's 15th-percentile power | `VAD` in [Apollo-data-preprocess](https://github.com/JusperLee/Apollo-data-preprocess) `musdb_preprocess.py` |
| MP3 through torchaudio's sox at the bitrate, cut to length, rolled into alignment by `match2` | `codec_simu`, `look2hear/datas` |
| both signals divided by their joint peak | same file |
| Apollo once over each whole clip | `test.py` |
| SDR: fast_bss_eval, 512-tap filter, zero mean; SI-SNR: fast_bss_eval `si_sdr`, zero mean; both averaged over channels | `look2hear/metrics/wrapper.py` |

The change is in `match2`'s lag search: both signals are tapered with a Hann window first. As released, the search
leaves some MP3s 25 ms late (see [The alignment failure in the released codec](#the-alignment-failure-in-the-released-codec)).

This project adds two metrics, spectral SNR and log-spectral distance (LSD) as in ADR-0007, and one method,
Apollo → εar-VAE: Apollo's output after an εar-VAE round trip, the chain of ADR-0011.

The released code does not settle three choices:

- **Whole mixtures, not the paper's categories.** The paper splits its clips into four categories of stems. Those
  categories only serve a comparison with the paper's set, which cannot be rebuilt. The mixtures are the songs as
  released, with no mixing choice of this project's own.
- **Metrics in float64.** The authors compute in float32 on the GPU. On their 15 published demo triplets the two differ
  by at most 0.05 dB per clip.
- **The same clips at every bitrate.** The sessions are cut from the clean mixture, so all six bitrates score the same
  3,677 clean clips.

### Exactness gates

Where the published material allows a check. All were measured on 2026-10-20 with `codec_simu` as released, before the
taper.

- **Model and metrics.** Apollo rendered here on the authors' three 24 kbps demo inputs matches their published outputs
  to 76–80 dB. The SDR / SI-SNR of ours and theirs are identical to two decimals: 9.83 / 8.24, 11.27 / 9.32,
  11.82 / 9.66.
- **Codec.** `codec_simu` on the authors' clean demo clips reproduces their MP3's 4,250 Hz edge, and its SDR against
  the clean clip within 0.02–0.17 dB. A bit-exact match cannot be checked, because their pairs were rescaled after
  encoding. The MP3 here goes through the system libsox from torchaudio 2.8.0, not the authors' torchaudio 2.0.1
  build.
- **Sessions.** The authors' unmodified `musdb_preprocess.py` (Apollo-data-preprocess at `bb37a7b`), run on each of
  the 50 test mixtures as a 16-bit WAV copy of its FLAC, stores the same 3,677 sessions as `musdb_eval`. Every session
  starts at the same position and is identical sample for sample. The float32 cast `musdb_eval` applies loses
  nothing, since 16-bit values fit in float32.
- **Their evaluation.** The authors' own `test.py` and metric code (Apollo at `b17bf53`), run on the same 3,677 clips
  at 24 kbps, give the same means as `musdb_eval`: SDR 11.889 against 11.891 dB, SI-SNR 10.368 against 10.368 dB.
  `test.py` cannot run as released: it loads `look2hear.models.GullFullband`, which is never defined, and reads a
  `codec-test` folder whose generating script was never published. Two things were changed: the model loads from the
  released checkpoint as their `inference.py` does, and ViSQOL is skipped. The pairs were written with their own
  `musdb_preprocess.py`, `codec_simu` and joint-peak rescale. Per clip, SI-SNR is identical, and SDR differs only where
  `test.py`'s float32 differs from the float64 here. Their per-clip output is `assets/0015/test-py-24k.csv`.

  ![test.py minus musdb_eval, per clip](assets/0015/test-py-agreement.png)
- **Rerun.** 24 kbps scored a second time on a different pod, on the first two tracks (128 clips), reproduces every
  per-clip score of the first run to within 3e-12 dB.

## Results (2026-10-22 run)

Per-clip SDR, SI-SNR, spectral SNR and LSD of the MP3, Apollo and Apollo → εar-VAE, and the MP3's lag, for the 3,677
clips at each bitrate are in [assets/0015](assets/0015).

![SDR, SI-SNR, spectral SNR and LSD per bitrate on the MUSDB18-HQ test mixtures, with the published points](assets/0015/bitrates.png)

The dotted line, on SDR and SI-SNR only, is Apollo as published. The paper shows bitrates only as a figure (Fig. 2),
with no numbers, so its points are read off the same comparison in Apollo's README (`asserts/bitrates.png`),
calibrated on its gridlines. They come from a different test set, MUSDB18-HQ plus an unreleased selection of MoisesDB
songs, and are not compared with the solid lines.

### The alignment failure in the released codec

`codec_simu` as released rolls the decoded MP3 by the lag that `match2` finds on the first channel. On some clips the
MP3 stays 1,105 samples (25 ms) late: LAME's 576-sample encoder delay plus the decoder's 529. Measured per clip as the
peak of the circular cross-correlation between the channel sums of the MP3 and the clean clip, this happened in the
2026-10-20 run on 93 of the 3,677 clips at 24 kbps (2.5%), from 23 of the 50 tracks. It gets rarer as the bitrate
rises: 68, 25, 10, 5 and 2 clips at 32, 48, 64, 96 and 128 kbps. Two clips at 24 kbps and one at 32 kbps are rolled by
a wrong lag of thousands of samples instead.

`match2` misses because of the clip edges. It cross-correlates the raw clips with a phase transform, which gives every
frequency the same weight. The correlation is circular, so the jump from a clip's last sample back to its first acts
as a sharp step at lag 0, in both the clean clip and the MP3. Above the MP3's cut-off, that step is all the MP3
contains, while the clean clip still holds music there. On clips with loud edges, the peak at lag 0 beats the true
peak at 1,105 samples, `match2` returns 0, and the roll leaves the MP3 late. A lower bitrate cuts off lower, so more
frequencies carry only the step, and more clips miss. Measured at 24 kbps on the 95 misaligned clips and 100 aligned
ones: `match2` returns 0 on 92 of the misaligned clips, where the correlation at lag 0 reaches 0.85 against about
0.12 at the true lag. On aligned clips it is about 0.001. Tapering both signals with a Hann window before the
transform finds 1,105 samples on all 95 misaligned clips and keeps the 100 aligned clips within one sample of their
lag.

`codec_simu` now applies this taper. Measured the same way over all clips, every MP3 is then within one sample of its
clean clip: 26 clips at 24 kbps and 2 at 32 kbps are one sample off, and all others are at 0.

Apollo's training pairs carried the same delay. Apollo moves the late MP3 back into place on 195 of the 196 late
clips at 24–64 kbps, and a model only learns that from training pairs that were late in the same way. The maintainer
wrote otherwise:
"Currently, I am using perfectly synchronized codec and original files"
([comment](https://github.com/JusperLee/Apollo/issues/13#issuecomment-2511437902)).

A negative SI-SNR is not a reliable sign of a late clip. On "Little Chicago's Finest - My Own" (88 clips) the MP3 is
aligned, with both of its channels at lag 0, yet it scores about 0 dB SI-SNR at every bitrate. The track's two
channels are nearly uncorrelated, with a correlation of 0.02 over the whole track. Its clips are the only aligned
ones with negative SI-SNR, 54 of them at 24 kbps.

### Training data

The released preprocessing puts MUSDB18-HQ's test stems in the same folders as its training stems. This was measured
by running `musdb_preprocess.py` from Apollo-data-preprocess at `bb37a7b`, unmodified, on the first 3 `train/` and the
first 3 `test/` tracks of MUSDB18-HQ. It writes one folder per stem, `sample1.hdf5` to `sample6.hdf5`, with no train
or test level. Matched sample for sample against the source WAVs, `sample1`–`sample3` are the train tracks and
`sample4`–`sample6` the test tracks in every stem folder. The training loader picks files with
`random.choice(os.listdir(...))` (`look2hear/datas/musdb_moisesdb_datamodule.py`, line 104), so it would sample both
alike.

The maintainer states that the test split was not used for the released checkpoint, and that the public script does
not reflect the training data actually selected for it
([JusperLee/Apollo#46](https://github.com/JusperLee/Apollo/issues/46#issuecomment-6281774888)).

### Stem mixes, tried and replaced

The first version of this set rebuilt the paper's four categories from MUSDB18-HQ test stems, with the authors'
training mixer: stems from random tracks and sessions, each gained by a whole number of dB in [−10, 10], 250 clips
per category, seed 0. On it the MP3 stayed late on 13.3% of the clips at 24 kbps, mostly single-stem clips. It was
replaced when MoisesDB was dropped, because the categories only served a comparison with the paper's set. Its
per-clip scores and lags are on the network volume `grooveback-US-MO-1`, as `/workspace/apollo-musdb/{k}k-scores.csv`
and `{k}k-lags.csv`.

## Consequences

- The 50 test mixtures at six bitrates are the project's held-out score for Apollo, and for later methods.
  `musdb_eval` scores Apollo, and Apollo → εar-VAE with `--earvae`; scoring another method means passing it in.
- `codec_simu` departs from the authors' code in the taper, so on the clips their code leaves late, the scores here
  differ from what their code gives. A late MP3 scores far below 0 dB SI-SNR, −21 dB on average at 24 kbps, and a
  method that only moved it back into place would gain on it without restoring anything.
- One bitrate, Apollo only, takes about 90 minutes on an L40S pod. Bitrates run side by side on one pod as separate
  processes: five ran in 90 minutes, for about $1.65, with `OMP_NUM_THREADS=2` each. Torch on a pod sees all of the
  host's cores (128–256), while the container is allowed 13–27, so the threads are capped to fit.

## Revisit triggers

- The authors share the paper's test clips → score Apollo on them directly.
- ViSQOL is needed → add it to `musdb_eval`.
- A method of this project is ready → score it on all six bitrates.
