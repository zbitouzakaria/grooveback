# 15. Reproduce Apollo's test protocol on the MUSDB18-HQ test mixtures

Date: 2026-10-20

## Status

Accepted — Apollo, and Apollo followed by εar-VAE, are scored at six MP3 bitrates on the 50 MUSDB18-HQ test mixtures
with the authors' own preprocessing, codec simulation and metrics. One fix to their `codec_simu` keeps every MP3
within one sample of its clean clip.

## Context

Apollo (Li & Luo, ICASSP 2025) is this project's strongest baseline (ADR-0006). The paper reports it on "the combined
test set of MUSDB18-HQ and MoisesDB (with 5000 samples for each case)", MP3-compressed at 24 to 128 kbps. That set was
never released: the authors' `JusperLee/Apollo-val-test-data` on Hugging Face holds only a README, and the paper never
names its MoisesDB songs.

The MUSDB18-HQ test split can be used instead. The maintainer states that it "was not used to train or retrain the
released checkpoint" ([JusperLee/Apollo#46](https://github.com/JusperLee/Apollo/issues/46#issuecomment-6281774888)),
so its 50 songs are a held-out test for Apollo, and later for this project's own methods.

## Decision

`python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps K --earvae` scores, per clip, the untouched MP3,
Apollo, and Apollo → εar-VAE (Apollo's output after an εar-VAE round trip, ADR-0011). Each step is ported from the
authors' released code:

- 6 s clips (the authors' "sessions") of each mixture with a 3 s hop, kept when more than half of their 10 ms frames
  exceed the 15th-percentile power of the track's non-silent frames (`VAD` in Apollo-data-preprocess's
  `musdb_preprocess.py`): 3,677 clips, the same at every bitrate;
- MP3 through torchaudio's sox, shifted back into alignment by the lag that `match2` finds (`codec_simu`), then both
  signals divided by their joint peak (`MusdbMoisesdbDataset.__getitem__`), both in `look2hear/datas`;
- Apollo once over each whole clip (`test.py`);
- SDR with fast_bss_eval and a 512-tap filter, and SI-SNR with fast_bss_eval's `si_sdr`, both zero-mean and averaged
  over channels (`look2hear/metrics/wrapper.py`). This project adds spectral SNR and log-spectral distance (LSD), as
  in ADR-0007.

Departures from the released code and the paper:

- **The alignment fix.** `codec_simu` applies a Hann window to both signals for `match2`'s lag search. As released, it
  leaves some MP3s 25 ms late (below).
- **Whole mixtures.** The paper also scores four categories of stem mixes, which only serve a comparison with its own
  test set. An earlier version rebuilt them from the test stems with the authors' training mixer (random tracks and
  sessions, whole-dB gains in [−10, 10], 250 clips per category, seed 0) and was dropped along with MoisesDB. Its
  per-clip scores and lags are on the network volume `grooveback-US-MO-1`, as `/workspace/apollo-musdb/{k}k-scores.csv`
  and `{k}k-lags.csv`.
- **Metrics in float64.** The authors compute in float32. On the 3,677 clips at 24 kbps, this changes SDR by up to
  1.2 dB on single clips and by 0.001 dB in the mean, and SI-SNR by under 0.002 dB (figure under Their evaluation).

### Exactness gates

Measured with `codec_simu` as released, before the fix.

- **Model and metrics.** Apollo run here on the authors' three 24 kbps demo inputs gives outputs within 76–80 dB SNR of
  their published outputs, with the same SDR and SI-SNR to two decimals.
- **Codec.** On the same three clips, `codec_simu` gives the SDR of their MP3 against the clean clip to within
  0.02–0.17 dB, and the same 4,250 Hz cut-off on the two clips where one can be measured. One of their MP3s is one
  sample late and is compared after shifting it back. Bit-exactness cannot be checked, because their pairs were
  rescaled after encoding. Here the MP3 is encoded through libsox by torchaudio 2.7.1 and 2.8.0; theirs by 2.0.1.
- **Clips.** The authors' unmodified `musdb_preprocess.py` (Apollo-data-preprocess at `bb37a7b`), run on 16-bit WAV
  copies of the 50 test mixtures, stores the same 3,677 clips as `musdb_eval`, at the same positions and identical
  sample for sample.
- **Their evaluation.** The authors' `test.py` and metric code (Apollo at `b17bf53`), on the same clips at 24 kbps,
  give the same means as `musdb_eval`: SDR 11.889 dB (`test.py`) against 11.891 dB, SI-SNR 10.368 dB for both. Per
  clip, SI-SNR is identical, and SDR differs only through float32. `test.py` ran with two changes: the model loads from
  the released checkpoint, since the `look2hear.models.GullFullband` it calls is never defined, and ViSQOL is skipped.
  Its clip pairs were written with the authors' preprocessing and `codec_simu`, as the script that made theirs is
  unpublished. Its per-clip output is `assets/0015/test-py-24k.csv`.

  ![test.py minus musdb_eval, per clip](assets/0015/test-py-agreement.png)
- **Rerun.** 24 kbps scored again on another pod, on the first two tracks (128 clips), reproduces every score to within
  4e-12 dB.

## Results (2026-10-22)

![SDR, SI-SNR, spectral SNR and LSD per bitrate on the MUSDB18-HQ test mixtures, with the published points](assets/0015/bitrates.png)

Apollo → εar-VAE scores worse than Apollo on all four metrics at every bitrate, and changes little with bitrate.
Per-clip scores and MP3 lags are in [assets/0015](assets/0015). The dotted line is Apollo as published, read off the
bitrate figure in Apollo's README (`asserts/bitrates.png`) by calibrating on its gridlines, as the paper prints no
values. It comes from a different test set, MUSDB18-HQ plus an unreleased selection of MoisesDB songs, and is not
comparable with the solid lines.

### `codec_simu` as released leaves some MP3s late

Each clip's lag is measured as the peak of a plain circular cross-correlation between the channel sums of the MP3 and
the clean clip. In the 2026-10-20 run with `codec_simu` as released, the MP3 stayed 1,105 samples (25 ms) late, LAME's
576-sample encoder delay plus the decoder's 529, on 2.5% of the clips at 24 kbps, from 23 of the 50 tracks, and on
fewer at higher bitrates. Two clips at 24 kbps, one of them again at 32 kbps, were shifted by thousands of samples
instead. Late or shifted by thousands, these are the misaligned clips of the figure. On the stem mixes of the earlier
version, 13.3% of the 24 kbps clips were late, mostly single-stem ones. That run's per-clip lags and scores are in
[assets/0015/released](assets/0015/released).

![MP3s more than one sample off per bitrate, before and after the fix; match2's correlation at lag 0 and at the true lag, per clip](assets/0015/alignment.png)

`match2` cross-correlates in the frequency domain with a phase transform: each bin of the cross-spectrum is divided by
its own magnitude, so every frequency has nearly the same weight in the lag estimate, whatever its energy. It then
takes the largest peak in absolute value. The FFT treats each 6 s clip as periodic, so the difference between its last
and first samples becomes a step at position 0, in the clean clip and in the MP3, with energy at every frequency. The
MP3's low-pass cannot remove this step, because it is in the analysis, not in the audio: for white noise low-passed at
4.2 kHz and cut to 6 s, the energy from 4.25 kHz up is −47 dB of the total, and −164 dB with a Hann window. Below the
MP3's cut-off, both signals hold music, which points to 1,105 samples. Above it, the MP3 holds only its step, so a bin
there points to lag 0 wherever the clean clip's own step is stronger than its music. This was not measured band by
band, but what was measured fits it. The edges make a peak at lag 0 on almost every clip: at 24 kbps its sign is the
sign of the two edge jumps multiplied together on all 95 misaligned clips and on 91 of 100 aligned ones. On 93 of the
misaligned clips this peak is higher than the one at 1,105; on the aligned ones it never is (figure, right).

The fix applies the Hann window to both signals for the lag search only; the MP3 itself is shifted untapered. At
24 kbps it finds 1,105 on all 95 misaligned clips, and on 100 aligned clips chosen at random it stays within one sample
of the lag found before. Zero-padding instead fixes 32 of the 95, because the padding makes new edges. With the fix,
every MP3 at every bitrate is within one sample of its clean clip, and all but 28, at 24–32 kbps, are exactly aligned
(figure, left). The 24 kbps diagnostics are in `assets/0015/match2-24k.csv` and `assets/0015/match2-edges-24k.csv`.

With `codec_simu` as released, Apollo moves a late MP3 back into place at low bitrates. At 24–48 kbps, on 177 of the
186 late clips, its output scores within 3 dB SI-SNR of its output on the same clip aligned, while the late MP3 itself
averages −21 dB at 24 kbps. At 64 kbps this holds on 5 of 10 late clips, and at 96–128 kbps on none of 7. This fits
training pairs that carried the same delay: the released training loader makes its pairs with this same `codec_simu`
(`look2hear/datas/musdb_moisesdb_datamodule.py`, line 123). The maintainer wrote otherwise: "Currently, I am using
perfectly synchronized codec and original files"
([JusperLee/Apollo#13](https://github.com/JusperLee/Apollo/issues/13#issuecomment-2511437902)).

A negative SI-SNR does not by itself mean a clip is late: "Little Chicago's Finest - My Own" is aligned at every
bitrate, yet its MP3 scores about 0 dB SI-SNR, and its clips are the only aligned ones below 0 dB.

### Training data

Run unmodified on the first 3 `train/` and the first 3 `test/` tracks, the released `musdb_preprocess.py` writes
`sample1`–`sample3` from the train tracks and `sample4`–`sample6` from the test tracks into every stem folder, as
matched sample for sample against the source WAVs. The training loader picks a file from each folder at random
(`random.choice(os.listdir(...))`, `look2hear/datas/musdb_moisesdb_datamodule.py`, line 104), so the public pipeline
would train on the test split. That the 50 test songs are held out rests on the maintainer's statement that the
released checkpoint was not trained on them, and that the public script does not reflect the training data selected
for it (#46).

## Consequences

- The 50 test mixtures at six bitrates are the project's held-out score for Apollo and for later methods. Scoring
  another method means adding it to `musdb_eval.evaluate`, which runs only Apollo and Apollo → εar-VAE today.
- With `codec_simu` as released, a method could raise its score on the late clips just by shifting the MP3 back into
  place; with the fix it cannot. On those clips, the scores here differ from what the authors' code gives.
- Six bitrates with Apollo → εar-VAE take about an hour on one H100 pod, about $5.

## Revisit triggers

- The authors share the paper's test clips → score Apollo on them directly.
- ViSQOL is needed → add it to `musdb_eval`.
- A method of this project is ready → score it on all six bitrates.
