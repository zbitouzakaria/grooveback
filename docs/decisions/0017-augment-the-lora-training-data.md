# 17. Augment the LoRA's training data

Date: 2026-10-29

## Status

Proposed. We propose that the next long run of the LoRA fine-tune add four augmentations:

- remixes with one stem of each kind;
- a random EQ on the clean window;
- a left/right swap;
- a second codec, Opus or AAC, on top of half of the MP3s.

Every other proposal is listed below. Each one is deferred until a measurement asks for it, moved to inference, or
rejected, with the reason.

## Context

The LoRA learns to restore MP3s from pairs of a clean window and its MP3 (README, "LoRA fine-tune"). Half of the
windows come from the 86 MUSDB18-HQ training mixtures, and half are remixes of their stems. Every MP3 is LAME CBR at
one of six bitrates, coded by torchaudio's sox backend: the codec of ADR-0015's test. So the model has seen one
encoder, six cut-off frequencies (4.29 to 16.74 kHz), and the balance and dynamics of MUSDB's masters.

The material we want to restore is wider. The README's typical rip is a record played on someone's turntable, saved
as a 128 or 192 kbps MP3, then re-encoded by YouTube to AAC or Opus. Our one real rip, the AN-2 MP3, is 192 kbps,
yet its content stops near 15.7 kHz. LAME's 192 kbps low-pass sits at 18.74 kHz, so the file was probably coded
twice; we infer that from the spectrum alone. Three digital house masters are also louder than MUSDB. Their
peak-to-loudness ratio (PLR) is 9.0 to 9.8 dB, against a median of 13.2 dB on 45 MUSDB test clips.

We looked for augmentations in two research passes. The first covered EQ, stems, stereo, pitch and tempo, and
codecs. The second covered what the first skipped: clip length, vinyl and transfer damage, mastering dynamics, and
the recipes of published restorers. For every proposal, a second reviewer re-checked each claim, and several numbers
below are its corrections. The measurements come from research scripts on a few clips each. These scripts are not in
the repository. The tests of the implementation measure again what the accepted items rely on.

## We judged every proposal by what it does to the clean target

ADR-0004 sets the rule. The clean target must stay a plausible master, and randomness goes only where the MP3
carries evidence of it.

An augmentation applied before coding changes the target and the MP3 together, so the model still only undoes the
codec. Damage added to the MP3 alone is different: the clean original stays the target, so the model learns to
change the master. It learns to de-limit, to de-reverberate, or to brighten. That is remastering, and every proposal
of that kind is rejected below.

## We propose four augmentations for the next run

### Remixes hold one stem of each kind

Today a remix follows Apollo's recipe. It sums 1 to 4 stems drawn with repeats from the four kinds, each stem from
its own random song and scaled by a whole-dB gain of −10 to +10 dB. So 44% of remixes stack two stems of one kind,
such as two bass lines in different keys, and only 2% hold all four kinds.

Demucs, the reference recipe on MUSDB, builds every remix from all four kinds. It takes one stem of each from a
different song, and scales each by a random factor of 0.25 to 1.25, which is −12 to +1.9 dB
([config](https://github.com/facebookresearch/demucs/blob/main/conf/config.yaml),
[augment.py](https://github.com/facebookresearch/demucs/blob/main/demucs/augment.py)). MERL measured what repeats
cost in separation ([Jeon et al. 2026](https://www.merl.com/publications/TR2026-012), Table 7). Sampling every stem
twice lowered average SDR from 8.2 to 7.6 dB, and sampling bass twice lowered it to 7.3 dB.

From now on, a remix draws one stem of each kind, each from its own random song. In one remix out of four, one kind,
picked at random, is left out. Each stem is scaled by a random factor of 0.8 to 1.2 (−1.9 to +1.6 dB). That range is
narrower than Demucs's on purpose: a remix keeps roughly the balance each stem had in its own song.

### A random EQ changes the master's balance, before coding

With probability 0.5, the whole window goes through one random EQ before it is coded, for tracks and remixes alike.
For remixes, the EQ runs once, on the sum. We use the mixture settings of MERL's released code:

- a low shelf at 50 to 150 Hz;
- a high shelf at 6 to 12 kHz;
- 0 to 4 peaks between the two shelves, with Q drawn from 0.7 to 3;
- every gain drawn from −6 to +6 dB.

The paper says 1 to 4 peaks, and the code draws 0 to 4. We follow the code, since it produced the paper's results.
One change from MERL: our shelves are monotonic (Q 0.707), so they add no random bump above the codec's cut-off.

The target and the MP3 carry the same balance, so the EQ teaches the model to keep the input's balance rather than
pull it toward MUSDB's. The EQ barely changes the damage. On 18 clips, a ±6 dB high shelf moved LAME's cut-off by
at most 43 Hz, and the in-band SNR of the MP3 by at most 0.2 dB. The cost is at 24 to 48 kbps. There, the high shelf
moves the target's top band in a region the MP3 removed, and the model has to guess that level. In MERL's
random-mix models, mixture EQ alone moved average SDR from 8.2 to 8.3 dB at 470k steps, and from 8.4 to 8.5 dB at
1M steps. That is a small gain, in separation, from single runs. The EQ costs 7.3 ms per window, against 130 ms
for the codec at 64 kbps.

### Left and right are swapped half the time

With probability 0.5, the channels are swapped. A track window is swapped as a whole. A remix swaps each stem on its
own, before the sum, as Demucs does: the stems come from independent songs, so their images do not need to agree.
The codec mirrors exactly. On 26 of 26 windows at 32 and 128 kbps, the MP3 of a swapped window, swapped back,
was bit-identical to the MP3 of the original. We expect a gain near zero. The swap is cheap insurance for a long
run.

### Half of the MP3s get a second codec, as YouTube adds one

Every item is still coded as MP3 first, at one of the six bitrates. Half of the MP3s go through torchaudio's sox,
as today, and the other half through ffmpeg's libmp3lame. Then, with probability 0.5, the decoded MP3 goes through
a second round trip. That round trip is Opus (libopus) or AAC (ffmpeg's own encoder), picked with equal odds, at a
bitrate drawn from 64, 96, 128, 160 and 192 kbps. This is the README's rip: an uploader's MP3, re-encoded by
YouTube.

Opus is the codec whose damage differs most from LAME's. At 48 to 64 kbps, Opus keeps the power of the 16 to
20 kHz band within 0.4 to 1.5 dB of the clean band, but fills it with unrelated content, at a band SNR of −2.3 to
−4.0 dB. LAME CBR 128 leaves the same band empty, at −14.0 dB. A model trained on LAME alone has never seen a high
band that is present but wrong. This rests on two measured windows.

AAC's damage is the closest to MP3's: low-passed, with an almost empty band above. ffmpeg's own encoder is the only
AAC encoder on a stock Linux pod. YouTube's encoder is not public, so ours may not match it.

The two MP3 paths use the same LAME library, and at 128 kbps they gave the same damage: 23.28 dB SNR for both.
They differ in the decoder (sox uses libmad, ffmpeg its own) and in how the encoder delay is removed. So the split
adds little damage variety by itself. The variety comes from the second codec.

Every stage has to stay aligned, because a shift of one sample ruins waveform metrics (ADR-0007). ffmpeg's decoders
trim Opus's pre-skip and AAC's priming samples when the container records them, as Ogg and MP4 do. `coded_pair`'s
search of ±4 samples catches what is left. The tests send a click through each path and check where it lands.

The second codec costs CPU, not GPU time. One Opus round trip takes 152 to 194 ms per 10 s window, against 99 to
154 ms for the sox MP3, on an M3. Training is bound by the GPU, which leaves about 1,470 ms of CPU per item at
1.75 s per step.

Validation and the test do not change. Both keep LAME CBR through sox at the six bitrates, so their scores stay
comparable with earlier runs and with ADR-0015.

## Other proposals wait for a measurement

- **Vinyl surface noise and ticks, on both sides.** Not yet. We would add it to 20% of the clean windows before
  coding, so the target keeps the record's character. At the AN-2 rip's level, it barely changes the codec damage:
  SNR moves by +0.01 dB at 32 kbps and +0.05 dB at 128 kbps, because the noise sits 48 to 57 dB below the codec's
  own error. The first simulation went too far. At its top level, +30 dB over the rip's floor, it put noise 18.8 dB
  below the track's RMS into the target, mostly at 20 to 300 Hz. Our two real transfers sit at 0 and +13 dB over the
  floor. If added, the range is 0 to +13 dB, varied only above 300 Hz.
- **Vinyl noise on the MP3 only, so the model removes it.** This changes the task from restoring the codec to
  restoring the record, and the model may remove crackle that belongs to the music. It is also small: the noise
  floor of the rip lies 74.4 dB below the track's RMS. [Moliner and Välimäki 2022](https://arxiv.org/abs/2202.08702)
  is the reference recipe. If ever chosen, it would be an opt-in mode with its own prompt.
- **Recorded surface noise instead of synthetic noise.** The open dataset holds 78 rpm shellac at 2 to 20 dB SNR.
  Our LP transfers sit 61 to 74 dB below the track's RMS. We found no open set of LP run-out grooves.
- **A brickwall limiter on the clean window.** A look-ahead limiter, with a drive picked for each window, brings
  MUSDB's median PLR from 13.2 to 9.6 dB, within the house masters' 9.0 to 9.8 dB. It costs 11 ms per window. It
  waits on two things. First, a decision on whether loud digital masters are part of the target: the one vinyl rip,
  at 12.0 dB PLR, is close to MUSDB. Second, a check of whether the model is sensitive to level: render the loud
  windows as they are and at −6 dB. If loudness normalisation at inference fixes them, the limiter is not needed.
  Its first implementation had no working release, and it needs a textbook one.
- **tanh soft clipping, as a branch of the limiter.** There is no published gain. It comes only after the limiter.
- **LAME VBR.** Its damage is of the same kind as CBR's, and milder: V5 gives 27.30 dB SNR, against 23.28 dB for
  CBR 128, with the same 16.75 kHz cut-off. It is cheaper to code than CBR. It waits on held-out VBR test columns.
- **LAME with a moved low-pass (ffmpeg's `-cutoff`).** The model has only seen LAME's six edges. Real files have
  others: LAME 192 cuts at 18.74 kHz, Apple AAC 128 at 18.0 kHz, Opus at 20.5 kHz, and the AN-2 rip near 15.7 kHz.
  With `-cutoff`, the edge followed the request, and the in-band SNR stayed at 24.57 to 24.58 dB, against 24.57 dB
  with LAME's own low-pass. The only ablation on music, [Sulun and Davies
  2021](https://arxiv.org/abs/2011.07274), found a model trained on one low-pass filter scoring below its own input
  on an unseen filter at the same cut-off. The ffmpeg half of the MP3s could carry it, with the cut-off drawn
  between 0.75 E and min(E + 3 kHz, 20 kHz), where E is the bitrate's own edge.
- **Extra CBR bitrates (40, 56, 80 and 112 kbps).** They fill the gaps between edges with no code change. This is
  the fallback if the moved low-pass is not built.
- **Chains in the other order, with the MP3 last.** AAC 128 then LAME 192 would match the AN-2 rip, and Opus 160
  then LAME 128 another common path. They wait on held-out chain test columns.
- **Speed change by resampling (pitch and tempo together).** It is the only pitch or tempo change that keeps a
  clean master, if the resampler's filter is pinned. In Demucs its benefit was against overfitting on MUSDB alone,
  and our run already remixes. Add it if validation loss turns up.
- **A random gain on the whole window.** Codec damage does not depend on level: SNR moved +0.01 dB at −12 dB and
  +0.05 dB at −24 dB, on 26 windows at 32 kbps. So it helps only if the model is sensitive to level, which the same
  −6 dB render checks.
- **Remixes of stems from one song.** These would be the most plausible remixes, but track windows already give
  50% coherent targets. They also need activity masks per song.
- **One window length per batch (107, 65, or 32 to 106 frames).** The test clips are 6 s (65 frames), while training
  uses 107 frames. A render that costs about $1 tells length and noise apart. Render the test clips at 65 frames with
  the last 65 frames of a 107-frame noise tensor, and at 107 frames with the same tensor. Padding short windows to
  107 frames inside a batch does not work on our pods: without flash-attn, SA3's attention still attends to the
  padded frames.
- **Sequences of 256 to 1,938 frames in training.** A 180 s restore in one pass holds steady along the file: LSD
  stays within ±0.3 dB over 70 to 180 s, with no penalty at the end. That is one file.
- **A check of the training songs for hidden low-passes.** On the pod, flag any song with a drop of 15 dB or more
  between adjacent 250 Hz bands below 21 kHz. The 26 local MUSDB test tracks show none. Dark masters are real and
  stay.
- **Mono windows.** They help only on mono rips, and they shift training toward easier damage: +1.16 dB of SNR at
  32 kbps. Revisit when mono rips are part of the target.

## Some fixes belong at inference

- **A different noise seed per clip,** from a hash of the track and start, the same across checkpoints. With seed 0
  for every clip, one noise pattern put an audible hit at 5.6 to 5.9 s into most clips of the listening pages. The
  peak excess of that frame was +14.2 dB with seed 0, against +1.9 and +3.9 dB with seeds 1 and 2. The hit costs
  about 0.04 dB on the scores, so the gain is for listening. Per-clip LSD moves by up to 11.4 dB with the seed, so the
  full test is scored again after.
- **Loudness normalisation of the input before encoding,** if the −6 dB render shows the model is sensitive to
  level. It costs nothing in training.
- **A 20 Hz high-pass on the side channel.** In the 10 to 20 Hz band, the side channel of the 2003 AN-2 MP3 sits at
  −36.3 dBFS, 28 to 38 dB above that of the masters. A filter removes it exactly. Make it the default only if renders
  show rumble passing to the output.
- **A steep low-pass at 16.5 kHz on inputs whose edge lies above 16.74 kHz,** so an Opus input looks like the MP3s
  the model knows. It throws away real content at Opus 128 to 160 kbps. Score it on the Opus test columns first.
- **Overlapping windows with crossfades for long files.** The one-pass restore shows no loss along 180 s.
  Crossfading is safe only for a consistent model (rules/audio.md), and the LoRA's output depends on its noise.
  Test consistency first.
- **Test clips padded with zeros to 107 frames.** The expected gain is about 0 dB, because the end cost comes from
  the MP3's own last 25 ms.

## The rest is rejected

Mixes and stems:

- **EQ on each stem.** It helps separation by making stems easier to tell apart, and our target is the mix.
- **Two drum kits in one remix.** MERL's D-2 gained 0.1 dB in separation, but two kits at different tempi is not a
  plausible master.
- **Random silence on each stem (BS-RoFormer).** It does the stem count's job with weaker control, and it can
  draw zero stems.
- **A time offset for each stem within one song (Demucs's `shift_same: false`).** It puts flams and doubled hits
  into the target.
- **Cross-song remixes matched in tempo and key.** Stretching and pitching would put artefacts into the clean
  target.
- **Speed changes for each stem in a remix.** They rely on a separation mechanism, and cross-song stems already
  differ in key and tempo.
- **A swap or polarity flip for each stem of a mix rebuilt from one song.** Stems of one song share bleed and
  reverb, so the mix gets phase and image errors that no master has.
- **Changing the 50/50 weights of tracks and remixes.** Two runs at 0% and 50% remixes ended within 0.002 of
  validation loss of each other.

EQ:

- **audiomentations' seven-band EQ.** Its median span is 17.9 dB, against 9.0 dB for MERL's mixture EQ, and its
  gain reaches +14.1 dB at the 95th percentile.
- **FFT spectral weighting (FilterAugment).** It costs 30 ms against 7.3 ms. Its step type puts hard gain steps
  into the target, and it moves the band above every cut-off, where the MP3 carries no evidence.
- **EQ on the MP3 only.** It teaches the model to undo any EQ, which is remastering.
- **Band-limiting filters on the clean window.** They remove from the target the top band the model must learn to
  generate.
- **EQ only above the codec's cut-off (14 to 20 kHz).** It puts randomness where the MP3 holds nothing, at five of
  the six bitrates.
- **EQ normalisation at inference.** MERL measured a loss with it.

Time and pitch:

- **WSOLA time-stretch or pitch shift (sox, ffmpeg, soundstretch).** A tempo change of ×1.06 put −14.2 dB of a
  steady chord's energy into warble sidebands, against −93.6 dB in the original.
- **Phase-vocoder stretch or pitch (rubberband, librosa).** It changed the level by −0.3 to −3.7 dB and widened the
  image by +0.4 to +9.0 dB of side over mid.
- **Speed change with the resampler's default filter.** torchaudio's default took −1.5 to −4.6 dB off the target
  at 19 to 20 kHz, which teaches the model to roll off its own top octave.
- **Time reversal.** It gives rising decays, and the MP3's pre-echo lands after the attacks.

Level and stereo:

- **Polarity flip.** The codec cannot see it: the MP3 of −x matched −MP3(x) to 146 dB, and no listener hears it.
- **Normalising both windows to a common peak after coding (Apollo).** It breaks the match with validation and
  inference, whose windows keep their level.
- **Stereo width changes.** Widening brings side reverb and noise into the target beyond the engineer's choice, and
  moves the damage by only ±0.3 dB of SNR.

Dynamics and room:

- **A mix-bus compressor.** It moved PLR the wrong way: from 13.19 to 13.61 dB on MUSDB, and from 9.39 to 11.01 dB
  on aerofunk.
- **pedalboard's Limiter as the limiter.** It passed its ceiling by 0.31 dB, and it has no look-ahead.
- **Reverb on the whole mix.** It changed the codec damage by −0.14 to +0.11 dB of SNR, has no published gain, and
  adding a room to a finished mix is not a mastering step.
- **Compression, clipping or reverb on the input only, with the original as target (SonicMaster, the MSR
  recipe).** They teach de-compression and de-reverberation, which is remastering.
- **musdb-XL-train as training data.** Its 4 s segments do not fit our 10 s windows with codec pads, and it takes
  18.2 GB compressed.

Record and transfer:

- **Rumble in training.** A filter at inference removes it exactly.
- **Hum.** No hum line was found on the rip: every peak was 3.3 dB or less, against up to 13 dB for bass notes in
  the masters.
- **Wow and flutter.** We measured 0.077% at 0.750 Hz. On both sides that is a pitch wobble of 1.3 cents with no
  effect on the codec. On the MP3 only, it shifts the input against the target by up to ±7 samples.
- **Cartridge or RIAA mismatch, as an item of its own.** Two transfers of AN-2 differ by up to 3.1 dB above 4 kHz,
  which fits inside the EQ's ±6 dB.
- **Groove distortion and ADC noise.** They cannot be measured without an aligned clean reference.

Codecs:

- **A soft Butterworth low-pass on the MP3 side.** Half of the draws darkened the band below the edge by more than
  1 dB, so the model would learn to brighten soft roll-offs, a vinyl rip's own included. If the edge moves, it moves
  through LAME's own steep low-pass (deferred above).
- **HE-AAC.** A stock Linux pod has no encoder for it, and its 3,074-sample decoder delay nearly fills the
  4,096-sample pad.
- **Ogg Vorbis.** It is not a format the rips arrive in.
- **True low-rate MP3 (MPEG-2 at 22.05 kHz).** It differs little from what we train on: 15.72 against 16.95 dB of
  SNR, with a cut-off of 4.0 against 4.25 kHz. Two reports found independently that the "24 kbps" setting writes
  32 kbps MPEG-1 frames.
- **Variety in LAME's `-q` and stereo mode.** It changed SNR by at most 1.6 dB and did not move the cut-off.
- **MP3s above 128 kbps.** There the SAME-L autoencoder caps the output. At 128 kbps, the LoRA scores 9.23 dB SDR,
  against 9.82 dB for SAME-L's round trip of the clean master and 19.39 dB for the MP3, on 50 clips.
- **More weight on 24 and 32 kbps.** Real rips rarely sit there, and it would likely cost at 96 and 128 kbps.
- **fast-mp3-augment as a faster CBR coder.** It solves a speed problem the run does not have, and it swaps libmad
  for minimp3, so training no longer decodes as validation does.
- **A 44.1 → 48 → 44.1 kHz round trip.** It moved SNR by 0.05 dB or less, and LSD by 0.02 dB or less.

Model side:

- **Noise or blur on the MP3's latents.** Our conditioning is the exact encoding of the real input, and blurring it
  blurs evidence.
- **Conditioning on the input's cut-off (AudioLBM).** It needs a new conditioning path, and AudioLBM measured a
  small gain: music LSD went from 1.187 to 1.166.
- **Padding short items to 107 frames inside a batch.** Without flash-attn's variable-length path, SA3 zeroes the
  padded values but leaves their keys in the softmax.
- **Telling the model 9.94 s on a 6 s clip.** Paired against the clip's true length, the excess at the clip's end
  was 1.3 dB worse at 32 kbps and 0.5 dB worse at 64 kbps.
- **Training on MP3s coded alone, as the test codes its clips.** 95% of the end cost sits in the last 25 ms, where
  the MP3 holds no music. Apollo, trained on exactly this, removes only 16 to 29% of it. That leaves a gain of
  about 0.1 to 0.2 dB of spectral SNR, for a quirk of the test protocol that real rips do not have.

## What is left

The four accepted items have no measured effect on this task yet. Their evidence comes from separation papers and
from measurements of the damage. We will score the next run on three sets:

- the MUSDB test at the six bitrates, where scores must not drop;
- held-out codec columns (Opus 64 and 128, AAC 128, and MP3 then Opus), where the next run should gain;
- a few masters from outside MUSDB (two Nesta tracks, the AN-2 vinyl and aerofunk), by blinded listening.

Without a control run at the same number of steps, the effect of these four items cannot be told apart from the
effect of training longer.
