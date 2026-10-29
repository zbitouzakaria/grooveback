# 17. Augment the LoRA's training data

Date: 2026-10-29

## Status

Proposed.

## Context

This is the data augmentation strategy for the next long run of the LoRA fine-tune. The LoRA learns to restore MP3s
from pairs of a clean window and its coded copy (README, "LoRA fine-tune"). Half of the windows come from whole
tracks and half are remixes of MUSDB18-HQ stems. Today every coded copy is one LAME CBR MP3, made by torchaudio's
sox backend at one of six bitrates.

Every augmentation below runs on the clean window before coding. The target and its coded copy change together, so
the model still only learns to undo the codec (ADR-0004).

## Remixes hold one stem of each kind

A remix takes one stem of each kind, each from its own random song, as Demucs does. Half of the remixes hold all
four kinds. The other half hold three, and the missing kind is picked at random. Each stem is scaled by a random
factor of 0.25 to 1.25, Demucs's `scale` setting
([config](https://github.com/facebookresearch/demucs/blob/main/conf/config.yaml)).

## A random EQ on each stem and on the mix

We use the EQ augmentation of [Jeon et al. 2026](https://www.merl.com/publications/TR2026-012) (Algorithm 1,
Tables 3 and 4), with the probabilities of their best run. Each EQ is a low shelf at 50 to 150 Hz, a high shelf at 6
to 12 kHz, and 0 to 4 peaks between them. The shelves and the peaks draw their Q the same way.

- With probability 0.5, every stem of a remix gets its own EQ, with Q from 0.7 to 5 and gains from −9 to +9 dB.
- With probability 0.5, the whole window gets one EQ, with Q from 0.7 to 3 and gains from −6 to +6 dB. This applies
  to tracks and remixes alike.

Tracks are finished mixes with no stems, so they only get the EQ on the mix. In Jeon et al.'s remix-only models,
these two probabilities raised average separation SDR from 8.4 to 8.6 dB at 1M steps (Table 6).

## Left and right are swapped half the time

With probability 0.5, left and right are swapped. A track is swapped as a whole. A remix swaps each stem on its own
before the sum, as Demucs's `flip` does.

## Each window goes through one or two codecs

Each window is coded by MP3, Opus or AAC, then sometimes coded again. Two codecs in a row are what a real file often
went through: an uploader's MP3, then YouTube's Opus or AAC. The same codec may come twice, since a re-uploaded video
is AAC over AAC.

- MP3 is LAME CBR at 24, 32, 48, 64, 96 or 128 kbps, the bitrates of the test (ADR-0015). Half of the MP3s go
  through torchaudio's sox, the other half through ffmpeg's libmp3lame.
- Opus (libopus) and AAC (ffmpeg's own encoder) are at 64, 96, 128, 160 or 192 kbps.

Two settings control the chain. `codec_rounds` maps a number of codec round trips to how often it is drawn, as
URGENT's simulator does for its degradations. `codec_choices` lists, for each round trip, the codecs it draws from
with equal odds. The default codes 40% of the windows once and 60% twice, with any codec at any round:

```yaml
codec_rounds: {1: 0.4, 2: 0.6}
codec_choices: [[mp3, opus, aac], [mp3, opus, aac]]
codec_bitrates:
  mp3: [24, 32, 48, 64, 96, 128]
  opus: [64, 96, 128, 160, 192]
  aac: [64, 96, 128, 160, 192]
mp3_ffmpeg: 0.5
```

"Half of the windows MP3 alone, half MP3 then Opus or AAC" is then `codec_rounds: {1: 0.5, 2: 0.5}` with
`codec_choices: [[mp3], [opus, aac]]`.

The coded copy has to come back aligned, because a shift of one sample ruins the waveform metrics (ADR-0007). Today
the alignment lives inside `codec_simu`: after the sox round trip, it rolls the MP3 by the lag that a phase-transform
cross-correlation finds against the clean window (ADR-0015). We move that step out of `codec_simu` and run it once,
after the last round trip, against the clean window, so the coded copy is aligned whatever chain it went through.
`coded_pair`'s search of ±4 samples then runs as today. A sox MP3 comes out 1,105 samples late before that step, and
the 4,096-sample pads hold the lag of two round trips. `codec_simu` still aligns its own output, so the test does not
change.

The test of that alignment moves to the data loader. Today it codes one clip with `codec_simu` and checks that the
cross-correlation of the clean clip and the MP3 peaks at lag 0. Its clip is the one that broke Apollo's `match2`:
noise below 2 kHz under a loud ramp. The new test asks the same of the pairs that `TrackDataset` and `RemixDataset`
return, with the real codecs. It runs once for every chain of one or two codecs, with the sox and ffmpeg MP3s
counted apart, at each codec's lowest bitrate. So it passes only if the pairs are aligned whichever codecs ran.
`codec_simu`'s own test stays, for the test protocol.

Validation and the test do not change: one sox MP3 at the six bitrates. Their scores stay comparable with earlier
runs and with ADR-0015.

## What is left

We will score the next run on the MUSDB test at the six bitrates, where scores must not drop, and on held-out Opus
and AAC columns, where it should gain. Without a control run at the same number of steps, the effect of these
augmentations cannot be told apart from the effect of training longer.
