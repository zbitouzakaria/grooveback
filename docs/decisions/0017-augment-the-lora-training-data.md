# 17. Augment the LoRA's training data

Date: 2026-10-29

## Status

Accepted.

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

The weights of `codec_rounds` must sum to 1, and a config whose weights do not is refused. This matters on the
command line, where Hydra merges a dict with the config's instead of replacing it:
`augmentations.codec_rounds={1:1.0}` becomes `{1: 1.0, 2: 0.6}`, which would still code 37.5% of the windows twice.
Give every count instead: `augmentations.codec_rounds={1:1.0,2:0}`.

The coded copy has to come back aligned, because a shift of one sample ruins the waveform metrics (ADR-0007). Today
the alignment lives inside `codec_simu`: after the sox round trip, it rolls the MP3 by the lag that a phase-transform
cross-correlation finds against the clean window (ADR-0015). We move that step out of `codec_simu` and run it once,
after the last round trip, against the clean window, so the coded copy is aligned whatever chain it went through.
`coded_pair`'s search of ±4 samples then runs as today. A sox MP3 comes out 1,105 samples late before that step, and
the 4,096-sample pads hold the lag of two round trips. `codec_simu` still aligns its own output, so the test does not
change.

Validation and the test do not change: one sox MP3 at the six bitrates. Their scores stay comparable with earlier
runs and with ADR-0015.

## How we test it

Each test fails under a deliberate bug that breaks what its name says, and passes on the real code. Most replace the
codecs with a stand-in; the ones on the real codecs need sox and ffmpeg.

Codecs:

- `test_each_codec_gives_a_lower_snr_at_its_lowest_bitrate_than_at_its_highest`
- `test_the_opus_round_trip_removes_a_dc_offset_as_libopus_does`
- `test_a_sox_round_trip_comes_back_1_105_samples_late`
- `test_an_ffmpeg_round_trip_comes_back_aligned`

Chains and alignment:

- `test_opus_after_an_mp3_at_24_kbps_codes_the_mp3_not_the_clean_window`
- `test_a_track_pair_lines_up_whichever_codecs_coded_it`, for each of the 20 chains of one or two codecs
- `test_a_remix_pair_lines_up_whichever_codecs_coded_it`, for each of the 20 chains of one or two codecs
- `test_the_delay_of_two_sox_mp3s_stays_inside_the_codec_pads`
- `test_an_mp3_a_sample_off_is_paired_with_the_clean_audio_it_holds`

Codec draws:

- `test_codec_rounds_of_one_at_0_25_codes_about_a_quarter_of_the_windows_once`
- `test_the_second_round_trip_draws_from_the_second_list_of_codec_choices`
- `test_a_window_coded_once_draws_its_codec_from_the_first_list_of_codec_choices`
- `test_the_two_round_trips_of_a_window_give_all_four_pairs_of_mp3_and_opus`
- `test_a_round_trip_picks_each_of_its_codecs_with_equal_odds`
- `test_each_codec_is_coded_at_one_of_its_own_bitrates`
- `test_each_remix_item_draws_its_own_bitrate`
- `test_mp3_ffmpeg_at_0_25_codes_about_a_quarter_of_the_mp3s_with_ffmpeg`

Configs that are refused:

- `test_augmentations_refuse_more_round_trips_than_codec_choices`
- `test_augmentations_refuse_an_unknown_codec`
- `test_augmentations_refuse_a_codec_without_bitrates`
- `test_augmentations_refuse_codec_rounds_whose_weights_do_not_sum_to_1`
- `test_refuses_codec_rounds_1_at_1_0_given_on_the_command_line_before_loading_the_model`

Remixes:

- `test_every_remix_holds_one_stem_of_each_kind`
- `test_remix_stem_gains_spread_from_0_25_to_1_25`
- `test_drop_kind_at_0_5_leaves_one_random_kind_out_of_about_half_the_remixes`
- `test_drop_kind_never_leaves_out_the_only_stem_kind_with_audio`
- `test_remixes_still_draw_from_the_other_kinds_when_one_kind_is_silent_throughout`
- `test_tracks_at_weight_0_25_make_about_a_quarter_of_the_training_windows`

Left and right swap:

- `test_a_swapped_track_window_plays_entirely_on_the_other_side`
- `test_swap_at_0_25_swaps_left_and_right_in_about_a_quarter_of_the_track_windows`
- `test_a_remix_swaps_left_and_right_of_each_stem_on_its_own`

EQ:

- `test_the_mix_eq_moves_a_track_by_up_to_6_db_at_0_hz`
- `test_mix_eq_at_0_25_equalizes_about_a_quarter_of_the_track_windows`
- `test_the_stem_eq_boosts_a_stem_by_up_to_9_db_at_0_hz`
- `test_the_mix_eq_moves_a_remix_by_up_to_6_db_even_with_stem_eq_at_0`

Augmentations before coding, and the trainer's config:

- `test_every_remix_item_pairs_its_clean_window_with_the_codec_output`
- `test_a_config_naming_only_opus_codes_every_track_window_with_opus`
- `test_a_config_naming_only_opus_codes_every_remix_with_opus`

Not tested: the EQ's shelf frequencies and number of peaks, whether the stem EQ goes on all of a remix's stems or
none, and the stem EQ's cuts.
