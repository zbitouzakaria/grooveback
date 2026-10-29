"""The training datasets. The prepared pairs are encoded by a tiny Stable Audio 3
built by SA3's own factory (tests/conftest.py), so the real model code runs on
CPU with no download; the remixes need no model. Stand-ins replace the codecs,
which need sox and ffmpeg, except in the alignment tests at the end."""

import itertools
import json
import subprocess

import numpy as np
import pytest
import torch

from grooveback import audio as ga
from grooveback import dataset as gd
from grooveback import training as gt
from grooveback.cli import musdb_eval

SR = 44_100
CPU = torch.device("cpu")
# Each MUSDB track of the remix tests is a sine with this many cycles per
# 2_048-sample window, so in any window its energy sits in that one FFT bin.
CYCLES = {"mixture": 7, "bass": 3, "drums": 5, "other": 11, "vocals": 13}
STEMS = ("bass", "drums", "other", "vocals")
# Every chain of one or two codec round trips, the sox and ffmpeg MP3s apart, each
# codec at its lowest bitrate in configs/train_lora.yaml: 4 chains of one, 16 of two.
LOWEST = {"sox": 24, "libmp3lame": 24, "libopus": 64, "aac": 64}
CHAINS = [
    pytest.param([(codec, LOWEST[codec]) for codec in chain], id="+".join(chain))
    for chain in [*([codec] for codec in LOWEST), *itertools.product(LOWEST, repeat=2)]
]


def mean_encode(model, audio):
    """Stands in for the tiny SA3's encoder, whose random weights map any audio to
    the same latents: the mean of every 64 samples, repeated to 8 channels."""
    padded = np.pad(audio, ((0, 0), (0, -audio.shape[-1] % 64)))
    return torch.from_numpy(padded.reshape(2, -1, 64).mean(-1)).repeat(4, 1)


def write_song(root, name, *, seconds, bitrates, seed):
    """One song in prepare_dataset's layout: white noise, and per bitrate a copy
    scaled by kbps / 256, so each bitrate's version differs from the others."""
    rng = np.random.default_rng(seed)
    original = rng.uniform(-0.5, 0.5, size=(2, int(seconds * SR))).astype(np.float32)
    ga.save(root / name / "original.wav", original, SR)
    for kbps in bitrates:
        ga.save(root / name / f"{kbps}k.wav", kbps / 256 * original, SR)


def write_musdb_song(root, name, *, seconds, amplitudes, silent_seconds, right_gain, sample_rate):
    """One song in MUSDB18-HQ's layout, 16-bit like the real files: every track
    a sine at its own `CYCLES` bin and `amplitudes[kind]` on the left, times
    `right_gain` on the right, after `silent_seconds` of digital silence. The
    mixture is not the stems' sum, so a window of it is told apart from any
    remix by its bin alone."""
    t = np.arange(int(seconds * sample_rate))
    for kind, cycles in CYCLES.items():
        tone = amplitudes[kind] * np.sin(2 * np.pi * cycles * t / 2048)
        tone[: int(silent_seconds * sample_rate)] = 0.0
        audio = np.stack([tone, right_gain * tone]).astype(np.float32)
        ga.save(root / name / f"{kind}.wav", audio, sample_rate, subtype="PCM_16")


def write_edge_jump_track(path, *, seconds, seed):
    """A 16-bit stereo track like the clip that broke Apollo's `match2`
    (tests/test_musdb_eval.py): noise below 2 kHz, scaled to peak at 0.3, under
    a ramp from -0.5 to +0.5."""
    samples = int(seconds * SR)
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal((2, samples)), axis=-1)
    spectrum[:, int(2_000 * seconds) :] = 0  # bin k sits at k / seconds Hz
    noise = np.fft.irfft(spectrum, n=samples, axis=-1)
    audio = 0.3 * noise / np.abs(noise).max() + np.linspace(-0.5, 0.5, samples)
    ga.save(path, audio.astype(np.float32), SR, subtype="PCM_16")


class HalvingCodec:
    """Stands in for `round_trip`, which needs sox and ffmpeg: halves the audio,
    and records the codec, bitrate and input peak of every call."""

    def __init__(self):
        self.codecs = []
        self.bitrates = []
        self.peaks = []

    def __call__(self, audio, *, codec, kbps):
        self.codecs.append(codec)
        self.bitrates.append(kbps)
        self.peaks.append(float(np.abs(audio).max()))
        return 0.5 * audio


def test_an_epoch_holds_each_window_once_whatever_the_bitrates(
    tmp_path, monkeypatch, tiny_checkpoint
):
    """1 s is 689 frames of 64 samples, so 21 windows of 32 at either bitrate."""
    write_song(tmp_path / "songs", "a", seconds=1.0, bitrates=[24, 128], seed=0)
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)

    dataset = gd.LatentPairDataset(tmp_path / "songs", model, bitrates=[24, 128], window_frames=32)

    assert len(dataset) == 21


def test_a_training_item_pairs_its_window_with_one_of_its_mp3s(
    tmp_path, monkeypatch, tiny_checkpoint
):
    write_song(tmp_path / "songs", "a", seconds=1.0, bitrates=[24, 128], seed=0)
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    monkeypatch.setattr(gt, "encode", mean_encode)
    model = gt.load_model("small-music-base", device=CPU)
    dataset = gd.LatentPairDataset(tmp_path / "songs", model, bitrates=[24, 128], window_frames=32)

    clean, mp3 = dataset[3]

    window = slice(3 * 32, 4 * 32)  # item 3 is the song's fourth window
    torch.testing.assert_close(clean, dataset.clean["a"][:, window], rtol=0, atol=0)
    low, high = dataset.mp3[("a", 24)][:, window], dataset.mp3[("a", 128)][:, window]
    assert torch.equal(mp3, low) or torch.equal(mp3, high)


def test_a_validation_item_uses_its_stored_bitrate(tmp_path, monkeypatch, tiny_checkpoint):
    write_song(tmp_path / "songs", "a", seconds=1.0, bitrates=[24, 128], seed=0)
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    monkeypatch.setattr(gt, "encode", mean_encode)
    model = gt.load_model("small-music-base", device=CPU)
    windows = [["a", 32 * index] for index in range(21)]
    stored = {"drawn_from": [24, 128], "windows": windows, "bitrates": [128] + [24] * 20}
    (tmp_path / "bitrates.json").write_text(json.dumps(stored))
    dataset = gd.LatentPairDataset(
        tmp_path / "songs",
        model,
        bitrates=[24, 128],
        window_frames=32,
        bitrates_file=tmp_path / "bitrates.json",
    )

    _, mp3 = dataset[0]

    torch.testing.assert_close(mp3, dataset.mp3[("a", 128)][:, 0:32], rtol=0, atol=0)


def test_fixed_bitrates_store_one_bitrate_per_window_on_first_use(tmp_path):
    windows = [("a", 0), ("a", 107), ("b", 0)]

    drawn = gd.fixed_bitrates(tmp_path / "bitrates.json", windows, bitrates=[24, 128])

    assert json.loads((tmp_path / "bitrates.json").read_text())["bitrates"] == drawn
    assert len(drawn) == 3
    assert set(drawn) <= {24, 128}


def test_fixed_bitrates_refuse_a_draw_stored_for_other_windows(tmp_path):
    """A new window length moves every window, so the stored draw no longer fits."""
    gd.fixed_bitrates(tmp_path / "bitrates.json", [("a", 0), ("a", 107)], bitrates=[24, 128])

    with pytest.raises(ValueError, match="other windows"):
        gd.fixed_bitrates(tmp_path / "bitrates.json", [("a", 0), ("a", 64)], bitrates=[24, 128])


def test_fixed_bitrates_refuse_a_draw_from_another_bitrate_list(tmp_path):
    """A draw from [24, 32] read by a six-bitrate run would quietly validate it at
    its two hardest bitrates only."""
    gd.fixed_bitrates(tmp_path / "bitrates.json", [("a", 0), ("a", 107)], bitrates=[24, 32])

    with pytest.raises(ValueError, match="drawn from"):
        gd.fixed_bitrates(
            tmp_path / "bitrates.json", [("a", 0), ("a", 107)], bitrates=[24, 32, 48, 64, 96, 128]
        )


def test_every_remix_item_pairs_its_clean_window_with_the_codec_output(tmp_path, monkeypatch):
    """Stems at 0.9 pass full scale in many remixes, and the EQs, swaps and left-out
    kinds change them, so the pairs include windows scaled down and augmented
    before coding: the clean side returned is the one coded."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.9, "drums": 0.9, "other": 0.9, "vocals": 0.9}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=0.5,
        sample_rate=SR,
    )
    augmentations = gd.Augmentations(drop_kind=0.5, stem_eq=0.5, mix_eq=0.5, swap=0.5)
    dataset = gd.RemixDataset(tmp_path, window_samples=2048, augmentations=augmentations)

    items = [dataset[index] for index in range(50)]

    assert {(clean.shape, clean.dtype) for clean, _ in items} == {((2, 2048), torch.float32)}
    torch.testing.assert_close(
        torch.stack([mp3 for _, mp3 in items]),
        0.5 * torch.stack([clean for clean, _ in items]),
        rtol=0,
        atol=0,
    )


def test_each_remix_item_draws_its_own_bitrate(tmp_path, monkeypatch):
    """50 items at 24 or 128 kbps: a fixed bitrate, or one per dataset, would
    show one value; a fair draw misses one with probability 2 * 0.5**50."""
    torch.manual_seed(0)
    codec = HalvingCodec()
    monkeypatch.setattr(gd, "round_trip", codec)
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    augmentations = gd.Augmentations(codec_bitrates={"mp3": [24, 128]})
    dataset = gd.RemixDataset(tmp_path, window_samples=2048, augmentations=augmentations)

    [dataset[index] for index in range(50)]

    assert set(codec.bitrates) == {24, 128}


def test_a_window_goes_through_as_many_round_trips_as_codec_rounds_draws(tmp_path, monkeypatch):
    """One round trip at weight 0.25, two at 0.75. The stand-in halves the audio on
    every round trip, so a pair at 0.5 went through one and a pair at 0.25 through
    two. Of 400 windows, 66 to 134 go through one: four standard deviations (8.7)
    around 100."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    augmentations = gd.Augmentations(
        codec_rounds={1: 0.25, 2: 0.75}, codec_choices=[["mp3"], ["mp3"]]
    )
    dataset = gd.TrackDataset(
        tmp_path, glob="*/mixture.wav", window_samples=2048, augmentations=augmentations
    )

    pairs = [dataset[index] for index in range(400)]

    ratios = [float(mp3.abs().max() / clean.abs().max()) for clean, mp3 in pairs]

    assert sorted(set(ratios)) == [0.25, 0.5]
    assert 66 <= ratios.count(0.5) <= 134


def test_each_round_trip_draws_from_its_own_codec_choices(tmp_path, monkeypatch):
    """An MP3 first, then Opus or AAC: 50 windows make 100 calls in pairs. A fair
    draw misses Opus or AAC in the second round trips with probability
    2 * 0.5**50."""
    torch.manual_seed(0)
    codec = HalvingCodec()
    monkeypatch.setattr(gd, "round_trip", codec)
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    augmentations = gd.Augmentations(
        codec_rounds={2: 1.0},
        codec_choices=[["mp3"], ["opus", "aac"]],
        codec_bitrates={"mp3": [24], "opus": [64], "aac": [64]},
    )
    dataset = gd.TrackDataset(
        tmp_path, glob="*/mixture.wav", window_samples=2048, augmentations=augmentations
    )

    [dataset[index] for index in range(50)]

    assert (set(codec.codecs[0::2]), set(codec.codecs[1::2])) == ({"sox"}, {"libopus", "aac"})


def test_each_codec_is_coded_at_one_of_its_own_bitrates(tmp_path, monkeypatch):
    """MP3 at 24 kbps only, Opus at 192 only: any other pairing would show."""
    torch.manual_seed(0)
    codec = HalvingCodec()
    monkeypatch.setattr(gd, "round_trip", codec)
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    augmentations = gd.Augmentations(
        codec_choices=[["mp3", "opus"]], codec_bitrates={"mp3": [24], "opus": [192]}
    )
    dataset = gd.TrackDataset(
        tmp_path, glob="*/mixture.wav", window_samples=2048, augmentations=augmentations
    )

    [dataset[index] for index in range(50)]

    assert set(zip(codec.codecs, codec.bitrates, strict=True)) == {("sox", 24), ("libopus", 192)}


def test_a_share_of_the_mp3s_goes_through_ffmpeg(tmp_path, monkeypatch):
    """mp3_ffmpeg at 0.25: of 400 MP3s, 66 to 134 go through ffmpeg's libmp3lame,
    four standard deviations (8.7) around 100, and the others through sox."""
    torch.manual_seed(0)
    codec = HalvingCodec()
    monkeypatch.setattr(gd, "round_trip", codec)
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    augmentations = gd.Augmentations(mp3_ffmpeg=0.25)
    dataset = gd.TrackDataset(
        tmp_path, glob="*/mixture.wav", window_samples=2048, augmentations=augmentations
    )

    [dataset[index] for index in range(400)]

    assert set(codec.codecs) == {"sox", "libmp3lame"}
    assert 66 <= codec.codecs.count("libmp3lame") <= 134


def test_a_remix_mp3_ends_with_the_window_not_the_codec_lead_in(tmp_path, monkeypatch):
    """A sox MP3 comes back 1_105 samples late and is cut to the clip's length, so
    once aligned its last 1_105 samples are decoder lead-in, not the clip's end
    (ADR-0015). The stand-in zeroes them. Coded with pads, the window keeps the
    codec's clip end in the pad that is cut off."""
    torch.manual_seed(0)

    def lead_in_codec(audio, *, codec, kbps):
        coded = audio.copy()
        coded[:, -1_105:] = 0.0
        return coded

    monkeypatch.setattr(gd, "round_trip", lead_in_codec)
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(tmp_path, window_samples=2048)

    clean, mp3 = dataset[0]

    torch.testing.assert_close(mp3, clean, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shift", [pytest.param(-1, id="one-sample-early"), pytest.param(1, id="one-sample-late")]
)
def test_an_mp3_a_sample_off_is_paired_with_the_clean_audio_it_holds(tmp_path, monkeypatch, shift):
    """The alignment can miss by one sample at 24 kbps: on one window of 10 s it
    rolled the MP3 by 1_106 samples, not LAME's 1_105, leaving it one sample
    early. A pair one sample apart would teach the model a delay. The stand-in
    alignment misses by `shift`. White noise, since a test tone would also line
    up with itself one period away."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    monkeypatch.setattr(musdb_eval, "align", lambda coded, *, clean: np.roll(coded, shift, axis=-1))
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2, 10 * SR)).astype(np.float32)
    ga.save(tmp_path / "a" / "mixture.wav", noise, SR, subtype="PCM_16")
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)

    clean, mp3 = dataset[0]

    torch.testing.assert_close(mp3, 0.5 * clean, rtol=0, atol=0)


def test_a_note_that_gets_louder_keeps_every_pair_aligned(tmp_path, monkeypatch):
    """The stand-in codec does not shift, so every pair must hold the same audio.
    A held 1 kHz note that steps up 10 dB: searched up to 4_096 samples either
    side with best_lag, whose correlation is not normalised, windows matched a
    louder period of the note about 4_000 samples away. The step sits in the
    middle of a 0.5 s track, so the louder half is within that reach of most
    windows."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    t = np.arange(SR // 2)
    tone = np.where(t < SR // 4, 0.1, 0.316) * np.sin(2 * np.pi * 1_000 * t / SR)
    audio = np.stack([tone, tone]).astype(np.float32)
    ga.save(tmp_path / "a" / "mixture.wav", audio, SR, subtype="PCM_16")
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)

    pairs = [dataset[index] for index in range(20)]

    torch.testing.assert_close(
        torch.stack([mp3 for _, mp3 in pairs]),
        0.5 * torch.stack([clean for clean, _ in pairs]),
        rtol=0,
        atol=0,
    )


def test_a_remix_holds_stems_and_never_the_mixture(tmp_path, monkeypatch):
    """Windows of the real mixtures are TrackDataset's. 1e-3 sits far above the
    16-bit noise in a bin (about 3e-7) and far below the sines (0.1 and 0.5)."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(tmp_path, window_samples=2048)

    spectra = [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) * 2 / 2048 for index in range(200)]

    assert max(s[CYCLES["mixture"]] for s in spectra) < 1e-3


def test_every_remix_holds_each_stem_kind_once_at_a_gain_of_0_25_to_1_25(tmp_path, monkeypatch):
    """Each stem is a sine at 0.1 in its own bin, so a remix's bin amplitudes are
    its stems' gains times 0.1: from 0.025 to 0.125 for every kind, in every
    remix. A kind left out would show 0, and two stems of one kind would sum past
    0.125 in some remixes. abs=1e-3 covers the 16-bit rounding."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(tmp_path, window_samples=2048)

    spectra = np.stack(
        [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) * 2 / 2048 for index in range(100)]
    )

    stems = spectra[:, [CYCLES[kind] for kind in STEMS]]
    assert 0.025 - 1e-3 < stems.min()
    assert stems.max() < 0.125 + 1e-3


def test_half_the_remixes_leave_out_one_random_kind(tmp_path, monkeypatch):
    """drop_kind at 0.5: of 400 remixes, 160 to 240 hold three kinds, four standard
    deviations (10) around 200, the others all four, and each kind is the one left
    out in some of them. A kind is there when its bin passes 0.02, below the
    quietest stem's 0.025."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(
        tmp_path, window_samples=2048, augmentations=gd.Augmentations(drop_kind=0.5)
    )

    spectra = np.stack(
        [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) * 2 / 2048 for index in range(400)]
    )

    present = spectra[:, [CYCLES[kind] for kind in STEMS]] > 0.02
    three = present.sum(axis=1) == 3
    assert set(present.sum(axis=1)) == {3, 4}
    assert 160 <= three.sum() <= 240
    assert {STEMS[kind] for kind in np.argmin(present[three], axis=1)} == set(STEMS)


def test_a_set_with_one_active_kind_never_leaves_it_out(tmp_path, monkeypatch):
    """Only the bass is ever active, so drop_kind at 1 has nothing it can leave
    out: dropped anyway, the remix would hold no stem at all."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.0, "other": 0.0, "vocals": 0.0}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(
        tmp_path, window_samples=2048, augmentations=gd.Augmentations(drop_kind=1.0)
    )

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(50)]

    assert min(rms) > 0.01


def test_a_track_window_swaps_left_and_right_as_a_whole(tmp_path, monkeypatch):
    """The mixture plays on the left only. Swapped half the time, a window plays on
    the left or on the right, never on both; a fair draw keeps one side in all 50
    with probability 2 * 0.5**50."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=0.0,
        sample_rate=SR,
    )
    dataset = gd.TrackDataset(
        tmp_path,
        glob="*/mixture.wav",
        window_samples=2048,
        augmentations=gd.Augmentations(swap=0.5),
    )

    sides = {tuple((dataset[index][0].abs().amax(dim=-1) > 0.1).tolist()) for index in range(50)}

    assert sides == {(True, False), (False, True)}


def test_a_remix_swaps_left_and_right_of_each_stem_on_its_own(tmp_path, monkeypatch):
    """Every stem plays on the left only. Swapped on its own half the time, each of
    a remix's four stems lands on either side, so all four share a side in 1 of 8
    remixes: of 50, 34 or more mix both sides, four standard deviations (2.3)
    below 43.75. Swapped as a whole, no remix would."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=0.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(
        tmp_path, window_samples=2048, augmentations=gd.Augmentations(swap=0.5)
    )

    spectra = np.stack(
        [np.abs(np.fft.rfft(dataset[index][0].numpy(), axis=-1)) * 2 / 2048 for index in range(50)]
    )

    on_left = spectra[:, 0, [CYCLES[kind] for kind in STEMS]] > 0.02
    both_sides = on_left.any(axis=1) & ~on_left.all(axis=1)
    assert both_sides.sum() >= 34


def test_the_mix_eq_moves_a_track_by_up_to_6_db_at_0_hz(tmp_path, monkeypatch):
    """At 0 Hz only the low shelf acts, since a peak and a high shelf pass it
    unchanged, so a steady level moves by the shelf's gain: uniform from -6 to +6
    dB on the mix (Jeon et al. 2026, Table 4). Of 100 windows, some move past +3
    dB and some past -3 dB, each with probability 1 - 0.75**100. abs=0.1 dB covers
    the slowest shelf the mix EQ draws, at 50 Hz with Q 3, still settling after
    the 4_096-sample pad."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    level = np.full((2, SR), 0.1, dtype=np.float32)
    ga.save(tmp_path / "a" / "mixture.wav", level, SR, subtype="PCM_16")
    dataset = gd.TrackDataset(
        tmp_path,
        glob="*/mixture.wav",
        window_samples=2048,
        augmentations=gd.Augmentations(mix_eq=1.0),
    )

    gains_db = np.array(
        [20 * np.log10(float(dataset[index][0].mean()) / 0.1) for index in range(100)]
    )

    assert -6.1 < gains_db.min() < -3.0
    assert 3.0 < gains_db.max() < 6.1


def test_a_remix_past_full_scale_is_scaled_to_it_not_clipped(tmp_path, monkeypatch):
    """Four stems at 0.9 pass full scale in most remixes, so some of 100 windows
    reach the codec scaled to a peak of exactly 1. Scaling keeps every sine in its
    bin; clipping would spread energy into harmonics, far above the 1e-6 of the
    power that 16-bit noise leaves outside the bins."""
    torch.manual_seed(0)
    codec = HalvingCodec()
    monkeypatch.setattr(gd, "round_trip", codec)
    amplitudes = {"mixture": 0.5, "bass": 0.9, "drums": 0.9, "other": 0.9, "vocals": 0.9}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(tmp_path, window_samples=2048)

    power = np.stack(
        [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) ** 2 for index in range(100)]
    )

    assert max(codec.peaks) == 1.0
    outside = np.delete(power, list(CYCLES.values()), axis=1).sum(axis=1)
    assert (outside / power.sum(axis=1)).max() < 1e-6


def test_a_track_window_below_full_scale_keeps_its_level(tmp_path, monkeypatch):
    """Where Apollo would scale every window to peak at 1. The 0.1 sine reads back
    as 3_276 / 32_768 from 16 bits; abs=1e-3 covers that and the sampled peak."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.1, "bass": 0.05, "drums": 0.05, "other": 0.05, "vocals": 0.05}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)

    peaks = [float(dataset[index][0].abs().max()) for index in range(20)]

    assert peaks == pytest.approx([0.1] * 20, abs=1e-3)


def test_remix_windows_come_from_the_active_stretch_of_each_track(tmp_path, monkeypatch):
    """Every stem is silent for its first 0.5 s, so without the activity check all
    four stems of a remix would be silence about one time in 16. The quietest
    remix the sines can make holds four stems at a gain of 0.25: an RMS near
    0.035."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.5,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(tmp_path, window_samples=2048)

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(100)]

    assert min(rms) > 0.01


def test_a_stem_kind_silent_throughout_is_left_out_of_remixes(tmp_path, monkeypatch):
    """An instrumental's vocals keep no active window. Drawn anyway, a vocal stem
    would have no window to come from, and every remix would fail."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.0}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.RemixDataset(tmp_path, window_samples=2048)

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(200)]

    assert min(rms) > 0.01


def test_active_starts_keep_windows_whose_frames_are_mostly_loud():
    """10 frames of 441 samples, the first 6 at a steady 0.5 and the rest silent;
    a window is 4 frames. The windows starting at frames 0 to 6 hold 4, 4, 4, 3,
    2, 1 and 0 loud frames, so those at frames 0 to 3 keep more than half. The
    steady frames all equal the percentile, which counts as loud."""
    track = np.zeros((2, 4_410), dtype=np.int16)
    track[:, : 6 * 441] = 16_384  # 0.5 of full scale

    starts = gd.active_starts(track, window_samples=4 * 441)

    assert starts.tolist() == [0, 441, 882, 1_323]


def test_augmentations_refuse_more_round_trips_than_codec_choices():
    """Two round trips with choices for one would quietly code every window once."""
    with pytest.raises(ValueError, match="round trips"):
        gd.Augmentations(codec_rounds={1: 0.5, 2: 0.5}, codec_choices=[["mp3"]])


def test_augmentations_refuse_an_unknown_codec():
    with pytest.raises(ValueError, match="Unknown codec"):
        gd.Augmentations(codec_choices=[["flac"]], codec_bitrates={"flac": [128]})


def test_augmentations_refuse_a_codec_without_bitrates():
    with pytest.raises(ValueError, match="no bitrates for opus"):
        gd.Augmentations(codec_choices=[["opus"]])


def test_remix_dataset_refuses_a_track_shorter_than_one_window(tmp_path):
    """1_764 samples, short of a 2_048-sample window, before its codec pads."""
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=0.04,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=SR,
    )

    with pytest.raises(ValueError, match="shorter than one window"):
        gd.RemixDataset(tmp_path, window_samples=2048)


def test_remix_dataset_refuses_audio_the_codec_cannot_take(tmp_path):
    """The codecs run at 44.1 kHz; MUSDB18-HQ's files are all 44.1 kHz."""
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.0,
        right_gain=1.0,
        sample_rate=48_000,
    )

    with pytest.raises(ValueError, match="48000 Hz"):
        gd.RemixDataset(tmp_path, window_samples=2048)


def test_track_windows_start_at_a_fresh_position_every_draw(tmp_path, monkeypatch):
    """10 s of white noise has about 980 starts that pass the activity check and
    leave room for the pads, so each window's first samples tell its start. 20
    fair draws repeat one start with probability about 0.18, and two with about
    0.02; the same window every time would give one."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2, 10 * SR)).astype(np.float32)
    ga.save(tmp_path / "a" / "mixture.wav", noise, SR, subtype="PCM_16")
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)

    starts = {tuple(dataset[index][0][0, :8].tolist()) for index in range(20)}

    assert len(starts) >= 18


def test_track_windows_come_from_the_active_stretch(tmp_path, monkeypatch):
    """The mixture is silent for its first 0.5 s, so without the activity check
    about half its windows would be silence. A window with more than half its
    frames on the 0.5 sine keeps an RMS above 0.25."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path,
        "a",
        seconds=1.0,
        amplitudes=amplitudes,
        silent_seconds=0.5,
        right_gain=1.0,
        sample_rate=SR,
    )
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(100)]

    assert min(rms) > 0.1


def test_track_dataset_refuses_a_glob_that_matches_no_file(tmp_path):
    """A flat folder of songs read with MUSDB18-HQ's default glob, say."""
    ga.save(tmp_path / "song.wav", np.zeros((2, SR), dtype=np.float32), SR)

    with pytest.raises(ValueError, match="none of the 0 files"):
        gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)


def test_a_float_wav_track_keeps_its_level(tmp_path, monkeypatch):
    """Read as int16, a float WAV's samples in (-1, 1) would all round to 0, and the
    track would be dropped as silent. abs=1e-4 covers the 16-bit rounding of the
    0.1 sine and its sampled peak."""
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "round_trip", HalvingCodec())
    tone = 0.1 * np.sin(2 * np.pi * 7 * np.arange(SR) / 2048)
    ga.save(tmp_path / "a" / "mixture.wav", np.stack([tone, tone]).astype(np.float32), SR)
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)

    peaks = [float(dataset[index][0].abs().max()) for index in range(20)]

    assert peaks == pytest.approx([0.1] * 20, abs=1e-4)


def test_track_dataset_refuses_a_mono_file(tmp_path):
    """Batched with stereo windows, a mono one would crash the DataLoader mid-epoch."""
    ga.save(tmp_path / "a" / "mixture.wav", np.zeros((1, 10 * SR), dtype=np.float32), SR)

    with pytest.raises(ValueError, match="not stereo"):
        gd.TrackDataset(tmp_path, glob="*/mixture.wav", window_samples=2048)


def skip_unless_codecs_run():
    """Skips the calling test unless sox's MP3 and ffmpeg's three encoders all run here."""
    try:
        for codec in LOWEST:
            gd.round_trip(np.zeros((2, 4_096), dtype=np.float32), codec=codec, kbps=64)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"a codec is unavailable: {error}")


@pytest.mark.parametrize("chain", CHAINS)
def test_a_track_pair_lines_up_whichever_codecs_coded_it(tmp_path, monkeypatch, chain):
    """One sox MP3 comes back 1_105 samples late and two 2_210, while ffmpeg's come
    back aligned; the pair must line up after any chain. The track is the clip
    that broke Apollo's `match2` (tests/test_musdb_eval.py): noise below 2 kHz
    leaves the band above the cut-off to the jump between the window's edges, and
    the ramp makes that jump loud. Every window gets the mix EQ, which moves the
    phase of what is coded, and half are swapped. The lowest bitrates are the
    ones where the alignment missed by a sample."""
    skip_unless_codecs_run()
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "draw_codecs", lambda augmentations: chain)
    write_edge_jump_track(tmp_path / "a" / "mixture.wav", seconds=6.0, seed=0)
    dataset = gd.TrackDataset(
        tmp_path,
        glob="*/mixture.wav",
        window_samples=4 * SR,
        augmentations=gd.Augmentations(mix_eq=1.0, swap=0.5),
    )

    clean, coded = dataset[0]

    # Opus takes out the ramp, which is nearly DC, so the pair is correlated on its first
    # differences, where the ramp is a constant. Aligned, the circular cross-correlation
    # of the channel sums peaks at index 0; one sox MP3 late, at 1_105.
    clean, coded = np.diff(clean.numpy().sum(axis=0)), np.diff(coded.numpy().sum(axis=0))
    xcorr = np.fft.irfft(np.fft.rfft(coded) * np.fft.rfft(clean).conj(), n=clean.shape[-1])
    assert int(np.argmax(xcorr)) == 0


@pytest.mark.parametrize("chain", CHAINS)
def test_a_remix_pair_lines_up_whichever_codecs_coded_it(tmp_path, monkeypatch, chain):
    """As for a track, with every stem the clip that broke Apollo's `match2`, each
    from its own noise. Every stem and every mix gets its own EQ, and half the
    stems are swapped and half the remixes leave out a kind."""
    skip_unless_codecs_run()
    torch.manual_seed(0)
    monkeypatch.setattr(gd, "draw_codecs", lambda augmentations: chain)
    write_edge_jump_track(tmp_path / "a" / "bass.wav", seconds=6.0, seed=0)
    write_edge_jump_track(tmp_path / "a" / "drums.wav", seconds=6.0, seed=1)
    write_edge_jump_track(tmp_path / "a" / "other.wav", seconds=6.0, seed=2)
    write_edge_jump_track(tmp_path / "a" / "vocals.wav", seconds=6.0, seed=3)
    augmentations = gd.Augmentations(drop_kind=0.5, stem_eq=1.0, mix_eq=1.0, swap=0.5)
    dataset = gd.RemixDataset(tmp_path, window_samples=4 * SR, augmentations=augmentations)

    clean, coded = dataset[0]

    # As for a track: first differences, then the circular cross-correlation's peak.
    clean, coded = np.diff(clean.numpy().sum(axis=0)), np.diff(coded.numpy().sum(axis=0))
    xcorr = np.fft.irfft(np.fft.rfft(coded) * np.fft.rfft(clean).conj(), n=clean.shape[-1])
    assert int(np.argmax(xcorr)) == 0
