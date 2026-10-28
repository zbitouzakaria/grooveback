"""The training datasets. The prepared pairs are encoded by a tiny Stable Audio 3
built by SA3's own factory (tests/conftest.py), so the real model code runs on
CPU with no download; the remixes need no model. Stand-ins replace the codec,
which needs sox."""

import json

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


def write_musdb_song(root, name, *, seconds, amplitudes, silent_seconds, sample_rate):
    """One song in MUSDB18-HQ's layout, 16-bit like the real files: every track
    a sine at its own `CYCLES` bin and `amplitudes[kind]`, after `silent_seconds`
    of digital silence. The mixture is not the stems' sum, so a window of it is
    told apart from any remix by its bin alone."""
    t = np.arange(int(seconds * sample_rate))
    for kind, cycles in CYCLES.items():
        tone = amplitudes[kind] * np.sin(2 * np.pi * cycles * t / 2048)
        tone[: int(silent_seconds * sample_rate)] = 0.0
        audio = np.stack([tone, tone]).astype(np.float32)
        ga.save(root / name / f"{kind}.wav", audio, sample_rate, subtype="PCM_16")


class HalvingCodec:
    """Stands in for codec_simu, which needs sox: halves the clip, and records
    every bitrate and input peak it is called with."""

    def __init__(self):
        self.bitrates = []
        self.peaks = []

    def __call__(self, clip, kbps):
        self.bitrates.append(kbps)
        self.peaks.append(float(np.abs(clip).max()))
        return 0.5 * clip


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
    """Stems at 0.9 pass full scale in many remixes, so the pairs include windows
    scaled down before coding: the clean side returned is the one coded."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.9, "drums": 0.9, "other": 0.9, "vocals": 0.9}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)

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
    monkeypatch.setattr(musdb_eval, "codec_simu", codec)
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.RemixDataset(tmp_path, bitrates=[24, 128], window_samples=2048)

    [dataset[index] for index in range(50)]

    assert set(codec.bitrates) == {24, 128}


def test_a_remix_mp3_ends_with_the_window_not_the_codec_lead_in(tmp_path, monkeypatch):
    """codec_simu cuts sox's output to the clip's length before removing its
    1_105-sample delay, so its last 1_105 samples are decoder lead-in, not the
    clip's end (ADR-0015). The stand-in zeroes them. Coded with pads, the window
    keeps the codec's clip end in the pad that is cut off."""
    torch.manual_seed(0)

    def lead_in_codec(clip, kbps):
        coded = clip.copy()
        coded[:, -1_105:] = 0.0
        return coded

    monkeypatch.setattr(musdb_eval, "codec_simu", lead_in_codec)
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)

    clean, mp3 = dataset[0]

    torch.testing.assert_close(mp3, clean, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shift", [pytest.param(-1, id="one-sample-early"), pytest.param(1, id="one-sample-late")]
)
def test_an_mp3_a_sample_off_is_paired_with_the_clean_audio_it_holds(tmp_path, monkeypatch, shift):
    """codec_simu's alignment can miss by one sample: on 48 windows of 10 s from
    four tracks at 24 kbps, it once rolled the MP3 by 1_106 samples, not LAME's
    1_105, leaving it one sample early. A pair one sample apart would teach the
    model a delay. White noise, since a test tone would also line up with itself
    one period away."""
    torch.manual_seed(0)
    monkeypatch.setattr(
        musdb_eval, "codec_simu", lambda clip, kbps: np.roll(0.5 * clip, shift, axis=-1)
    )
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2, 10 * SR)).astype(np.float32)
    ga.save(tmp_path / "a" / "mixture.wav", noise, SR, subtype="PCM_16")
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048)

    clean, mp3 = dataset[0]

    torch.testing.assert_close(mp3, 0.5 * clean, rtol=0, atol=0)


def test_a_remix_holds_stems_and_never_the_mixture(tmp_path, monkeypatch):
    """Windows of the real mixtures are TrackDataset's. 1e-3 sits far above the
    16-bit noise in a bin (about 3e-7) and far below the sines (0.1 and 0.5)."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)

    spectra = [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) * 2 / 2048 for index in range(200)]

    assert max(s[CYCLES["mixture"]] for s in spectra) < 1e-3


def test_interleaved_items_come_from_each_source_at_its_weight(tmp_path, monkeypatch):
    """Tracks at weight 0.25, remixes at 0.75: of 400 draws, 66 to 134 are mixture
    windows, four standard deviations (8.7 draws each) around 100. Equal odds
    would give about 200."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.InterleavedDataset(
        [
            gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048),
            gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048),
        ],
        probabilities=[0.25, 0.75],
        windows=400,
    )

    spectra = [
        np.abs(np.fft.rfft(dataset[index][0][0].numpy())) * 2 / 2048
        for index in range(len(dataset))
    ]

    assert 66 <= sum(s[CYCLES["mixture"]] > 0.25 for s in spectra) <= 134


def test_a_remix_past_full_scale_is_scaled_to_it_not_clipped(tmp_path, monkeypatch):
    """Stems at 0.9 pass full scale as soon as one is drawn louder than 0 dB, so
    some of 100 windows reach the codec scaled to a peak of exactly 1. Scaling
    keeps every sine in its bin; clipping would spread energy into harmonics,
    far above the 1e-6 of the power that 16-bit noise leaves outside the bins."""
    torch.manual_seed(0)
    codec = HalvingCodec()
    monkeypatch.setattr(musdb_eval, "codec_simu", codec)
    amplitudes = {"mixture": 0.5, "bass": 0.9, "drums": 0.9, "other": 0.9, "vocals": 0.9}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)

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
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.1, "bass": 0.05, "drums": 0.05, "other": 0.05, "vocals": 0.05}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048)

    peaks = [float(dataset[index][0].abs().max()) for index in range(20)]

    assert peaks == pytest.approx([0.1] * 20, abs=1e-3)


def test_remix_windows_come_from_the_active_stretch_of_each_track(tmp_path, monkeypatch):
    """Every stem is silent for its first 0.5 s, so without the activity check a
    remix of one stem would be silence about half the time. The quietest remix
    the sines can make is one stem at -10 dB: an RMS near 0.022."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.5, sample_rate=SR
    )
    dataset = gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(100)]

    assert min(rms) > 0.01


def test_a_stem_silent_throughout_is_never_drawn(tmp_path, monkeypatch):
    """An instrumental's vocals keep no active window. Drawn anyway, a remix of
    vocals alone would be silence: a quarter of the remixes hold each of 1 to 4
    stems, all vocals with probability 1/4 to 1/256, so about 8% of remixes,
    17 of these 200."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.0}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)

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


def test_remix_dataset_refuses_a_track_shorter_than_one_window(tmp_path):
    """1_764 samples, short of a 2_048-sample window, before its codec pads."""
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=0.04, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )

    with pytest.raises(ValueError, match="shorter than one window"):
        gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)


def test_remix_dataset_refuses_audio_the_codec_cannot_take(tmp_path):
    """codec_simu runs at 44.1 kHz; MUSDB18-HQ's files are all 44.1 kHz."""
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=48_000
    )

    with pytest.raises(ValueError, match="48000 Hz"):
        gd.RemixDataset(tmp_path, bitrates=[24], window_samples=2048)


def test_track_windows_start_at_a_fresh_position_every_draw(tmp_path, monkeypatch):
    """10 s of white noise has about 980 starts that pass the activity check and
    leave room for the pads, so each window's first samples tell its start. 20
    fair draws repeat one start with probability about 0.18, and two with about
    0.02; the same window every time would give one."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2, 10 * SR)).astype(np.float32)
    ga.save(tmp_path / "a" / "mixture.wav", noise, SR, subtype="PCM_16")
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048)

    starts = {tuple(dataset[index][0][0, :8].tolist()) for index in range(20)}

    assert len(starts) >= 18


def test_track_windows_come_from_the_active_stretch(tmp_path, monkeypatch):
    """The mixture is silent for its first 0.5 s, so without the activity check
    about half its windows would be silence. A window with more than half its
    frames on the 0.5 sine keeps an RMS above 0.25."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.5, sample_rate=SR
    )
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048)

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(100)]

    assert min(rms) > 0.1


def test_track_dataset_refuses_a_glob_that_matches_no_file(tmp_path):
    """A flat folder of songs read with MUSDB18-HQ's default glob, say."""
    ga.save(tmp_path / "song.wav", np.zeros((2, SR), dtype=np.float32), SR)

    with pytest.raises(ValueError, match="none of the 0 files"):
        gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048)


def test_a_float_wav_track_keeps_its_level(tmp_path, monkeypatch):
    """Read as int16, a float WAV's samples in (-1, 1) would all round to 0, and the
    track would be dropped as silent. abs=1e-4 covers the 16-bit rounding of the
    0.1 sine and its sampled peak."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    tone = 0.1 * np.sin(2 * np.pi * 7 * np.arange(SR) / 2048)
    ga.save(tmp_path / "a" / "mixture.wav", np.stack([tone, tone]).astype(np.float32), SR)
    dataset = gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048)

    peaks = [float(dataset[index][0].abs().max()) for index in range(20)]

    assert peaks == pytest.approx([0.1] * 20, abs=1e-4)


def test_track_dataset_refuses_a_mono_file(tmp_path):
    """Batched with stereo windows, a mono one would crash the DataLoader mid-epoch."""
    ga.save(tmp_path / "a" / "mixture.wav", np.zeros((1, 10 * SR), dtype=np.float32), SR)

    with pytest.raises(ValueError, match="not stereo"):
        gd.TrackDataset(tmp_path, glob="*/mixture.wav", bitrates=[24], window_samples=2048)
