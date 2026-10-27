"""The pieces of the LoRA fine-tune, run on a tiny Stable Audio 3 built by
SA3's own factory (tests/conftest.py), so the real model code runs on CPU
with no download."""

import json

import numpy as np
import pytest
import torch

from grooveback import audio as ga
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


def test_new_mp3_input_leaves_the_base_model_output_unchanged_before_training(
    monkeypatch, tiny_checkpoint
):
    """The new input's last layers start at zero, so until it trains the model
    is exactly base SA3, whatever MP3 it is given."""
    from stable_audio_3.factory import create_diffusion_cond_from_config

    config, weights = tiny_checkpoint
    base = create_diffusion_cond_from_config(config)
    base.load_state_dict(weights)
    base.eval()
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)
    noised = torch.randn(2, 8, 40, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([0.3, 0.7])
    mp3 = torch.randn(2, 8, 40, generator=torch.Generator().manual_seed(2))
    cond = base.conditioner([{"seconds_total": 40 * 64 / SR}] * 2, "cpu")  # 40 frames of 64 samples
    cond["inpaint_mask"] = [torch.zeros(2, 1, 40)]
    cond["inpaint_masked_input"] = [torch.zeros(2, 8, 40)]

    with torch.no_grad():
        prediction = gt.predict_velocity(model, noised, t, mp3=mp3, inpaint_mask_value=0.0)
        expected = base(noised, t, cond=cond, padding_mask=torch.ones(2, 40, dtype=torch.bool))

    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)


def test_trained_model_output_depends_on_the_mp3(monkeypatch, tiny_checkpoint):
    """SA3 silently skips a conditioning key it is not given, so this is what
    proves the MP3 reaches the model at all."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)
    optimizer = torch.optim.AdamW(gt.add_lora(model, gt.lora_config(rank=4, lora_alpha=4)), lr=1e-2)
    clean = torch.randn(4, 8, 40, generator=torch.Generator().manual_seed(1))
    mp3 = torch.randn(4, 8, 40, generator=torch.Generator().manual_seed(2))
    gt.training_step(model, clean, mp3=mp3, inpaint_mask_value=0.0).backward()
    optimizer.step()
    noised = torch.randn(1, 8, 40, generator=torch.Generator().manual_seed(3))
    t = torch.tensor([0.5])

    with torch.no_grad():
        silent = gt.predict_velocity(
            model, noised, t, mp3=torch.zeros(1, 8, 40), inpaint_mask_value=0.0
        )
        loud = gt.predict_velocity(
            model, noised, t, mp3=torch.ones(1, 8, 40), inpaint_mask_value=0.0
        )

    assert not torch.equal(silent, loud)


def test_training_step_changes_no_base_weight(monkeypatch, tiny_checkpoint):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)
    optimizer = torch.optim.AdamW(gt.add_lora(model, gt.lora_config(rank=4, lora_alpha=4)), lr=1e-2)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    clean = torch.randn(4, 8, 40, generator=torch.Generator().manual_seed(1))
    mp3 = torch.randn(4, 8, 40, generator=torch.Generator().manual_seed(2))

    gt.training_step(model, clean, mp3=mp3, inpaint_mask_value=0.0).backward()
    optimizer.step()

    changed = {
        name for name, tensor in model.state_dict().items() if not torch.equal(before[name], tensor)
    }
    # SA3's names for what may train: LoRA factors, and the new input's layers.
    base = {
        n
        for n in changed
        if not (n.endswith(("lora_A", "lora_B")) or ".modular_local_embeds." in n)
    }
    assert base == set()


def test_saved_weights_reload_to_the_same_validation_loss(tmp_path, monkeypatch, tiny_checkpoint):
    """A disk round trip of the trained weights: the reloaded model scores the
    same validation loss to the last bit, on the same seeded noise."""
    write_song(tmp_path / "songs", "a", seconds=1.0, bitrates=[24], seed=0)
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    config = gt.lora_config(rank=4, lora_alpha=4)
    trained = gt.load_model("small-music-base", device=CPU)
    optimizer = torch.optim.AdamW(gt.add_lora(trained, config), lr=1e-2)
    dataset = gt.LatentPairDataset(tmp_path / "songs", trained, bitrates=[24], window_frames=32)
    clean, mp3 = dataset[0]
    gt.training_step(trained, clean[None], mp3=mp3[None], inpaint_mask_value=0.0).backward()
    optimizer.step()
    gt.save_checkpoint(tmp_path / "ckpt", trained, optimizer, config=config, step=1, epoch=1)

    reloaded = gt.load_trained_model("small-music-base", tmp_path / "ckpt", device=CPU)

    settings = {"timesteps": [0.2, 0.8], "batch_size": 4, "inpaint_mask_value": 0.0}
    trained_loss = gt.validation_loss(trained, dataset, **settings)
    reloaded_loss = gt.validation_loss(reloaded, dataset, **settings)
    assert reloaded_loss == trained_loss


def test_an_epoch_holds_each_window_once_whatever_the_bitrates(
    tmp_path, monkeypatch, tiny_checkpoint
):
    """1 s is 689 frames of 64 samples, so 21 windows of 32 at either bitrate."""
    write_song(tmp_path / "songs", "a", seconds=1.0, bitrates=[24, 128], seed=0)
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)

    dataset = gt.LatentPairDataset(tmp_path / "songs", model, bitrates=[24, 128], window_frames=32)

    assert len(dataset) == 21


def test_a_training_item_pairs_its_window_with_one_of_its_mp3s(
    tmp_path, monkeypatch, tiny_checkpoint
):
    write_song(tmp_path / "songs", "a", seconds=1.0, bitrates=[24, 128], seed=0)
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    monkeypatch.setattr(gt, "encode", mean_encode)
    model = gt.load_model("small-music-base", device=CPU)
    dataset = gt.LatentPairDataset(tmp_path / "songs", model, bitrates=[24, 128], window_frames=32)

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
    dataset = gt.LatentPairDataset(
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

    drawn = gt.fixed_bitrates(tmp_path / "bitrates.json", windows, bitrates=[24, 128])

    assert json.loads((tmp_path / "bitrates.json").read_text())["bitrates"] == drawn
    assert len(drawn) == 3
    assert set(drawn) <= {24, 128}


def test_fixed_bitrates_refuse_a_draw_stored_for_other_windows(tmp_path):
    """A new window length moves every window, so the stored draw no longer fits."""
    gt.fixed_bitrates(tmp_path / "bitrates.json", [("a", 0), ("a", 107)], bitrates=[24, 128])

    with pytest.raises(ValueError, match="other windows"):
        gt.fixed_bitrates(tmp_path / "bitrates.json", [("a", 0), ("a", 64)], bitrates=[24, 128])


def test_fixed_bitrates_refuse_a_draw_from_another_bitrate_list(tmp_path):
    """A draw from [24, 32] read by a six-bitrate run would quietly validate it at
    its two hardest bitrates only."""
    gt.fixed_bitrates(tmp_path / "bitrates.json", [("a", 0), ("a", 107)], bitrates=[24, 32])

    with pytest.raises(ValueError, match="drawn from"):
        gt.fixed_bitrates(
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
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=50)

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
    dataset = gt.RemixDataset(tmp_path, bitrates=[24, 128], window_samples=2048, windows=50)

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
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=10)

    clean, mp3 = dataset[0]

    torch.testing.assert_close(mp3, clean, rtol=0, atol=0)


def test_every_remix_item_is_a_mixture_window_or_a_sum_of_stems(tmp_path, monkeypatch):
    """A mixture window holds the mixture's sine and no stem's; a remix holds no
    trace of the mixture. 1e-3 sits far above the 16-bit noise in a bin (about
    3e-7) and far below the sines (0.1 and 0.5)."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=200)

    spectra = [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) * 2 / 2048 for index in range(200)]

    mixtures = [s for s in spectra if s[CYCLES["mixture"]] > 0.25]
    remixes = [s for s in spectra if s[CYCLES["mixture"]] <= 0.25]
    assert max(s[CYCLES[stem]] for s in mixtures for stem in STEMS) < 1e-3
    assert max(s[CYCLES["mixture"]] for s in remixes) < 1e-3


def test_half_the_remix_items_are_real_mixture_windows(tmp_path, monkeypatch):
    """Apollo's split. Of 400 fair draws, 160 to 240 land on mixtures: four
    standard deviations (10 draws each) around 200."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=400)

    spectra = [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) * 2 / 2048 for index in range(400)]

    assert 160 <= sum(s[CYCLES["mixture"]] > 0.25 for s in spectra) <= 240


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
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=100)

    power = np.stack(
        [np.abs(np.fft.rfft(dataset[index][0][0].numpy())) ** 2 for index in range(100)]
    )

    assert max(codec.peaks) == 1.0
    outside = np.delete(power, list(CYCLES.values()), axis=1).sum(axis=1)
    assert (outside / power.sum(axis=1)).max() < 1e-6


def test_a_mixture_window_below_full_scale_keeps_its_level(tmp_path, monkeypatch):
    """Where Apollo would scale every window to peak at 1. The 0.1 sine reads back
    as 3_276 / 32_768 from 16 bits; abs=1e-3 covers that and the sampled peak."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.1, "bass": 0.05, "drums": 0.05, "other": 0.05, "vocals": 0.05}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=50)

    cleans = [dataset[index][0][0].numpy() for index in range(50)]

    mixtures = [c for c in cleans if np.abs(np.fft.rfft(c))[CYCLES["mixture"]] * 2 / 2048 > 0.05]
    assert mixtures
    assert [float(np.abs(c).max()) for c in mixtures] == pytest.approx(
        [0.1] * len(mixtures), abs=1e-3
    )


def test_remix_windows_come_from_the_active_stretch_of_each_track(tmp_path, monkeypatch):
    """Every track is silent for its first 0.5 s, so without the activity check
    about half the mixture windows alone would be silence. The quietest item a
    window from the sines can make is one stem at -10 dB: an RMS near 0.022."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.5, sample_rate=SR
    )
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=100)

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(100)]

    assert min(rms) > 0.01


def test_a_stem_silent_throughout_is_never_drawn(tmp_path, monkeypatch):
    """An instrumental's vocals keep no active window. Drawn anyway, a remix of
    vocals alone would be silence: half the items are remixes, a quarter of
    those hold 1 to 4 stems each, all vocals with probability 1/4 to 1/256,
    so about 4% of items, 8 of these 200."""
    torch.manual_seed(0)
    monkeypatch.setattr(musdb_eval, "codec_simu", HalvingCodec())
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.0}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )
    dataset = gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=200)

    rms = [float(dataset[index][0].square().mean().sqrt()) for index in range(200)]

    assert min(rms) > 0.01


def test_active_starts_keep_windows_whose_frames_are_mostly_loud():
    """10 frames of 441 samples, the first 6 at a steady 0.5 and the rest silent;
    a window is 4 frames. The windows starting at frames 0 to 6 hold 4, 4, 4, 3,
    2, 1 and 0 loud frames, so those at frames 0 to 3 keep more than half. The
    steady frames all equal the percentile, which counts as loud."""
    track = np.zeros((2, 4_410), dtype=np.int16)
    track[:, : 6 * 441] = 16_384  # 0.5 of full scale

    starts = gt.active_starts(track, window_samples=4 * 441)

    assert starts.tolist() == [0, 441, 882, 1_323]


def test_remix_dataset_refuses_a_track_shorter_than_one_window(tmp_path):
    """1_764 samples, short of a 2_048-sample window, before its codec pads."""
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=0.04, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=SR
    )

    with pytest.raises(ValueError, match="shorter than one window"):
        gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=10)


def test_remix_dataset_refuses_audio_the_codec_cannot_take(tmp_path):
    """codec_simu runs at 44.1 kHz; MUSDB18-HQ's files are all 44.1 kHz."""
    amplitudes = {"mixture": 0.5, "bass": 0.1, "drums": 0.1, "other": 0.1, "vocals": 0.1}
    write_musdb_song(
        tmp_path, "a", seconds=1.0, amplitudes=amplitudes, silent_seconds=0.0, sample_rate=48_000
    )

    with pytest.raises(ValueError, match="48000 Hz"):
        gt.RemixDataset(tmp_path, bitrates=[24], window_samples=2048, windows=10)


def test_resume_restores_the_saved_training_state(tmp_path, monkeypatch, tiny_checkpoint):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)
    params = gt.add_lora(model, gt.lora_config(rank=4, lora_alpha=4))
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    clean = torch.randn(2, 8, 40, generator=torch.Generator().manual_seed(1))
    mp3 = torch.randn(2, 8, 40, generator=torch.Generator().manual_seed(2))
    gt.training_step(model, clean, mp3=mp3, inpaint_mask_value=0.0).backward()
    optimizer.step()
    config = gt.lora_config(rank=4, lora_alpha=4)
    gt.save_checkpoint(tmp_path, model, optimizer, config=config, step=7, epoch=3)
    resumed = torch.optim.AdamW(params, lr=1e-3)

    step, epoch = gt.load_training_state(tmp_path, resumed)

    assert (step, epoch) == (7, 3)
    torch.testing.assert_close(
        resumed.state_dict()["state"][0]["exp_avg"],
        optimizer.state_dict()["state"][0]["exp_avg"],
        rtol=0,
        atol=0,
    )


def test_resume_takes_the_learning_rate_from_the_config(tmp_path, monkeypatch, tiny_checkpoint):
    """torch's Optimizer.load_state_dict brings back the saved lr; the new
    run's lr, the one its optimizer was built with, must replace it."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)
    params = gt.add_lora(model, gt.lora_config(rank=4, lora_alpha=4))
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    config = gt.lora_config(rank=4, lora_alpha=4)
    gt.save_checkpoint(tmp_path, model, optimizer, config=config, step=1, epoch=1)
    resumed = torch.optim.AdamW(params, lr=5e-5)

    gt.load_training_state(tmp_path, resumed)

    assert resumed.param_groups[0]["lr"] == 5e-5


def test_load_model_refuses_a_checkpoint_missing_base_weights(monkeypatch, tiny_checkpoint):
    config, weights = tiny_checkpoint
    weights = {
        name: value for name, value in weights.items() if "transformer.layers.1." not in name
    }
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: (config, weights))

    with pytest.raises(RuntimeError, match="base weights missing"):
        gt.load_model("small-music-base", device=CPU)


def test_reloading_weights_refuses_a_different_lora(tmp_path, monkeypatch, tiny_checkpoint):
    """A different alpha would silently rescale every trained update."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=CPU)
    optimizer = torch.optim.AdamW(gt.add_lora(model, gt.lora_config(rank=4, lora_alpha=4)))
    gt.save_checkpoint(
        tmp_path, model, optimizer, config=gt.lora_config(rank=4, lora_alpha=4), step=0, epoch=0
    )

    with pytest.raises(ValueError, match="trained with"):
        gt.load_lora_weights(
            model, tmp_path / "lora.safetensors", config=gt.lora_config(rank=4, lora_alpha=8)
        )
