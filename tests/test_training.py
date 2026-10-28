"""The pieces of the LoRA fine-tune, run on a tiny Stable Audio 3 built by
SA3's own factory (tests/conftest.py), so the real model code runs on CPU
with no download."""

import json

import numpy as np
import pytest
import torch

from grooveback import audio as ga
from grooveback import training as gt

SR = 44_100
CPU = torch.device("cpu")


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
