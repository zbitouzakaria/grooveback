"""The fine-tune entry point, run end to end on a tiny Stable Audio 3
(tests/conftest.py) through the real config file."""

from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from grooveback import audio as ga
from grooveback import training as gt
from grooveback.cli.train_lora import train

SR = 44_100
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def write_song(root, name, *, seconds, bitrates, seed):
    """One song in prepare_dataset's layout: white noise, and a quieter copy of it per bitrate."""
    rng = np.random.default_rng(seed)
    original = rng.uniform(-0.5, 0.5, size=(2, int(seconds * SR))).astype(np.float32)
    ga.save(root / name / "original.wav", original, SR)
    for kbps in bitrates:
        ga.save(root / name / f"{kbps}k.wav", 0.5 * original, SR)


def config(tmp_path, *overrides):
    """configs/train_lora.yaml on CPU with small windows. 0.5 s windows are 344
    frames of 64 samples on the tiny model, long enough for the listening
    pack's loudness meter (400 ms blocks)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(
            "train_lora",
            overrides=[
                f"train_dir={tmp_path / 'train'}",
                f"valid_dir={tmp_path / 'valid'}",
                f"save_dir={tmp_path / 'run'}",
                "device=cpu",
                "window_seconds=0.5",
                "batch_size=4",
                "bitrates=[24,32]",
                "sample_steps=2",
                "rank=4",
                "lora_alpha=4",
                *overrides,
            ],
        )


def test_one_epoch_writes_weights_optimizer_state_and_a_listening_pack(
    tmp_path, monkeypatch, tiny_checkpoint
):
    """2 s is 1_378 frames, so 4 windows of 344 at 2 bitrates: 8 items, 2 steps of 4."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    train(config(tmp_path))

    checkpoint = tmp_path / "run" / "checkpoint-2"
    assert (checkpoint / "lora.safetensors").is_file()
    assert (checkpoint / "state.pt").is_file()
    assert sorted(path.name for path in (checkpoint / "listen").iterdir()) == [
        "input.flac",
        "original-roundtrip.flac",
        "original.flac",
        "restored.flac",
    ]


def test_resume_continues_from_the_checkpoint_with_the_new_learning_rate(
    tmp_path, monkeypatch, capsys, tiny_checkpoint
):
    """The resumed run starts at step 2, in epoch 2, and its optimizer runs at
    the new lr rather than the 1e-4 saved in checkpoint-2."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)
    train(config(tmp_path, "lr=1e-4"))
    capsys.readouterr()

    train(
        config(
            tmp_path,
            "num_train_epochs=2",
            "lr=5e-5",
            f"resume_from_checkpoint={tmp_path / 'run' / 'checkpoint-2'}",
        )
    )

    log = capsys.readouterr().out
    assert "step 0 validation loss" not in log
    assert "epoch 2 step 4 validation loss" in log
    state = torch.load(tmp_path / "run" / "checkpoint-4" / "state.pt", weights_only=True)
    assert state["optimizer"]["param_groups"][0]["lr"] == 5e-5
