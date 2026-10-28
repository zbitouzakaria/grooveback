"""The fine-tune entry point, run end to end on a tiny Stable Audio 3
(tests/conftest.py) through the real config file."""

from pathlib import Path
from unittest import mock

import mlflow
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from safetensors.torch import load_file

from grooveback import audio as ga
from grooveback import training as gt
from grooveback.cli import musdb_eval
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


def write_musdb_song(root, name, *, seconds, seed):
    """One song in MUSDB18-HQ's layout, 16-bit: white noise per stem, the mixture their sum."""
    rng = np.random.default_rng(seed)
    stems = {
        kind: rng.uniform(-0.2, 0.2, size=(2, int(seconds * SR))).astype(np.float32)
        for kind in ("bass", "drums", "other", "vocals")
    }
    for kind, audio in {**stems, "mixture": sum(stems.values())}.items():
        ga.save(root / name / f"{kind}.wav", audio, SR, subtype="PCM_16")


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
                f"mlflow_db={tmp_path / 'mlflow.db'}",
                # Overridden like a user would, which a name clash with a config group breaks.
                "variant=small-music-base",
                "device=cpu",
                "log_system_metrics=false",
                "window_seconds=0.5",
                "batch_size=2",
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
    """2 s is 1_378 frames, so 4 windows of 344, one item each: 2 steps of 2."""
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


def test_remix_training_runs_epochs_of_epoch_windows_items(tmp_path, monkeypatch, tiny_checkpoint):
    """4 items at batch 2 make 2 steps an epoch, so 2 epochs end at checkpoint-4,
    which is kept as the last; checkpoint-2 stays only if it is the best.
    In-process loading: workers spawned on macOS would not see the codec stand-in."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    monkeypatch.setattr(musdb_eval, "codec_simu", lambda clip, kbps: 0.5 * clip)
    write_musdb_song(tmp_path / "musdb", "a", seconds=2.0, seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    train(
        config(
            tmp_path,
            "train_dir=null",
            f"remix_dir={tmp_path / 'musdb'}",
            "epoch_windows=4",
            "num_workers=0",
            "num_train_epochs=2",
        )
    )

    names = {path.name for path in (tmp_path / "run").iterdir()}
    assert "checkpoint-4" in names
    assert names <= {"checkpoint-2", "checkpoint-4"}


def test_remix_training_steps_on_windows_of_window_seconds(tmp_path, monkeypatch, tiny_checkpoint):
    """0.5 s is 344 frames of 64 samples on the tiny model: the remixes must be
    encoded to as many latent frames as the validation windows."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    monkeypatch.setattr(musdb_eval, "codec_simu", lambda clip, kbps: 0.5 * clip)
    frames = []
    training_step = gt.training_step

    def recording_step(model, clean, *, mp3, inpaint_mask_value):
        frames.append((clean.shape[-1], mp3.shape[-1]))
        return training_step(model, clean, mp3=mp3, inpaint_mask_value=inpaint_mask_value)

    monkeypatch.setattr(gt, "training_step", recording_step)
    write_musdb_song(tmp_path / "musdb", "a", seconds=2.0, seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    train(
        config(
            tmp_path,
            "train_dir=null",
            f"remix_dir={tmp_path / 'musdb'}",
            "epoch_windows=4",
            "num_workers=0",
        )
    )

    assert set(frames) == {(344, 344)}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(["remix_dir=musdb"], id="both"),
        pytest.param(["train_dir=null"], id="neither"),
    ],
)
def test_refuses_anything_but_exactly_one_training_set_before_loading_the_model(
    tmp_path, overrides
):
    with mock.patch.object(gt, "load_model", autospec=True) as load_model:
        with pytest.raises(ValueError, match="exactly one"):
            train(config(tmp_path, *overrides))

    load_model.assert_not_called()


def test_refuses_a_song_in_both_training_and_validation_before_loading_the_model(tmp_path):
    """MUSDB18-HQ's own train/ folder holds musdb's validation songs too."""
    write_musdb_song(tmp_path / "musdb", "b", seconds=2.0, seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    with mock.patch.object(gt, "load_model", autospec=True) as load_model:
        with pytest.raises(ValueError, match="in both the training and the validation"):
            train(config(tmp_path, "train_dir=null", f"remix_dir={tmp_path / 'musdb'}"))

    load_model.assert_not_called()


def test_mlflow_metrics_are_indexed_by_epoch_with_the_untrained_model_at_0(
    tmp_path, monkeypatch, tiny_checkpoint
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    train(config(tmp_path, "num_train_epochs=2"))

    client = mlflow.tracking.MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    (run,) = client.search_runs([client.get_experiment_by_name("train_lora").experiment_id])
    validation = client.get_metric_history(run.info.run_id, "validation_loss")
    training = client.get_metric_history(run.info.run_id, "training_loss")
    assert [metric.step for metric in validation] == [0, 1, 2]
    assert [metric.step for metric in training] == [1, 2]


def test_mlflow_batch_loss_is_indexed_by_step(tmp_path, monkeypatch, tiny_checkpoint):
    """One epoch of 2 steps, logged every step."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    train(config(tmp_path, "log_every=1"))

    client = mlflow.tracking.MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    (run,) = client.search_runs([client.get_experiment_by_name("train_lora").experiment_id])
    batch = client.get_metric_history(run.info.run_id, "batch_loss")
    assert [metric.step for metric in batch] == [1, 2]


def test_time_budget_stops_after_the_epoch_that_crosses_it(tmp_path, monkeypatch, tiny_checkpoint):
    """A budget of 0 minutes is crossed by the first epoch, so a 3-epoch run
    ends after it: 2 steps, checkpoint-2, with the listening pack there."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    train(config(tmp_path, "num_train_epochs=3", "max_train_minutes=0"))

    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == ["checkpoint-2"]
    assert (tmp_path / "run" / "checkpoint-2" / "listen" / "restored.flac").is_file()


def test_keeps_only_the_checkpoint_with_the_lowest_validation_loss_and_the_last(
    tmp_path, monkeypatch, tiny_checkpoint
):
    """Validation losses of 1.0 untrained, then 0.5, 0.3 and 0.4 after three
    epochs of 2 steps: checkpoint-4 is the best, checkpoint-6 the last."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    with mock.patch.object(gt, "validation_loss", autospec=True, side_effect=[1.0, 0.5, 0.3, 0.4]):
        train(config(tmp_path, "num_train_epochs=3"))

    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "checkpoint-4",
        "checkpoint-6",
    ]


def test_the_last_and_the_best_listening_packs_are_restored_with_their_own_weights(
    tmp_path, monkeypatch, tiny_checkpoint
):
    """checkpoint-6 is the last and checkpoint-4 the best. The tiny model's random
    decoder returns the same audio for any latents, so the packs cannot tell the
    two models apart; the weights each restore runs with can."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    restored_with = []
    restore = gt.restore

    def recording_restore(model, mp3, **settings):
        restored_with.append({name: t.clone() for name, t in gt._trainable_state(model).items()})
        return restore(model, mp3, **settings)

    monkeypatch.setattr(gt, "restore", recording_restore)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)

    with mock.patch.object(gt, "validation_loss", autospec=True, side_effect=[1.0, 0.5, 0.3, 0.4]):
        train(config(tmp_path, "num_train_epochs=3"))

    torch.testing.assert_close(
        restored_with,
        [
            load_file(tmp_path / "run" / "checkpoint-6" / "lora.safetensors"),
            load_file(tmp_path / "run" / "checkpoint-4" / "lora.safetensors"),
        ],
        rtol=0,
        atol=0,
    )
    assert (tmp_path / "run" / "checkpoint-4" / "listen" / "restored.flac").is_file()


def test_resume_never_deletes_the_checkpoints_of_earlier_runs(
    tmp_path, monkeypatch, tiny_checkpoint
):
    """The resumed run's losses, 0.6 then 0.4, make its checkpoint-6 both its best
    and its last, so its checkpoint-4 is deleted. checkpoint-2, written by the
    first run, is neither for the resumed run and stays."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    write_song(tmp_path / "train", "a", seconds=2.0, bitrates=[24, 32], seed=0)
    write_song(tmp_path / "valid", "b", seconds=2.0, bitrates=[24, 32], seed=1)
    train(config(tmp_path))

    with mock.patch.object(gt, "validation_loss", autospec=True, side_effect=[0.6, 0.4]):
        train(
            config(
                tmp_path,
                "num_train_epochs=3",
                f"resume_from_checkpoint={tmp_path / 'run' / 'checkpoint-2'}",
            )
        )

    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "checkpoint-2",
        "checkpoint-6",
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
