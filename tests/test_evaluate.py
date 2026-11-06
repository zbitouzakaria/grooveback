"""The test-set entry point, run end to end on a tiny Stable Audio 3
(tests/conftest.py) through the real config file and the real sox MP3."""

import csv
import hashlib
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from grooveback import audio as ga
from grooveback import training as gt
from grooveback.cli import evaluate as ev

SR = 44_100
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CPU = torch.device("cpu")


def config(tmp_path, *overrides):
    """configs/evaluate.yaml on CPU, reading tmp_path/test and tmp_path/checkpoint.
    Two sampling steps keep a 6 s clip on the tiny model under a second."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(
            "evaluate",
            overrides=[
                f"musdb_test={tmp_path / 'test'}",
                f"model.checkpoint={tmp_path / 'checkpoint'}",
                f"output_dir={tmp_path / 'out'}",
                "model.variant=small-music-base",
                "model.steps=2",
                "device=cpu",
                *overrides,
            ],
        )


def save_lora(directory):
    """A LoRA of the tiny SA3 in train_lora's checkpoint layout, after one training step:
    the MP3 input starts at zero, and until it trains the output ignores the clip."""
    lora_config = gt.lora_config(rank=4, lora_alpha=4)
    model = gt.load_model("small-music-base", device=CPU)
    optimizer = torch.optim.AdamW(gt.add_lora(model, lora_config), lr=1e-2)
    clean = torch.randn(2, 8, 40, generator=torch.Generator().manual_seed(1))
    mp3 = torch.randn(2, 8, 40, generator=torch.Generator().manual_seed(2))
    gt.training_step(model, clean, mp3=mp3, inpaint_mask_value=0.0).backward()
    optimizer.step()
    gt.save_checkpoint(directory, model, optimizer, config=lora_config, step=1, epoch=1)


@pytest.fixture
def musdb_test(tmp_path):
    """Two tracks of 9 s white noise in MUSDB18-HQ's layout, one 6 s clip each, at start 0."""
    rng = np.random.default_rng(0)
    for track in ("a", "b"):
        (tmp_path / "test" / track).mkdir(parents=True)
        noise = rng.uniform(-0.5, 0.5, size=(9 * SR, 2))
        sf.write(tmp_path / "test" / track / "mixture.wav", noise, SR, subtype="PCM_16")
    return tmp_path / "test"


def test_scores_csv_has_a_row_per_clip_at_each_bitrate(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[24,32]", "seeds=[0]"))

    rows = list(csv.DictReader((tmp_path / "out" / "scores.csv").open()))
    assert [(row["kbps"], row["track"], row["start"]) for row in rows] == [
        ("24", "a", "0"),
        ("24", "b", "0"),
        ("32", "a", "0"),
        ("32", "b", "0"),
    ]


def test_writes_the_mp3s_and_renders_by_bitrate_and_seed(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[24,32]", "seeds=[1]"))

    written = sorted(
        path.relative_to(tmp_path / "out").as_posix()
        for path in (tmp_path / "out").rglob("*.wav")
    )
    assert written == [
        "mp3/24k/a/0.wav",
        "mp3/24k/b/0.wav",
        "mp3/32k/a/0.wav",
        "mp3/32k/b/0.wav",
        "render/seed1/24k/a/0.wav",
        "render/seed1/24k/b/0.wav",
        "render/seed1/32k/a/0.wav",
        "render/seed1/32k/b/0.wav",
    ]


def test_scores_csv_scores_the_mp3_each_seed_and_their_mean_on_every_metric(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[32]", "seeds=[0,1]"))

    rows = list(csv.DictReader((tmp_path / "out" / "scores.csv").open()))
    assert sorted(rows[0]) == sorted(
        ["kbps", "track", "start"]
        + [
            f"{name}_{metric}"
            for name in ("input", "seed0", "seed1", "lora")
            for metric in ("sdr", "si_snr", "spectral_snr", "lsd")
        ]
    )


@pytest.mark.parametrize("metric", ["sdr", "si_snr", "spectral_snr", "lsd"])
def test_the_lora_score_of_a_clip_is_the_mean_of_its_seeds(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test, metric
):
    """rel=1e-12 covers the rounding of a mean of two doubles written as text."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[32]", "seeds=[0,1]", "clips=1"))

    (row,) = csv.DictReader((tmp_path / "out" / "scores.csv").open())
    seeds = float(row[f"seed0_{metric}"]), float(row[f"seed1_{metric}"])
    assert seeds[0] != seeds[1]
    assert float(row[f"lora_{metric}"]) == pytest.approx(sum(seeds) / 2, rel=1e-12)


def test_an_identity_restoration_scores_exactly_like_the_mp3(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    """The gate of .claude/rules/audio.md: every path from the MP3 to its score, disk
    round trips included. A render of the wrong clip or bitrate would score differently."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    monkeypatch.setattr(ev, "lora_batch", lambda model, clips, sample_rate, **settings: clips)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[24,32]", "seeds=[0]"))

    rows = list(csv.DictReader((tmp_path / "out" / "scores.csv").open()))
    assert [row["seed0_sdr"] for row in rows] == [row["input_sdr"] for row in rows]


def test_the_model_settings_of_the_config_reach_the_solver(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    received = []

    def identity(model, clips, sample_rate, **settings):
        received.append(settings)
        return clips

    monkeypatch.setattr(ev, "lora_batch", identity)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(
        config(
            tmp_path,
            "bitrates=[32]",
            "seeds=[7]",
            "model.steps=3",
            "model.sampler_type=rk4",
            "model.inpaint_mask_value=0.5",
        )
    )

    assert received == [
        {"steps": 3, "sampler_type": "rk4", "inpaint_mask_value": 0.5, "seed": 7},
    ]


def test_the_batch_size_of_the_config_sets_the_clips_per_sampling_run(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    """Two clips in all, one per run."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    sizes = []

    def identity(model, clips, sample_rate, **settings):
        sizes.append(len(clips))
        return clips

    monkeypatch.setattr(ev, "lora_batch", identity)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[32]", "seeds=[0]", "batch_size=1"))

    assert sizes == [1, 1]


def test_the_variant_of_the_config_is_the_base_model_loaded(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")
    variants = []

    def recording(variant):
        variants.append(variant)
        return tiny_checkpoint

    monkeypatch.setattr(gt, "base_checkpoint", recording)

    ev.evaluate(config(tmp_path, "bitrates=[32]", "seeds=[0]", "model.variant=medium-base"))

    assert variants == ["medium-base"]


def test_prints_the_mean_scores_of_each_bitrate(
    tmp_path, monkeypatch, capsys, tiny_checkpoint, musdb_test
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[24,32]", "seeds=[0]"))

    printed = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in printed if line.endswith("over 2 clips")] == [
        "input",
        "seed0",
        "lora",
        "input",
        "seed0",
        "lora",
    ]
    assert "24 kbps" in printed
    assert "32 kbps" in printed


def test_two_seeds_restore_a_clip_differently(tmp_path, monkeypatch, tiny_checkpoint, musdb_test):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")

    ev.evaluate(config(tmp_path, "bitrates=[32]", "seeds=[0,1]", "clips=1"))

    renders = tmp_path / "out" / "render"
    first = ga.load(next((renders / "seed0" / "32k").glob("*/*.wav")))[0]
    second = ga.load(next((renders / "seed1" / "32k").glob("*/*.wav")))[0]
    assert np.abs(first - second).max() > 1e-3


def test_a_second_run_keeps_the_renders_it_finds(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    """The render is rewritten at half level first: sampling again would give back the
    original, so an unchanged hash proves it was not done again."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")
    cfg = config(tmp_path, "bitrates=[32]", "seeds=[0]", "clips=1")
    ev.evaluate(cfg)
    render = next((tmp_path / "out" / "render" / "seed0" / "32k").glob("*/*.wav"))
    ga.save(render, 0.5 * ga.load(render)[0], SR)
    kept = hashlib.sha256(render.read_bytes()).hexdigest()

    ev.evaluate(cfg)

    assert hashlib.sha256(render.read_bytes()).hexdigest() == kept


def test_a_second_run_restores_the_renders_it_lacks(
    tmp_path, monkeypatch, tiny_checkpoint, musdb_test
):
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    save_lora(tmp_path / "checkpoint")
    cfg = config(tmp_path, "bitrates=[32]", "seeds=[0]", "clips=1")
    ev.evaluate(cfg)
    render = next((tmp_path / "out" / "render" / "seed0" / "32k").glob("*/*.wav"))
    render.unlink()

    ev.evaluate(cfg)

    assert render.is_file()


def test_refuses_a_folder_without_mixtures_before_loading_the_model(tmp_path):
    (tmp_path / "test").mkdir()

    with mock.patch.object(gt, "load_trained_model", autospec=True) as load_trained_model:
        with pytest.raises(FileNotFoundError, match="mixture"):
            ev.evaluate(config(tmp_path))

    load_trained_model.assert_not_called()


def test_an_unset_checkpoint_is_named_in_the_error():
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose("evaluate")

    with pytest.raises(MissingMandatoryValue, match="checkpoint"):
        ev.evaluate(cfg)


def test_the_checkpoint_is_the_only_setting_left_to_give():
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose("evaluate", overrides=["model.checkpoint=run/checkpoint-7"])

    assert OmegaConf.missing_keys(cfg) == set()


def test_the_default_output_dir_changes_with_the_sampling_settings():
    """Renders on disk are kept, so a run with other settings must not read them."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(
            "evaluate", overrides=["model.checkpoint=run/checkpoint-7", "model.steps=32"]
        )

    assert cfg.output_dir == "run/checkpoint-7/test-euler-32"


class TestRender:
    def test_restores_batch_size_clips_per_sampling_run(self, tmp_path, monkeypatch):
        ga.save(tmp_path / "mp3" / "a" / "0.wav", np.zeros((2, SR), dtype=np.float32), SR)
        ga.save(tmp_path / "mp3" / "a" / "1.wav", np.zeros((2, SR), dtype=np.float32), SR)
        ga.save(tmp_path / "mp3" / "b" / "0.wav", np.zeros((2, SR), dtype=np.float32), SR)
        sizes = []

        def identity(model, clips, sample_rate, **settings):
            sizes.append(len(clips))
            return clips

        monkeypatch.setattr(ev, "lora_batch", identity)

        ev.render(
            mock.Mock(sample_rate=SR),
            tmp_path / "mp3",
            tmp_path / "render",
            batch_size=2,
            steps=2,
            inpaint_mask_value=0.0,
            sampler_type="euler",
            seed=0,
        )

        assert sizes == [2, 1]

    def test_refuses_a_non_finite_restoration_before_saving_it(self, tmp_path, monkeypatch):
        """A NaN in a render reaches the scores as nan and nothing fails."""
        ga.save(tmp_path / "mp3" / "a" / "0.wav", np.zeros((2, SR), dtype=np.float32), SR)
        monkeypatch.setattr(
            ev,
            "lora_batch",
            lambda model, clips, sample_rate, **settings: [np.full((2, SR), np.nan)],
        )

        with pytest.raises(ValueError, match="not finite"):
            ev.render(
                mock.Mock(sample_rate=SR),
                tmp_path / "mp3",
                tmp_path / "render",
                batch_size=2,
                steps=2,
                inpaint_mask_value=0.0,
                sampler_type="euler",
                seed=0,
            )

        assert not (tmp_path / "render").exists()
