"""The A2SB config is generated text handed to a subprocess.

Nothing type-checks it, and a wrong value fails as a Lightning stack trace
minutes later, or worse, silently runs on the wrong device or checkpoint.
"""

from pathlib import Path

import pytest
import yaml

from grooveback.baselines import _a2sb_config


def parse(**kwargs):
    defaults = {
        "wav_in": Path("/tmp/x/in.wav"),
        "checkpoints": ["/ckpt/one.ckpt"],
        "device": "mps",
        "cutoff_hz": None,
    }
    return yaml.safe_load(_a2sb_config(**{**defaults, **kwargs}))


def test_overrides_the_cluster_defaults():
    """A2SB ships accelerator=gpu, ddp and a SLURM plugin. None of that applies."""
    cfg = parse()
    assert cfg["trainer"]["accelerator"] == "mps"
    assert cfg["trainer"]["strategy"] == "auto"
    assert cfg["trainer"]["devices"] == 1
    assert cfg["trainer"]["plugins"] is None


def test_single_checkpoint_has_no_time_cutoff():
    """t_cutoffs partitions the time range between checkpoints; one model needs none."""
    cfg = parse(checkpoints=["/ckpt/one.ckpt"])
    assert cfg["model"]["pretrained_checkpoints"] == ["/ckpt/one.ckpt"]
    assert cfg["model"]["t_cutoffs"] == []


def test_two_checkpoints_split_the_time_range():
    cfg = parse(checkpoints=["/ckpt/a.ckpt", "/ckpt/b.ckpt"])
    assert len(cfg["model"]["pretrained_checkpoints"]) == 2
    assert cfg["model"]["t_cutoffs"] == [0.5]


def test_predict_filelist_points_at_the_input():
    cfg = parse(wav_in=Path("/tmp/run/ch0_in.wav"))
    assert cfg["data"]["predict_filelist"][0]["filepath"] == "/tmp/run/ch0_in.wav"


def test_dataloader_workers_disabled():
    """Their default of 23 is sized for a cluster node."""
    assert parse()["data"]["num_workers"] == 0


def test_cutoff_omitted_by_default_so_a2sb_detects_it():
    assert "transforms_aug" not in parse()


def test_cutoff_when_given_is_a_fixed_band():
    cfg = parse(cutoff_hz=16000)
    args = cfg["data"]["transforms_aug"][0]["init_args"]
    assert args["upsample_mask_kwargs"] == {
        "min_cutoff_freq": 16000,
        "max_cutoff_freq": 16000,
    }
    assert args["p_upsample_mask"] == 1.0


@pytest.mark.parametrize("device", ["mps", "cpu", "gpu"])
def test_device_passes_through(device):
    assert parse(device=device)["trainer"]["accelerator"] == device
