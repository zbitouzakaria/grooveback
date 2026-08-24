"""The CLI is where a user enters, so these tests run file in, file out
through the real wiring — argparse, excerpting, the wrapper, disk. A renamed
flag or a broken wire between argparse and a wrapper fails here and nowhere
else. Only true externals are doubled, each at its own seam: the checkpoint
load (an identity model) and the A2SB subprocess (a fake fork)."""

import stat
from pathlib import Path

import numpy as np
import pytest
import torch

from grooveback import audio as ga
from grooveback import baselines
from grooveback.cli.baseline import main

SR = 44_100


class IdentityModel(torch.nn.Module):
    """`run_apollo` reads the target device off the first parameter."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, audio):
        return audio


def fake_fork(fork_dir: Path, script: str) -> Path:
    """A directory that looks like the fork: restore.py plus a .venv python
    whose behaviour is `script`."""
    fork_dir.mkdir(parents=True, exist_ok=True)
    (fork_dir / "restore.py").write_text("")
    python = fork_dir / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(script)
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return fork_dir


def test_apollo_writes_the_restored_file(tmp_path, monkeypatch):
    """Chunk, overlap and pad are shorter than the 3 s input so the chunked
    path runs; an identity model must then come back unchanged. atol covers
    float32 crossfade arithmetic on identical values."""
    rng = np.random.default_rng(0)
    audio = rng.uniform(-1.0, 1.0, size=(2, 3 * SR)).astype(np.float32)
    ga.save(tmp_path / "in.wav", audio, SR)
    monkeypatch.setattr(baselines, "load_apollo", lambda **kwargs: IdentityModel())

    main(
        [
            str(tmp_path / "in.wav"),
            str(tmp_path / "out.wav"),
            "--method", "apollo",
            "--chunk-seconds", "1.0",
            "--overlap-seconds", "0.2",
            "--chunk-pad-seconds", "0.1",
        ]
    )

    restored, sample_rate = ga.load(tmp_path / "out.wav")
    assert sample_rate == SR
    np.testing.assert_allclose(restored, audio, rtol=0, atol=1e-6)


def test_a2sb_writes_the_restored_file(tmp_path, monkeypatch):
    """The fake fork copies input to output: argv is (restore.py, in, out, ...)."""
    audio = np.tile(np.linspace(-0.5, 0.5, SR, dtype=np.float32), (2, 1))
    ga.save(tmp_path / "in.wav", audio, SR)
    monkeypatch.setattr(
        baselines, "A2SB_DIR", fake_fork(tmp_path / "fork", '#!/bin/sh\ncp "$2" "$3"\n')
    )

    main([str(tmp_path / "in.wav"), str(tmp_path / "out.wav"), "--method", "a2sb"])

    restored, sample_rate = ga.load(tmp_path / "out.wav")
    assert sample_rate == SR
    assert restored.shape == audio.shape
    np.testing.assert_array_equal(restored[0], audio[0])


def test_excerpt_flags_select_the_requested_slice(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    audio = rng.uniform(-1.0, 1.0, size=(2, 3 * SR)).astype(np.float32)
    ga.save(tmp_path / "in.wav", audio, SR)
    monkeypatch.setattr(baselines, "load_apollo", lambda **kwargs: IdentityModel())

    main(
        [
            str(tmp_path / "in.wav"),
            str(tmp_path / "out.wav"),
            "--chunk-seconds", "1.0",
            "--overlap-seconds", "0.2",
            "--chunk-pad-seconds", "0.1",
            "--start", "1.0",
            "--seconds", "1.0",
        ]
    )

    restored, _ = ga.load(tmp_path / "out.wav")
    np.testing.assert_allclose(restored, audio[:, SR : 2 * SR], rtol=0, atol=1e-6)


def test_match_loudness_flag_normalizes_the_output(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    audio = (0.5 * rng.uniform(-1.0, 1.0, size=(2, 2 * SR))).astype(np.float32)
    ga.save(tmp_path / "in.wav", audio, SR)
    monkeypatch.setattr(baselines, "load_apollo", lambda **kwargs: IdentityModel())

    main(
        [
            str(tmp_path / "in.wav"),
            str(tmp_path / "out.wav"),
            "--chunk-seconds", "1.0",
            "--overlap-seconds", "0.2",
            "--chunk-pad-seconds", "0.1",
            "--match-loudness",
        ]
    )

    restored, _ = ga.load(tmp_path / "out.wav")
    assert ga.loudness(restored, SR) == pytest.approx(ga.TARGET_LUFS, abs=0.1)
