"""The A2SB wrapper is a process boundary, so its failure modes are file-system
shaped: fork not cloned, environment not built, subprocess dying. Each must
fail loudly with the fix in the message — a boundary that fails vaguely costs
a debugging session."""

import stat
import sys

import numpy as np
import pytest

from grooveback import baselines

SR = 44_100


def signal(seconds: float = 0.1, channels: int = 2) -> np.ndarray:
    return np.zeros((channels, int(seconds * SR)), dtype=np.float32)


def fake_fork(tmp_path, script: str):
    """A directory that looks like the fork: restore.py plus a .venv python
    whose behaviour is `script`."""
    (tmp_path / "restore.py").write_text("")
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(script)
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def point_wrapper_at(monkeypatch, fork_dir):
    monkeypatch.setattr(baselines, "A2SB_DIR", fork_dir)
    monkeypatch.setattr(baselines, "A2SB_PYTHON", fork_dir / ".venv" / "bin" / "python")


def test_missing_fork_says_how_to_clone(monkeypatch, tmp_path):
    point_wrapper_at(monkeypatch, tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="git clone"):
        baselines.run_a2sb(signal(), SR)


def test_missing_env_says_how_to_build_it(monkeypatch, tmp_path):
    (tmp_path / "restore.py").write_text("")
    point_wrapper_at(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="setup.sh"):
        baselines.run_a2sb(signal(), SR)


def test_subprocess_failure_surfaces_its_stderr(monkeypatch, tmp_path):
    fork = fake_fork(tmp_path, "#!/bin/sh\necho the-model-exploded >&2\nexit 3\n")
    point_wrapper_at(monkeypatch, fork)
    with pytest.raises(RuntimeError, match="the-model-exploded"):
        baselines.run_a2sb(signal(), SR)


def test_missing_output_is_a_failure_even_on_exit_zero(monkeypatch, tmp_path):
    fork = fake_fork(tmp_path, "#!/bin/sh\nexit 0\n")
    point_wrapper_at(monkeypatch, fork)
    with pytest.raises(RuntimeError, match="A2SB failed"):
        baselines.run_a2sb(signal(), SR)


def test_wrong_sample_rate_rejected_before_any_subprocess():
    with pytest.raises(ValueError, match="44100"):
        baselines.run_a2sb(signal(), 48_000)


def test_mono_result_is_copied_across_input_channels(monkeypatch, tmp_path):
    # a "model" that copies input to output: argv is (restore.py, in, out, ...)
    fork = fake_fork(tmp_path, '#!/bin/sh\ncp "$2" "$3"\n')
    point_wrapper_at(monkeypatch, fork)
    stereo = np.stack(
        [np.linspace(-0.5, 0.5, SR, dtype=np.float32),
         np.linspace(0.5, -0.5, SR, dtype=np.float32)]
    )
    out = baselines.run_a2sb(stereo, SR)
    assert out.shape == stereo.shape
    assert np.array_equal(out[0], out[1]), "output must be mono in the input's shape"


def test_output_padded_back_to_input_length(monkeypatch, tmp_path):
    # a "model" that writes a shorter file, as A2SB does (~408 samples short)
    script = (
        "#!/bin/sh\n"
        f"{sys.executable} -c \"import soundfile as sf, numpy as np, sys; "
        "a, sr = sf.read(sys.argv[1], always_2d=True); "
        "sf.write(sys.argv[2], a[:-408], sr)\" \"$2\" \"$3\"\n"
    )
    fork = fake_fork(tmp_path, script)
    point_wrapper_at(monkeypatch, fork)
    audio = signal(seconds=1.0)
    out = baselines.run_a2sb(audio, SR)
    assert out.shape[1] == audio.shape[1]
    assert np.all(out[:, -408:] == 0.0)


def test_fork_sha_is_reported(monkeypatch, tmp_path, capsys):
    fork = fake_fork(tmp_path, '#!/bin/sh\ncp "$2" "$3"\n')
    point_wrapper_at(monkeypatch, fork)
    baselines.run_a2sb(signal(), SR)
    assert "fork @" in capsys.readouterr().out
