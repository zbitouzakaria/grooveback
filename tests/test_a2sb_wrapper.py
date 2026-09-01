"""The A2SB wrapper is a process boundary, so its failure modes are file-system
shaped: fork not cloned, environment not built, subprocess dying. Each must
fail loudly with the fix in the message — a boundary that fails vaguely costs
a debugging session."""

import stat
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from grooveback import baselines

SR = 44_100


def silence(seconds: float, channels: int = 2) -> np.ndarray:
    """`(channels, samples)` float32 silence."""
    return np.zeros((channels, int(seconds * SR)), dtype=np.float32)


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


def test_stereo_render_comes_back_unmixed(monkeypatch, tmp_path):
    """The fork restores stereo per channel, so distinct channels must survive.

    The fake fork copies input to output: argv is (restore.py, in, out, ...).
    """
    monkeypatch.setattr(baselines, "A2SB_DIR", fake_fork(tmp_path, '#!/bin/sh\ncp "$2" "$3"\n'))
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, SR, dtype=np.float32),
            np.linspace(0.5, -0.5, SR, dtype=np.float32),
        ]
    )

    out = baselines.run_a2sb(stereo, SR)

    assert out.shape == stereo.shape
    np.testing.assert_array_equal(out, stereo)


def test_mono_render_is_copied_across_input_channels(monkeypatch, tmp_path):
    """A mono render (an older fork) still comes back at the input's shape."""
    script = (
        "#!/bin/sh\n"
        f'{sys.executable} -c "import soundfile as sf, sys; '
        "a, sr = sf.read(sys.argv[1], always_2d=True); "
        "sf.write(sys.argv[2], a[:, :1], sr, subtype='FLOAT')\" \"$2\" \"$3\"\n"
    )
    monkeypatch.setattr(baselines, "A2SB_DIR", fake_fork(tmp_path, script))
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, SR, dtype=np.float32),
            np.linspace(0.5, -0.5, SR, dtype=np.float32),
        ]
    )

    out = baselines.run_a2sb(stereo, SR)

    assert out.shape == stereo.shape
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[0], stereo[0])


def test_fork_sha_is_reported(monkeypatch, tmp_path, capsys):
    """Every render must be attributable to a fork commit."""
    monkeypatch.setattr(baselines, "A2SB_DIR", fake_fork(tmp_path, '#!/bin/sh\ncp "$2" "$3"\n'))

    baselines.run_a2sb(silence(seconds=0.1), SR)

    assert "fork @" in capsys.readouterr().out


def test_wrong_sample_rate_is_rejected_before_any_subprocess():
    audio = silence(seconds=0.1)

    with mock.patch.object(baselines, "subprocess", autospec=True) as sub, pytest.raises(
        ValueError, match="44100"
    ):
        baselines.run_a2sb(audio, 48_000)
    sub.run.assert_not_called()


def test_missing_fork_says_how_to_clone(monkeypatch, tmp_path):
    monkeypatch.setattr(baselines, "A2SB_DIR", tmp_path / "nowhere")

    with pytest.raises(FileNotFoundError, match="git clone"):
        baselines.run_a2sb(silence(seconds=0.1), SR)


def test_missing_env_says_how_to_build_it(monkeypatch, tmp_path):
    (tmp_path / "restore.py").write_text("")
    monkeypatch.setattr(baselines, "A2SB_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="setup.sh"):
        baselines.run_a2sb(silence(seconds=0.1), SR)


def test_subprocess_failure_surfaces_its_stderr(monkeypatch, tmp_path):
    script = "#!/bin/sh\necho the-model-exploded >&2\nexit 3\n"
    monkeypatch.setattr(baselines, "A2SB_DIR", fake_fork(tmp_path, script))

    with pytest.raises(RuntimeError, match="the-model-exploded"):
        baselines.run_a2sb(silence(seconds=0.1), SR)


def test_missing_output_is_a_failure_even_on_exit_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(baselines, "A2SB_DIR", fake_fork(tmp_path, "#!/bin/sh\nexit 0\n"))

    with pytest.raises(RuntimeError, match="A2SB failed"):
        baselines.run_a2sb(silence(seconds=0.1), SR)


def test_output_shorter_than_input_is_padded_back_with_zeros(monkeypatch, tmp_path):
    """A2SB writes ~408 samples short; the wrapper must restore the input
    length so outputs stay comparable sample for sample."""
    script = (
        "#!/bin/sh\n"
        f'{sys.executable} -c "import soundfile as sf, numpy as np, sys; '
        "a, sr = sf.read(sys.argv[1], always_2d=True); "
        'sf.write(sys.argv[2], a[:-408], sr)" "$2" "$3"\n'
    )
    monkeypatch.setattr(baselines, "A2SB_DIR", fake_fork(tmp_path, script))
    audio = silence(seconds=1.0)

    out = baselines.run_a2sb(audio, SR)

    assert out.shape[1] == audio.shape[1]
    np.testing.assert_array_equal(out[:, -408:], 0.0)
