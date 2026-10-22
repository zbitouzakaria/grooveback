"""The AudioSR wrapper is a process boundary, like A2SB's: the failure modes
that matter are environment-shaped (venv not built, subprocess dying), and
they must fail loudly with the fix in the message."""

import stat

import numpy as np
import pytest

from grooveback import baselines

SR = 44_100


def fake_venv_python(tmp_path, script):
    """A stand-in for the audiosr venv's python whose behaviour is `script`.

    The wrapper invokes `python restore.py in.wav out.wav ...`, so in the
    script the driver path is $1, the input $2 and the output $3.
    """
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(script)
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return python


def test_stereo_render_comes_back_unmixed(monkeypatch, tmp_path):
    """End-to-end plumbing through the subprocess seam: distinct channels
    must survive the temp-wav round trip exactly."""
    monkeypatch.setattr(
        baselines,
        "AUDIOSR_VENV_PYTHON",
        fake_venv_python(tmp_path, '#!/bin/sh\ncp "$2" "$3"\n'),
    )
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, SR, dtype=np.float32),
            np.linspace(0.5, -0.5, SR, dtype=np.float32),
        ]
    )

    out = baselines.run_audiosr(stereo, SR)

    assert out.shape == stereo.shape
    np.testing.assert_array_equal(out, stereo)


def test_missing_env_says_how_to_build_it(monkeypatch, tmp_path):
    monkeypatch.setattr(baselines, "AUDIOSR_VENV_PYTHON", tmp_path / "nowhere")

    with pytest.raises(FileNotFoundError, match="audiosr_setup.sh"):
        baselines.run_audiosr(np.zeros((2, SR), dtype=np.float32), SR)


def test_subprocess_failure_surfaces_its_stderr(monkeypatch, tmp_path):
    script = "#!/bin/sh\necho the-model-exploded >&2\nexit 3\n"
    monkeypatch.setattr(
        baselines, "AUDIOSR_VENV_PYTHON", fake_venv_python(tmp_path, script)
    )

    with pytest.raises(RuntimeError, match="the-model-exploded"):
        baselines.run_audiosr(np.zeros((2, SR), dtype=np.float32), SR)
