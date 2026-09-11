"""The SonicMaster wrapper is a process boundary, like A2SB's and AudioSR's:
the failure modes that matter are environment-shaped (venv not built,
subprocess dying), and they must fail loudly with the fix in the message."""

import stat

import numpy as np
import pytest

from grooveback import baselines

SR = 44_100


def fake_venv_python(tmp_path, script):
    """A stand-in for the sonicmaster venv's python whose behaviour is
    `script`. The wrapper invokes `python infer_single.py --ckpt weights
    --input IN --prompt P --output OUT`, so the script parses the flags."""
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(script)
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return python


COPY_THROUGH = """#!/bin/sh
while [ $# -gt 0 ]; do
    case "$1" in
    --input) IN="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    *) shift ;;
    esac
done
cp "$IN" "$OUT"
"""


def test_stereo_render_comes_back_unmixed(monkeypatch, tmp_path):
    """End-to-end plumbing through the subprocess seam: the model is natively
    44.1 kHz stereo and nothing resamples, so a copy-through fake gives exact
    equality."""
    monkeypatch.setattr(
        baselines, "SONICMASTER_VENV_PYTHON", fake_venv_python(tmp_path, COPY_THROUGH)
    )
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, SR, dtype=np.float32),
            np.linspace(0.5, -0.5, SR, dtype=np.float32),
        ]
    )

    out = baselines.run_sonicmaster(stereo, SR)

    assert out.shape == stereo.shape
    np.testing.assert_array_equal(out, stereo)


def test_wrong_sample_rate_is_rejected_before_any_subprocess():
    from unittest import mock

    with mock.patch.object(baselines, "subprocess", autospec=True) as sub, pytest.raises(
        ValueError, match="44100"
    ):
        baselines.run_sonicmaster(np.zeros((2, SR), dtype=np.float32), 48_000)
    sub.run.assert_not_called()


def test_missing_env_says_how_to_build_it(monkeypatch, tmp_path):
    monkeypatch.setattr(baselines, "SONICMASTER_VENV_PYTHON", tmp_path / "nowhere")

    with pytest.raises(FileNotFoundError, match="sonicmaster_setup.sh"):
        baselines.run_sonicmaster(np.zeros((2, SR), dtype=np.float32), SR)


def test_subprocess_failure_surfaces_its_stderr(monkeypatch, tmp_path):
    script = "#!/bin/sh\necho the-model-exploded >&2\nexit 3\n"
    monkeypatch.setattr(
        baselines, "SONICMASTER_VENV_PYTHON", fake_venv_python(tmp_path, script)
    )

    with pytest.raises(RuntimeError, match="the-model-exploded"):
        baselines.run_sonicmaster(np.zeros((2, SR), dtype=np.float32), SR)
