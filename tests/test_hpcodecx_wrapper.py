"""The HP-codecX wrapper is a process boundary, like A2SB's and AudioSR's:
the failure modes that matter are environment-shaped (venv not built,
subprocess dying), and they must fail loudly with the fix in the message."""

import stat

import numpy as np
import pytest

from grooveback import baselines

SR = 44_100


def fake_venv_python(tmp_path, script):
    """A stand-in for the hpcodecx venv's python whose behaviour is `script`.

    The wrapper invokes `python scripts/predict.py --path ... --input IN
    --output OUT`, so the script sees the driver path as $1 and must parse
    the --input/--output flags to act on the pair directory.
    """
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(script)
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return python


COPY_PAIRS = """#!/bin/sh
while [ $# -gt 0 ]; do
    case "$1" in
    --input) IN="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    *) shift ;;
    esac
done
mkdir -p "$OUT"
cp "$IN"/*_48.wav "$OUT"/
"""
"""predict.py's observable contract: for every input pair, a 48 kHz output
file with the input's name — here the dummy 48 kHz twin copied through."""


def test_stereo_render_comes_back_unmixed(monkeypatch, tmp_path):
    """End-to-end plumbing through the subprocess seam: distinct channels
    must survive the pair layout and come back in order. The wrapper's own
    44.1->48->44.1 round trip is soxr VHQ, so equality holds to the same
    tolerance as the resample round-trip test in test_audio.py, edges
    excluded."""
    monkeypatch.setattr(
        baselines, "HPCODECX_VENV_PYTHON", fake_venv_python(tmp_path, COPY_PAIRS)
    )
    t = np.arange(SR, dtype=np.float32) / SR
    stereo = np.stack(
        [
            (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32),
            (0.5 * np.sin(2 * np.pi * 990.0 * t)).astype(np.float32),
        ]
    )

    out = baselines.run_hpcodecx(stereo, SR)

    assert out.shape == stereo.shape
    np.testing.assert_allclose(
        out[:, 1_000:-1_000], stereo[:, 1_000:-1_000], rtol=0, atol=1e-4
    )


def test_segmented_input_reconstructs_through_the_joins(monkeypatch, tmp_path):
    """The audio.md identity gate at the segmentation seam: with the model
    faked as identity (the dummy 48 kHz pair copied back), cutting a long
    input into segments and crossfading the joins must reconstruct it. A
    3 s input against 1 s segments forces two joins plus a partial last
    segment; tolerance as in the resample round-trip test, edges excluded."""
    monkeypatch.setattr(
        baselines, "HPCODECX_VENV_PYTHON", fake_venv_python(tmp_path, COPY_PAIRS)
    )
    t = np.arange(3 * SR, dtype=np.float32) / SR
    stereo = np.stack(
        [
            (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32),
            (0.5 * np.sin(2 * np.pi * 990.0 * t)).astype(np.float32),
        ]
    )

    out = baselines.run_hpcodecx(stereo, SR, segment_seconds=1.0)

    assert out.shape == stereo.shape
    np.testing.assert_allclose(
        out[:, 1_000:-1_000], stereo[:, 1_000:-1_000], rtol=0, atol=1e-4
    )


def test_missing_env_says_how_to_build_it(monkeypatch, tmp_path):
    monkeypatch.setattr(baselines, "HPCODECX_VENV_PYTHON", tmp_path / "nowhere")

    with pytest.raises(FileNotFoundError, match="hpcodecx_setup.sh"):
        baselines.run_hpcodecx(np.zeros((2, SR), dtype=np.float32), SR)


def test_subprocess_failure_surfaces_its_stderr(monkeypatch, tmp_path):
    script = "#!/bin/sh\necho the-model-exploded >&2\nexit 3\n"
    monkeypatch.setattr(
        baselines, "HPCODECX_VENV_PYTHON", fake_venv_python(tmp_path, script)
    )

    with pytest.raises(RuntimeError, match="the-model-exploded"):
        baselines.run_hpcodecx(np.zeros((2, SR), dtype=np.float32), SR)
