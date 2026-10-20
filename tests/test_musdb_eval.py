"""`evaluate` runs a model over MP3 twins of the test mixtures and scores it
with Apollo's metrics."""

import numpy as np
import pytest
import soundfile as sf
import torch

from grooveback.cli.musdb_eval import evaluate

SR = 44_100


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # run_apollo reads the target device off the first parameter.
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, audio):
        return audio


@pytest.fixture
def musdb_test(tmp_path):
    """Two tracks of 15 s white noise in MUSDB18-HQ's {track}/mixture.wav layout."""
    rng = np.random.default_rng(0)
    for track in ("a", "b"):
        (tmp_path / track).mkdir()
        noise = rng.uniform(-0.5, 0.5, size=(15 * SR, 2))
        sf.write(tmp_path / track / "mixture.wav", noise, SR, subtype="PCM_16")
    return tmp_path


def test_identity_model_scores_exactly_like_the_untouched_mp3(musdb_test):
    """The model must receive the MP3 and be scored against the clean session:
    fed the clean session instead, the identity would score infinitely high.
    A 15 s track holds 3 sessions of 6 s at a 3 s hop, so 2 tracks give 6."""
    try:
        import torchaudio

        torchaudio.functional.apply_codec(torch.zeros(2, SR), SR, format="mp3", compression=24)
    except (RuntimeError, OSError, ImportError) as error:
        pytest.skip(f"sox MP3 is unavailable: {error}")

    rows = evaluate(musdb_test, kbps=24, model=IdentityModel())

    assert len(rows) == 6
    assert [row["apollo_sdr"] for row in rows] == [row["input_sdr"] for row in rows]
    assert [row["apollo_si_snr"] for row in rows] == [row["input_si_snr"] for row in rows]
