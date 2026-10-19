"""The rebuilt Apollo test suite: `build` writes their clean/MP3 pairs, `score`
runs a model over them with their metrics."""

import numpy as np
import pytest
import soundfile as sf
import torch

from grooveback import audio as ga
from grooveback.cli.musdb_eval import build, score
from grooveback.evaluation import best_lag

SR = 44_100


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # run_apollo reads the target device off the first parameter.
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, audio):
        return audio


def tone(freq: float, amplitude: float, seconds: float = 6.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def musdb_test(tmp_path):
    """Two tracks of 15 s white-noise stems in MUSDB18-HQ's {track}/{stem}.wav layout."""
    rng = np.random.default_rng(0)
    for track in ("a", "b"):
        (tmp_path / "musdb" / track).mkdir(parents=True)
        for stem in ("bass", "drums", "other", "vocals"):
            noise = rng.uniform(-0.1, 0.1, size=(15 * SR, 2))
            sf.write(tmp_path / "musdb" / track / f"{stem}.wav", noise, SR, subtype="PCM_16")
    return tmp_path / "musdb"


def test_score_averages_the_channels_in_db(tmp_path):
    """The identity model returns the input, whose left channel carries an
    error tone 20 dB under its reference and right one 40 dB under. Apollo's
    metrics score each channel alone and average the dB: (20 + 40) / 2 = 30.
    Pooling the channels first would read 10·log10(2 / (0.01 + 0.0001)) =
    23 dB. Every tone completes whole cycles in 6 s, so they are orthogonal;
    abs=0.5 on SDR covers its 512-tap filter's regularization."""
    ori = np.stack([tone(100.0, amplitude=1.0), tone(150.0, amplitude=1.0)])
    codec = ori + np.stack([tone(200.0, amplitude=0.1), tone(200.0, amplitude=0.01)])
    ga.save(tmp_path / "vocal" / "0000" / "ori_wav.wav", ori, SR)
    ga.save(tmp_path / "vocal" / "0000" / "codec_wav.wav", codec, SR)

    rows = score(tmp_path, model=IdentityModel())

    assert rows[0]["apollo_si_snr"] == pytest.approx(30.0, abs=0.05)
    assert rows[0]["apollo_sdr"] == pytest.approx(30.0, abs=0.5)


def test_built_pairs_are_aligned_within_one_sample(tmp_path, musdb_test):
    """The decoded MP3 starts 1,105 samples late (LAME's 576-sample encoder
    delay plus the decoder's 529, measured), and their match2 roll must undo
    it in every category. Within one sample, not exactly: their
    phase-transform correlation and best_lag's plain one can round the
    codec's delay to adjacent samples — the authors' own 24 kbps demo pair
    `sample2` sits one sample off."""
    try:
        import torchaudio

        torchaudio.functional.apply_codec(torch.zeros(2, SR), SR, format="mp3", compression=24)
    except (RuntimeError, OSError, ImportError) as error:
        pytest.skip(f"sox MP3 is unavailable: {error}")

    build(musdb_test, tmp_path / "set", kbps=24, per_category=1)

    pairs = sorted((tmp_path / "set").glob("*/*/ori_wav.wav"))
    lags = [best_lag(ga.load(p)[0], ga.load(p.with_name("codec_wav.wav"))[0]) for p in pairs]
    assert len(lags) == 4
    assert set(lags) <= {-1, 0, 1}
