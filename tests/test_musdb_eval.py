"""`evaluate` runs a model over MP3 twins of the test mixtures and scores it
with Apollo's metrics and this project's spectral ones."""

import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from grooveback.cli.musdb_eval import clip_starts, codec_simu, evaluate, main

SR = 44_100


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # run_apollo reads the target device off the first parameter.
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, audio):
        return audio


class ToneModel(torch.nn.Module):
    """Adds a 1 kHz tone, so its output scores differently from its input."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, audio):
        t = torch.arange(audio.shape[-1]) / SR
        return audio + 0.1 * torch.sin(2 * torch.pi * 1_000 * t)


class IdentityEarVAE(torch.nn.Module):
    """Stands in for εar-VAE: its latents are the audio itself."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode(self, audio, use_sample):
        return audio

    def decode(self, latents):
        return latents


@pytest.fixture(scope="module")
def long_musdb_test(tmp_path_factory):
    """Four tracks of 60 s white noise: 18 clips each, 72 in all, so a broken
    draw cannot pass the draw tests by chance."""
    root = tmp_path_factory.mktemp("long") / "test"
    rng = np.random.default_rng(0)
    for track in ("a", "b", "c", "d"):
        (root / track).mkdir(parents=True)
        noise = rng.uniform(-0.5, 0.5, size=(60 * SR, 2))
        sf.write(root / track / "mixture.wav", noise, SR, subtype="PCM_16")
    return root


@pytest.fixture
def musdb_test(tmp_path):
    """Two tracks of 15 s white noise in MUSDB18-HQ's {track}/mixture.wav layout."""
    rng = np.random.default_rng(0)
    for track in ("a", "b"):
        (tmp_path / "test" / track).mkdir(parents=True)
        noise = rng.uniform(-0.5, 0.5, size=(15 * SR, 2))
        sf.write(tmp_path / "test" / track / "mixture.wav", noise, SR, subtype="PCM_16")
    return tmp_path / "test"


def test_identity_model_scores_exactly_like_the_untouched_mp3(musdb_test):
    """The model must receive the MP3 and be scored against the clean clip:
    fed the clean clip instead, the identity would score infinitely high.
    A 15 s track holds 3 clips of 6 s at a 3 s hop, so 2 tracks give 6."""
    try:
        import torchaudio

        torchaudio.functional.apply_codec(torch.zeros(2, SR), SR, format="mp3", compression=24)
    except (RuntimeError, OSError, ImportError) as error:
        pytest.skip(f"sox MP3 is unavailable: {error}")

    rows = evaluate(musdb_test, kbps=24, model=IdentityModel())

    assert len(rows) == 6
    assert [row["apollo_sdr"] for row in rows] == [row["input_sdr"] for row in rows]
    assert [row["apollo_si_snr"] for row in rows] == [row["input_si_snr"] for row in rows]
    assert [row["apollo_spectral_snr"] for row in rows] == [
        row["input_spectral_snr"] for row in rows
    ]
    assert [row["apollo_lsd"] for row in rows] == [row["input_lsd"] for row in rows]


def test_identity_earvae_scores_exactly_like_the_model_it_follows(musdb_test):
    """The εar-VAE round trip must receive the model's output, not the MP3: the
    tone the model adds makes the two score differently, and an identity
    round trip then scores exactly like the model."""
    try:
        import torchaudio

        torchaudio.functional.apply_codec(torch.zeros(2, SR), SR, format="mp3", compression=24)
    except (RuntimeError, OSError, ImportError) as error:
        pytest.skip(f"sox MP3 is unavailable: {error}")

    rows = evaluate(
        musdb_test, kbps=24, model=ToneModel(), autoencoders={"earvae": IdentityEarVAE()}
    )

    assert [row["apollo_earvae_sdr"] for row in rows] == [row["apollo_sdr"] for row in rows]
    assert [row["apollo_earvae_lsd"] for row in rows] == [row["apollo_lsd"] for row in rows]
    assert [row["apollo_sdr"] for row in rows] != [row["input_sdr"] for row in rows]


def test_identity_earvae_alone_scores_exactly_like_the_untouched_mp3(musdb_test):
    """Without Apollo, the round trip must receive the MP3 itself."""
    try:
        import torchaudio

        torchaudio.functional.apply_codec(torch.zeros(2, SR), SR, format="mp3", compression=24)
    except (RuntimeError, OSError, ImportError) as error:
        pytest.skip(f"sox MP3 is unavailable: {error}")

    rows = evaluate(musdb_test, kbps=24, autoencoders={"earvae": IdentityEarVAE()})

    assert [row["earvae_sdr"] for row in rows] == [row["input_sdr"] for row in rows]
    assert [row["earvae_lsd"] for row in rows] == [row["input_lsd"] for row in rows]


def test_cli_scores_copies_of_the_mp3s_it_wrote_exactly_like_the_mp3s(musdb_test, tmp_path):
    """The two calls around a GPU render, end to end: the first draws 4 of the
    6 clips and writes their MP3s, the second scores renders that copy those
    MP3s. Both calls must draw the same 4 clips, and each render must be read
    from the file its MP3 was written as and scored against its clean clip."""
    try:
        import torchaudio

        torchaudio.functional.apply_codec(torch.zeros(2, SR), SR, format="mp3", compression=24)
    except (RuntimeError, OSError, ImportError) as error:
        pytest.skip(f"sox MP3 is unavailable: {error}")
    main(
        [str(musdb_test), str(tmp_path / "mp3.csv"), "--kbps", "24", "--no-apollo",
         "--clips", "4", "--mp3-dir", str(tmp_path / "mp3")]
    )  # fmt: skip

    main(
        [str(musdb_test), str(tmp_path / "scores.csv"), "--kbps", "24", "--no-apollo",
         "--clips", "4", "--render", f"copy={tmp_path / 'mp3'}"]
    )  # fmt: skip

    with (tmp_path / "scores.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert [row["copy_sdr"] for row in rows] == [row["input_sdr"] for row in rows]
    assert [row["copy_lsd"] for row in rows] == [row["input_lsd"] for row in rows]


def test_clip_draw_picks_distinct_clips(long_musdb_test):
    every = clip_starts(long_musdb_test)

    drawn = clip_starts(long_musdb_test, clips=10)

    assert len(set(drawn)) == 10
    assert set(drawn) <= set(every)


def test_clip_draw_samples_across_tracks(long_musdb_test):
    """The first 10 clips in track order all come from track a, which holds 18."""
    drawn = clip_starts(long_musdb_test, clips=10)

    assert len({path.parent.name for path, _ in drawn}) > 1


def test_clip_draw_is_the_same_on_every_call(long_musdb_test):
    """Every bitrate is scored in its own call, so they score the same
    clips only if the draw repeats."""
    first = clip_starts(long_musdb_test, clips=10)

    second = clip_starts(long_musdb_test, clips=10)

    assert first == second


def test_smaller_clip_draw_is_part_of_a_larger_one(long_musdb_test):
    """A run on more clips then only renders the clips the smaller run did not.
    Two independent draws of 10 and 20 of the 72 clips would nest by chance
    with a probability of about 3e-7."""
    larger = clip_starts(long_musdb_test, clips=20)

    smaller = clip_starts(long_musdb_test, clips=10)

    assert set(smaller) <= set(larger)


def test_codec_simu_aligns_the_mp3_of_a_clip_whose_edges_jump():
    """Regression: Apollo's `match2` correlated the untapered clips, so the jump
    from a clip's last sample back to its first sat at lag 0 in both the clean
    clip and the MP3. Above the MP3's cut-off it outweighed the true delay, and
    the MP3 stayed 1,105 samples late. The fixture needs both parts: noise below
    2 kHz leaves the band above the cut-off to the jump alone, and a ramp from
    -0.5 to +0.5 makes the jump loud. The noise alone does not trigger it."""
    try:
        import torchaudio

        torchaudio.functional.apply_codec(torch.zeros(2, SR), SR, format="mp3", compression=24)
    except (RuntimeError, OSError, ImportError) as error:
        pytest.skip(f"sox MP3 is unavailable: {error}")

    rng = np.random.default_rng(0)
    spectrum = np.fft.rfft(rng.standard_normal((2, 6 * SR)), axis=-1)
    spectrum[:, 2_000 * 6 :] = 0  # over 6 s, bin k sits at k / 6 Hz
    noise = np.fft.irfft(spectrum, n=6 * SR, axis=-1)
    clip = (0.3 * noise / np.abs(noise).max() + np.linspace(-0.5, 0.5, 6 * SR)).astype(np.float32)

    codec = codec_simu(clip, kbps=24)

    # Circular cross-correlation of the channel sums peaks at index 0 when aligned, 1,105 when late.
    spectra = np.fft.rfft(codec.sum(axis=0)) * np.fft.rfft(clip.sum(axis=0)).conj()
    xcorr = np.fft.irfft(spectra, n=6 * SR)
    assert int(np.argmax(xcorr)) == 0
