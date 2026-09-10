"""The autoencoder registry's wiring and the damage arithmetic.

Real autoencoders need checkpoints and a GPU, so identity and recording fakes
stand in; the contract under test is ours, not the model authors'. The damage
functions are pure numpy and get hand-computed oracles.
"""

import numpy as np
import pytest
import torch

from grooveback import latents as gl

SR = 44_100


class IdentityAE:
    """Stands in for a SAME autoencoder: encode_audio/decode_audio return the
    batch unchanged, so decode(encode(x)) must return x exactly — any
    transpose, dtype, or squeeze mistake in the wiring breaks equality."""

    def __init__(self):
        self.calls = []

    def parameters(self):
        yield torch.zeros(1)

    def encode_audio(self, batch):
        self.calls.append(("encode", tuple(batch.shape)))
        return batch

    def decode_audio(self, batch):
        self.calls.append(("decode", tuple(batch.shape)))
        return batch


def test_roundtrip_through_an_identity_model_returns_the_input():
    model = IdentityAE()
    audio = np.stack([np.linspace(0, 1, 500), np.linspace(0, -1, 500)]).astype(np.float32)

    out = gl.roundtrip(audio, SR, model)

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, audio)


def test_encode_rejects_wrong_sample_rate_before_the_model_runs():
    model = IdentityAE()
    audio = np.zeros((2, 500), dtype=np.float32)

    with pytest.raises(ValueError, match="44100"):
        gl.encode(audio, 48_000, model)
    assert model.calls == []


def test_encode_rejects_non_finite_audio_before_the_model_runs():
    model = IdentityAE()
    audio = np.full((2, 500), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN"):
        gl.encode(audio, SR, model)
    assert model.calls == []


def test_decode_rejects_non_finite_latents_before_the_model_runs():
    model = IdentityAE()
    latents = np.full((256, 4), np.inf, dtype=np.float32)

    with pytest.raises(ValueError, match="Latents"):
        gl.decode(latents, model)
    assert model.calls == []


def test_decode_rejects_a_model_that_emits_non_finite_audio():
    """Subtracted latents sit outside the training distribution; a decoder
    blowing up must fail here, not poison loudness matching downstream."""

    class NaNDecoder(IdentityAE):
        def decode_audio(self, batch):
            return torch.full_like(batch, torch.nan)

    with pytest.raises(ValueError, match="Decoded audio"):
        gl.decode(np.zeros((256, 4), dtype=np.float32), NaNDecoder())


def test_load_ae_rejects_unknown_names():
    with pytest.raises(ValueError, match="autoencoder"):
        gl.load_ae("same-xl")


def test_mean_damage_is_the_frame_mean_of_the_latent_difference():
    clean = np.zeros((2, 2), dtype=np.float32)
    degraded = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

    damage = gl.mean_damage(clean, degraded)

    np.testing.assert_array_equal(damage, np.array([2.0, 3.0], dtype=np.float32))


def test_mean_damage_tolerates_a_boundary_frame_overrun():
    """Codecs round lengths up to whole frames, so the two encodings of the
    same donor can differ by a frame or two at the edge; the mean uses the
    shared frames."""
    clean = np.zeros((2, 4), dtype=np.float32)
    degraded = np.ones((2, 5), dtype=np.float32)

    damage = gl.mean_damage(clean, degraded)

    np.testing.assert_array_equal(damage, np.ones(2, dtype=np.float32))


def test_mean_damage_refuses_a_real_frame_grid_mismatch():
    clean = np.zeros((2, 10), dtype=np.float32)
    degraded = np.zeros((2, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="frame counts differ"):
        gl.mean_damage(clean, degraded)


def test_mean_damage_refuses_latents_from_different_autoencoders():
    clean = np.zeros((256, 10), dtype=np.float32)
    degraded = np.zeros((128, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="channel counts differ"):
        gl.mean_damage(clean, degraded)


def test_subtract_damage_with_a_zero_vector_changes_nothing():
    latents = np.arange(8, dtype=np.float32).reshape(2, 4)

    out = gl.subtract_damage(latents, np.zeros(2, dtype=np.float32))

    np.testing.assert_array_equal(out, latents)


def test_subtract_damage_subtracts_the_vector_from_every_frame():
    latents = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    damage = np.array([0.25, 0.5], dtype=np.float32)

    out = gl.subtract_damage(latents, damage)

    np.testing.assert_array_equal(
        out, np.array([[0.75, 0.75], [1.5, 1.5]], dtype=np.float32)
    )


def test_subtract_damage_refuses_a_vector_from_another_autoencoder():
    latents = np.zeros((256, 4), dtype=np.float32)
    damage = np.zeros(128, dtype=np.float32)

    with pytest.raises(ValueError, match="different autoencoder"):
        gl.subtract_damage(latents, damage)


def test_subtract_damage_refuses_a_non_vector_damage():
    latents = np.zeros((2, 4), dtype=np.float32)
    damage = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="one vector"):
        gl.subtract_damage(latents, damage)
