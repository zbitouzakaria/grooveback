"""The SDEdit solver's wiring: what reaches the model, what comes back.

The real prior needs gated checkpoints and a GPU, so a recording fake stands
in; the contract under test is ours, not Stability's.
"""

import numpy as np
import pytest
import torch

from grooveback.solvers import SDEDIT_MAX_SECONDS, sdedit

SR = 44_100


class RecordingPrior:
    """Records every generate() call; returns a fixed stereo batch whose two
    channels are distinguishable."""

    def __init__(self, samples: int = 2_000):
        self.calls = []
        self._samples = samples

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch = torch.zeros((1, 2, self._samples))
        batch[0, 0] = 0.25
        batch[0, 1] = -0.5
        return batch


def test_sdedit_returns_channels_by_samples_float32_trimmed_to_input():
    model = RecordingPrior(samples=2_000)
    audio = np.zeros((2, 1_500), dtype=np.float32)

    out = sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)

    assert out.shape == (2, 1_500)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], 0.25)
    np.testing.assert_array_equal(out[1], -0.5)


def test_sdedit_passes_input_as_channels_by_samples_tensor():
    """stable-audio-3 reads 2-D *numpy* init audio as (samples, channels) and
    transposes it, so the solver must hand over a torch tensor, which passes
    through untransposed."""
    model = RecordingPrior()
    audio = np.arange(6, dtype=np.float32).reshape(2, 3)

    sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)

    in_sr, tensor = model.calls[0]["init_audio"]
    assert in_sr == SR
    assert isinstance(tensor, torch.Tensor)
    torch.testing.assert_close(
        tensor, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    )


def test_sdedit_maps_noise_level_and_sampling_settings_onto_generate():
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    sdedit(
        model, audio, SR,
        noise_level=0.3, steps=50, cfg_scale=7.0, prompt="a prompt", seed=7,
    )

    call = model.calls[0]
    assert call["init_noise_level"] == 0.3
    assert call["steps"] == 50
    assert call["cfg_scale"] == 7.0
    assert call["prompt"] == "a prompt"
    assert call["seed"] == 7


def test_sdedit_requests_the_inputs_full_length_past_the_default_clamp():
    """generate()'s sample_size defaults to 120 s of samples and silently
    truncates anything longer (ADR-0008), so on a longer input the solver
    must ask for the input's own length plus the schedule's 6 s padding —
    and a duration that covers every input sample."""
    model = RecordingPrior()
    n = 200 * SR  # 200 s, past the 120 s default clamp
    audio = np.zeros((2, n), dtype=np.float32)

    sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)

    call = model.calls[0]
    assert call["sample_size"] >= n + 6 * SR
    assert call["duration"] * SR >= n


@pytest.mark.parametrize("noise_level", [0.0, -0.2, 1.5])
def test_noise_level_outside_unit_interval_is_rejected_before_generation(noise_level):
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="noise_level"):
        sdedit(model, audio, SR, noise_level=noise_level, steps=8, cfg_scale=1.0)
    assert model.calls == []


@pytest.mark.parametrize("steps", [0, -8])
def test_non_positive_steps_is_rejected_before_generation(steps):
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="steps"):
        sdedit(model, audio, SR, noise_level=0.5, steps=steps, cfg_scale=1.0)
    assert model.calls == []


def test_wrong_sample_rate_is_rejected_before_generation():
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="44100 Hz"):
        sdedit(model, audio, 48_000, noise_level=0.5, steps=8, cfg_scale=1.0)
    assert model.calls == []


def test_non_finite_input_is_rejected_before_generation():
    model = RecordingPrior()
    audio = np.full((2, 1_000), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN"):
        sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)
    assert model.calls == []


def test_input_longer_than_the_cap_is_rejected_before_generation():
    model = RecordingPrior()
    n = int(SDEDIT_MAX_SECONDS * SR) + SR  # one second past the cap

    with pytest.raises(ValueError, match="whole files up to"):
        sdedit(
            model, np.zeros((2, n), dtype=np.float32), SR,
            noise_level=0.5, steps=8, cfg_scale=1.0,
        )
    assert model.calls == []
