"""The prior wrapper's wiring: what reaches the model, what comes back.

The real model needs gated checkpoints and a GPU, so a recording fake stands
in; the contract under test is ours, not Stability's.
"""

import numpy as np
import pytest
import torch

from grooveback.priors import PRIOR_VARIANTS, generate, load_prior


class RecordingPrior:
    """Records every generate() call; returns a fixed stereo batch whose two
    channels are distinguishable."""

    def __init__(self, samples: int = 1_000):
        self.calls = []
        self._samples = samples

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch = torch.zeros((1, 2, self._samples))
        batch[0, 0] = 0.25
        batch[0, 1] = -0.5
        return batch


def test_generate_returns_channels_by_samples_float32():
    model = RecordingPrior(samples=1_000)

    out = generate(model, "a prompt", seconds=12.0, seed=0)

    assert out.shape == (2, 1_000)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], 0.25)
    np.testing.assert_array_equal(out[1], -0.5)


def test_generate_maps_seconds_to_duration_and_passes_sampling_through():
    model = RecordingPrior()

    generate(model, "a prompt", seconds=12.0, seed=7, steps=50, cfg_scale=7.0)

    assert model.calls == [
        {"prompt": "a prompt", "duration": 12.0, "seed": 7,
         "steps": 50, "cfg_scale": 7.0}
    ]


def test_variant_table_pins_the_released_model_types():
    """The six repo names and the two sampling families are public contract:
    post-trained models sample in 8 unguided steps, -base models in 50 steps
    at cfg 7 (stable-audio-3's inference guide)."""
    assert sorted(PRIOR_VARIANTS) == [
        "medium", "medium-base",
        "small-music", "small-music-base",
        "small-sfx", "small-sfx-base",
    ]
    for name, sampling in PRIOR_VARIANTS.items():
        if name.endswith("-base"):
            assert sampling == {"steps": 50, "cfg_scale": 7.0}
        else:
            assert sampling == {"steps": 8, "cfg_scale": 1.0}


def test_unknown_variant_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="variant"):
        load_prior("small-jazz")
