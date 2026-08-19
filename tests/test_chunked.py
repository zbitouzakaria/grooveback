"""Chunked Apollo inference, as delegated to the vendored fork.

Chunking lives in the fork's `inference.run_model` (chunk padding). These
tests drive it through the same import path `run_apollo` uses, so a submodule
checked out to a version without the fix fails loudly here instead of
silently reintroducing seam artifacts.
"""

import numpy as np
import pytest
import torch

from grooveback.baselines import (
    APOLLO_MAX_SECONDS,
    APOLLO_SAMPLE_RATE,
    _apollo_inference,
    run_apollo,
)

SR = APOLLO_SAMPLE_RATE


class IdentityModel(torch.nn.Module):
    def forward(self, audio):
        return audio


class NonlinearModel(torch.nn.Module):
    """Invents content deterministically from its input, like Apollo does.

    Apollo is safe to crossfade because two chunks looking at the same music
    produce the same thing wherever they are well inside their window — the
    fork's padding guarantees only well-inside output is used. A pointwise
    nonlinearity has that property in its purest form.

    A model that invents content differently on each call — a diffusion
    sampler, for instance — does NOT have it, and overlap crossfading loses
    power on such models no matter the padding. Revisit before chunking
    anything stochastic.
    """

    def forward(self, audio):
        return 0.5 * torch.sin(50.0 * audio)


def run_chunked(model, audio, chunk_s, overlap_s, pad_s):
    inference = _apollo_inference()
    with torch.inference_mode():
        out = inference.run_model(
            model,
            audio,
            torch.device("cpu"),
            chunk_samples=int(chunk_s * SR),
            overlap_samples=int(overlap_s * SR),
            chunk_pad_samples=int(pad_s * SR),
        )
    return out


def test_identity_model_passes_through():
    rng = np.random.default_rng(0)
    audio = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(1, 2, 3 * SR)).astype(np.float32)
    )
    out = run_chunked(IdentityModel(), audio, chunk_s=1.0, overlap_s=0.2, pad_s=0.1)
    assert out.shape == audio.shape
    assert torch.allclose(out, audio, atol=1e-5)


def test_chunked_matches_unchunked_for_a_consistent_model():
    """Chunked output must equal a single full pass, everywhere including
    the seams, for any model whose invention is consistent given its input."""
    rng = np.random.default_rng(1)
    audio = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(1, 2, 5 * SR)).astype(np.float32)
    )
    model = NonlinearModel()
    chunked = run_chunked(model, audio, chunk_s=1.0, overlap_s=0.2, pad_s=0.2)
    with torch.inference_mode():
        full = model(audio)
    assert torch.allclose(chunked, full, atol=1e-5)


def test_apollo_rejects_input_past_the_rotary_limit():
    """Apollo's rotary tables cover a fixed span; past it the model dies on a
    shape mismatch deep inside a forward pass. Fail early with the reason —
    and the padded chunk, not the bare chunk, is what must fit."""
    too_long = np.zeros((2, int((APOLLO_MAX_SECONDS + 1) * SR)), np.float32)
    with pytest.raises(ValueError, match="rotary embeddings"):
        run_apollo(too_long, SR, chunk_seconds=None)

    short = np.zeros((2, SR), np.float32)
    with pytest.raises(ValueError, match="rotary embeddings"):
        run_apollo(short, SR, chunk_seconds=APOLLO_MAX_SECONDS - 1.0,
                   chunk_pad_seconds=1.0)
