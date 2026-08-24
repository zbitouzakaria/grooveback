"""Chunking belongs to the vendored fork and is tested there; what is ours in
`run_apollo` is the plumbing around it — unit conversion, tensor wrapping, the
rotary-limit guard. An identity model through `run_apollo` covers exactly that
seam: it fails on a fork parameter rename or a wrong conversion, which nothing
else does."""

import numpy as np
import pytest
import torch

from grooveback.baselines import APOLLO_MAX_SECONDS, APOLLO_SAMPLE_RATE, run_apollo

SR = APOLLO_SAMPLE_RATE


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # run_apollo reads the target device off the first parameter.
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, audio):
        return audio


def test_identity_model_returns_the_input():
    """The ADR-0002 identity gate, kept at grooveback's seam.

    Chunk, overlap and pad are shorter than the 3 s input so the chunked path
    runs. atol covers float32 crossfade arithmetic on identical values.
    """
    rng = np.random.default_rng(0)
    audio = rng.uniform(-1.0, 1.0, size=(2, 3 * SR)).astype(np.float32)

    out = run_apollo(
        audio,
        SR,
        model=IdentityModel(),
        chunk_seconds=1.0,
        overlap_seconds=0.2,
        chunk_pad_seconds=0.1,
    )

    assert out.shape == audio.shape
    np.testing.assert_allclose(out, audio, rtol=0, atol=1e-6)


def test_apollo_rejects_input_past_the_rotary_limit():
    """Apollo's rotary tables cover a fixed span; past it the model dies on a
    shape mismatch deep inside a forward pass. Fail early with the reason —
    and the padded chunk, not the bare chunk, is what must fit."""
    too_long = np.zeros((2, int((APOLLO_MAX_SECONDS + 1) * SR)), np.float32)
    with pytest.raises(ValueError, match="rotary embeddings"):
        run_apollo(too_long, SR, chunk_seconds=None)

    short = np.zeros((2, SR), np.float32)
    with pytest.raises(ValueError, match="rotary embeddings"):
        run_apollo(
            short, SR, chunk_seconds=APOLLO_MAX_SECONDS - 1.0, chunk_pad_seconds=1.0
        )
