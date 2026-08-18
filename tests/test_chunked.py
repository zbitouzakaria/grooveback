"""Chunked overlap-add must be transparent when the model is a no-op.

Chunking is where audio code breaks silently: an off-by-one at a boundary or an
unnormalized crossfade produces output that plays fine and is quietly wrong.
An identity model turns that whole class of bug into a failing assert.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from grooveback.baselines import chunked

IDENTITY = lambda batch: batch  # noqa: E731


def signal(samples: int, channels: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(1, channels, samples)).astype(np.float32)
    )


@given(
    total=st.integers(min_value=1, max_value=5000),
    chunk=st.integers(min_value=1, max_value=1000),
    overlap_frac=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=200, deadline=None)
def test_identity_roundtrip(total, chunk, overlap_frac):
    """Any chunk and overlap combination returns the input untouched."""
    audio = signal(total)
    out = chunked(IDENTITY, audio, chunk, int(chunk * overlap_frac))
    assert out.shape == audio.shape
    assert torch.allclose(out, audio, atol=1e-5)


@given(
    total=st.integers(min_value=1, max_value=5000),
    chunk=st.integers(min_value=2, max_value=1000),
    context_frac=st.floats(min_value=0.0, max_value=0.49),
    join=st.integers(min_value=0, max_value=500),
)
@settings(max_examples=200, deadline=None)
def test_identity_roundtrip_discard(total, chunk, context_frac, join):
    """Discard mode is also transparent for an identity model."""
    audio = signal(total)
    context = min(int(chunk * context_frac), (chunk - 1) // 2)
    out = chunked(IDENTITY, audio, chunk, context, mode="discard", join_samples=join)
    assert out.shape == audio.shape
    assert torch.allclose(out, audio, atol=1e-5)


def test_length_preserved_when_not_divisible():
    """The final short chunk is padded for the model and cropped afterwards."""
    audio = signal(44_100 + 1)
    out = chunked(IDENTITY, audio, 10_000, 1_000)
    assert out.shape[-1] == 44_101


def test_no_chunking_paths_agree():
    """Chunking off, and a chunk longer than the input, both bypass overlap-add."""
    audio = signal(1000)
    assert torch.equal(chunked(IDENTITY, audio, None), audio)
    assert torch.equal(chunked(IDENTITY, audio, 5000, 100), audio)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_batching_does_not_change_output(batch_size):
    """Batching is a throughput knob, not a signal path."""
    audio = signal(20_000)
    out = chunked(IDENTITY, audio, 3_000, 500, batch_size=batch_size)
    assert torch.allclose(out, audio, atol=1e-5)


def test_crossfade_does_not_change_level():
    """A constant signal stays constant across boundaries.

    An unnormalized crossfade dips at every seam. On music that reads as a faint
    tremolo, which is easy to miss and hard to trace back.
    """
    audio = torch.full((1, 2, 30_000), 0.5)
    out = chunked(IDENTITY, audio, 4_000, 1_000)
    assert torch.allclose(out, audio, atol=1e-6)


def test_overlap_larger_than_half_chunk_rejected():
    with pytest.raises(ValueError, match="half the chunk"):
        chunked(IDENTITY, signal(10_000), 1_000, 600)


def test_discard_context_consuming_whole_chunk_rejected():
    """context*2 == chunk leaves no core to keep."""
    with pytest.raises(ValueError, match="half the chunk"):
        chunked(IDENTITY, signal(10_000), 1_000, 500, mode="discard")


def test_discard_level_constant():
    audio = torch.full((1, 2, 30_000), 0.5)
    out = chunked(IDENTITY, audio, 4_000, 1_000, mode="discard")
    assert torch.allclose(out, audio, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_discard_batching_does_not_change_output(batch_size):
    audio = signal(20_000)
    out = chunked(IDENTITY, audio, 3_000, 500, batch_size=batch_size, mode="discard")
    assert torch.allclose(out, audio, atol=1e-5)


def test_discard_single_realisation_outside_joins():
    """Away from the ~join-width seams, output equals one chunk's estimate
    exactly — no blending — which is the property the mode exists for."""
    audio = signal(20_000)
    out = chunked(lambda b: b * 0.5, audio, 3_000, 500, mode="discard", join_samples=100)
    assert torch.allclose(out, audio * 0.5, atol=1e-6)


def test_gain_model_is_applied_uniformly():
    """A non-identity model still comes back with no seams."""
    audio = torch.full((1, 2, 30_000), 0.4)
    out = chunked(lambda b: b * 0.5, audio, 4_000, 1_000)
    assert torch.allclose(out, audio * 0.5, atol=1e-6)


def test_apollo_rejects_input_past_the_rotary_limit():
    """Apollo's rotary tables cover a fixed span; past it the model dies on a
    shape mismatch deep inside a forward pass. Fail early with the reason."""
    from grooveback.baselines import APOLLO_MAX_SECONDS, APOLLO_SAMPLE_RATE, run_apollo

    too_long = np.zeros((2, int((APOLLO_MAX_SECONDS + 1) * APOLLO_SAMPLE_RATE)), np.float32)
    with pytest.raises(ValueError, match="rotary embeddings"):
        run_apollo(too_long, APOLLO_SAMPLE_RATE, chunk_seconds=None)

    short = np.zeros((2, APOLLO_SAMPLE_RATE), np.float32)
    with pytest.raises(ValueError, match="rotary embeddings"):
        run_apollo(short, APOLLO_SAMPLE_RATE, chunk_seconds=APOLLO_MAX_SECONDS + 1)
