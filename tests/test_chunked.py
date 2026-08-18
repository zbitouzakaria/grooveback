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
    context = min(int(chunk * overlap_frac), (chunk - 1) // 2)
    out = chunked(IDENTITY, audio, chunk, context)
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


def test_join_does_not_change_level():
    """A constant signal stays constant across boundaries.

    An unnormalized join dips at every seam. On music that reads as a faint
    tremolo, which is easy to miss and hard to trace back.
    """
    audio = torch.full((1, 2, 30_000), 0.5)
    out = chunked(IDENTITY, audio, 4_000, 1_000)
    assert torch.allclose(out, audio, atol=1e-6)


def test_context_consuming_whole_chunk_rejected():
    """context*2 == chunk leaves no core to keep."""
    with pytest.raises(ValueError, match="half the chunk"):
        chunked(IDENTITY, signal(10_000), 1_000, 500)


def test_single_realisation_outside_joins():
    """Away from the ~join-width seams, output equals one window's estimate
    exactly — no blending — which is the property this stitching exists for."""
    audio = signal(20_000)
    out = chunked(lambda b: b * 0.5, audio, 3_000, 500, join_samples=100)
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


def test_invented_content_survives_the_seams():
    """A model that invents content must not lose power where windows meet.

    Apollo synthesises everything above the codec cutoff, so two windows
    covering the same instant produce the same band at *different phase*.
    Crossfading them sums incoherently and loses about 3 dB in the middle of
    every overlap — audible as a periodic dropout, and invisible to the
    identity-model tests above, which are coherent by construction.

    The fake model here is that failure in its purest form: one tone, fixed
    amplitude, random phase per call.
    """
    sr, freq = 44_100, 18_000.0
    audio = torch.zeros((1, 1, 5 * sr))
    phases = iter(np.random.default_rng(0).uniform(0, 2 * np.pi, 64).tolist())

    def invent(batch: torch.Tensor) -> torch.Tensor:
        t = torch.arange(batch.shape[-1], dtype=torch.float32) / sr
        out = torch.empty_like(batch)
        for i in range(batch.shape[0]):
            out[i] = 0.5 * torch.sin(2 * np.pi * freq * t + next(phases))
        return out

    out = chunked(invent, audio, sr, int(0.2 * sr))[0, 0].numpy()

    # Power in 20 ms frames. A faithful stitch holds it constant; an
    # incoherent blend digs a hole wherever two realisations were averaged.
    frame = int(0.02 * sr)
    power = (out[: len(out) // frame * frame] ** 2).reshape(-1, frame).mean(axis=1)
    db = 10 * np.log10(power + 1e-12)
    dip = float(np.median(db) - db.min())

    # The join is 10 ms inside a 20 ms frame, so the worst frame can sag ~1.2 dB.
    # A crossfaded overlap would sit 3 dB down for 200 ms.
    assert dip < 2.0, f"lost {dip:.1f} dB at a seam; windows are being blended"
