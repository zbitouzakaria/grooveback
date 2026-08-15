"""Baseline restoration methods.

Apollo (Li & Luo, ICASSP 2025) restores MP3-compressed music. It is a standing
baseline, not a component — see ADR-0005.

The model comes from the Apollo repo, vendored as a submodule at
`third_party/apollo` because it ships no installable package. Chunking is ours:
their crossfade lives in a top-level `inference.py` script rather than in the
`look2hear` package, and importing a module called `inference` into this
namespace is a trap.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

APOLLO_SAMPLE_RATE = 44_100
APOLLO_CHECKPOINT = "JusperLee/Apollo"
_APOLLO_REPO = Path(__file__).resolve().parents[2] / "third_party" / "apollo"

# Apollo's hyperparameters are not in the checkpoint — the Hugging Face repo
# holds only pytorch_model.bin, no config — so they are pinned here from the
# upstream inference script.
_APOLLO_ARCH = {"sr": APOLLO_SAMPLE_RATE, "win": 20, "feature_dim": 256, "layer": 6}


def select_device(requested: str = "auto") -> torch.device:
    """Resolve a device, preferring CUDA, then MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_apollo(checkpoint: str | Path = APOLLO_CHECKPOINT, device: str = "auto"):
    """Load the pretrained Apollo model.

    The default downloads the official checkpoint from Hugging Face and loads it
    with `torch.load`, which is arbitrary deserialization — the usual bargain
    with research checkpoints.
    """
    if not (_APOLLO_REPO / "look2hear").is_dir():
        raise FileNotFoundError(
            f"Apollo submodule missing at {_APOLLO_REPO}. "
            "Run: git submodule update --init --recursive"
        )
    if str(_APOLLO_REPO) not in sys.path:
        sys.path.insert(0, str(_APOLLO_REPO))

    import look2hear.models

    if str(checkpoint) == APOLLO_CHECKPOINT:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            repo_id=APOLLO_CHECKPOINT, filename="pytorch_model.bin"
        )

    model = look2hear.models.BaseModel.from_pretrain(str(checkpoint), **_APOLLO_ARCH)
    return model.to(select_device(device)).eval()


def chunked(
    fn: Callable[[torch.Tensor], torch.Tensor],
    audio: torch.Tensor,
    chunk_samples: int | None,
    overlap_samples: int = 0,
    batch_size: int = 1,
) -> torch.Tensor:
    """Apply `fn` over overlapping chunks and crossfade the results back.

    `audio` is `(1, channels, samples)`. Output length always equals input
    length: chunks are padded for the forward pass and cropped afterwards, and
    the overlap-add is normalized by its own weights so the crossfade cannot
    change the level. Tested against an identity `fn`, which is what catches the
    off-by-one and windowing bugs this function exists to hide.
    """
    total = audio.shape[-1]
    if chunk_samples is None or total <= chunk_samples:
        return fn(audio)
    if overlap_samples * 2 > chunk_samples:
        raise ValueError("Overlap must not exceed half the chunk length.")

    hop = chunk_samples - overlap_samples
    starts = [0]
    while starts[-1] + chunk_samples < total:
        starts.append(starts[-1] + hop)

    out_sum = torch.zeros_like(audio)
    weight_sum = torch.zeros((1, 1, total), dtype=audio.dtype)

    for i in range(0, len(starts), batch_size):
        batch = starts[i : i + batch_size]
        chunks, lengths = [], []
        for start in batch:
            end = min(start + chunk_samples, total)
            chunk = audio[..., start:end]
            lengths.append(end - start)
            if chunk.shape[-1] < chunk_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, chunk_samples - chunk.shape[-1])
                )
            chunks.append(chunk)

        processed = fn(torch.cat(chunks, dim=0))

        for offset, (start, length) in enumerate(zip(batch, lengths, strict=True)):
            weights = _crossfade(
                length,
                overlap_samples,
                fade_in=i + offset > 0,
                fade_out=i + offset < len(starts) - 1,
                dtype=audio.dtype,
            )
            out_sum[..., start : start + length] += (
                processed[offset : offset + 1, ..., :length] * weights
            )
            weight_sum[..., start : start + length] += weights

    return out_sum / weight_sum.clamp(min=1e-8)


def _crossfade(
    length: int, overlap: int, fade_in: bool, fade_out: bool, dtype: torch.dtype
) -> torch.Tensor:
    """Linear edge ramps for normalized overlap-add.

    The ramp is open at both ends — `i/(fade+1)` for `i` in `1..fade`, never
    reaching 0 or 1. A ramp starting at 0 zeroes the shared sample from both
    sides when `overlap == 1`, leaving zero total weight and a hole in the
    output. Upstream Apollo's `linspace(0, 1, fade)` has this bug; normalization
    hides it for wider overlaps but still skews the crossfade at the seams.
    """
    weights = torch.ones(length, dtype=dtype)
    fade = min(overlap, length)
    if fade:
        ramp = (torch.arange(fade, dtype=dtype) + 1.0) / (fade + 1.0)
        if fade_in:
            weights[:fade] = ramp
        if fade_out:
            weights[-fade:] = ramp.flip(0)
    return weights.view(1, 1, -1)


def run_apollo(
    audio: np.ndarray,
    sample_rate: int,
    model=None,
    device: str = "auto",
    chunk_seconds: float | None = 10.0,
    overlap_seconds: float = 1.0,
    batch_size: int = 1,
) -> np.ndarray:
    """Restore `(channels, samples)` audio with Apollo.

    Chunking defaults to on. Apollo will happily try to process a seven-minute
    track in one pass and exhaust 16 GB of unified memory doing it.
    """
    if sample_rate != APOLLO_SAMPLE_RATE:
        raise ValueError(
            f"Apollo expects {APOLLO_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains NaN or infinite values.")

    if model is None:
        model = load_apollo(device=device)
    target = next(model.parameters()).device

    tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    chunk_samples = (
        int(round(chunk_seconds * sample_rate)) if chunk_seconds else None
    )

    def forward(batch: torch.Tensor) -> torch.Tensor:
        return model(batch.to(target)).detach().to("cpu")

    with torch.inference_mode():
        output = chunked(
            forward,
            tensor,
            chunk_samples,
            int(round(overlap_seconds * sample_rate)),
            batch_size,
        )
    return output.squeeze(0).numpy()
