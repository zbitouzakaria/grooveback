"""Baseline restoration methods.

Standing baselines, not components — see ADR-0005.

**Apollo** (Li & Luo, ICASSP 2025) targets codec artifacts. Vendored as a
submodule at `third_party/apollo` because it ships no installable package, and
imported directly: its model needs only torch, numpy and huggingface_hub.
Chunking is ours, since their crossfade lives in a top-level `inference.py`
script rather than in the `look2hear` package.

**A2SB** (NVIDIA) targets missing bandwidth. It is kept behind a subprocess
boundary in its own environment rather than imported: it wants pytorch_lightning
and `ssr_eval`, and the latter is not installable at all. Its entry point is a
Lightning CLI driven by YAML, so this module generates a config and shells out.
It is also mono, so stereo costs two passes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]

APOLLO_SAMPLE_RATE = 44_100
APOLLO_CHECKPOINT = "JusperLee/Apollo"
_APOLLO_REPO = _REPO / "third_party" / "apollo"

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


# --- A2SB ------------------------------------------------------------------

A2SB_SAMPLE_RATE = 44_100
A2SB_REPO = _REPO / "third_party" / "a2sb"
A2SB_PYTHON = _REPO / ".venvs" / "a2sb" / "bin" / "python"
A2SB_SHIMS = _REPO / "third_party" / "shims"
A2SB_HF_REPO = "nvidia/audio_to_audio_schrodinger_bridge"
A2SB_SINGLE_SPLIT = "ckpt/A2SB_onesplit_0.0_1.0_release.ckpt"
A2SB_TWO_SPLIT = (
    "ckpt/A2SB_twosplit_0.0_0.5_release.ckpt",
    "ckpt/A2SB_twosplit_0.5_1.0_release.ckpt",
)


def a2sb_checkpoints(ensemble: bool = False) -> list[str]:
    """Fetch A2SB weights, 2.26 GB per checkpoint.

    `ensemble=True` uses the two-split pair the paper reports, at twice the
    download, memory and compute of the single-split model.
    """
    from huggingface_hub import hf_hub_download

    names = A2SB_TWO_SPLIT if ensemble else (A2SB_SINGLE_SPLIT,)
    return [hf_hub_download(A2SB_HF_REPO, n) for n in names]


def _a2sb_config(wav_in: Path, checkpoints: list[str], device: str, cutoff_hz: float | None) -> str:
    """Override for A2SB's cluster-shaped defaults: gpu + ddp + a SLURM plugin."""
    cutoff = "" if cutoff_hz is None else f"""
  transforms_aug:
    - class_path: corruption.corruptions.MultinomialInpaintMaskTransform
      init_args:
        p_upsample_mask: 1.0
        p_extension_mask: 0.0
        p_inpaint_mask: 0.0
        fill_noise_level: 0.5
        sampling_rate: {A2SB_SAMPLE_RATE}
        upsample_mask_kwargs:
          min_cutoff_freq: {int(cutoff_hz)}
          max_cutoff_freq: {int(cutoff_hz)}
        inpainting_mask_kwargs:
          min_inpainting_frac: 0.1013
          max_inpainting_frac: 0.1013
          is_random: false"""
    ckpts = "\n".join(f"    - {c}" for c in checkpoints)
    cutoffs = "[0.5]" if len(checkpoints) == 2 else "[]"
    return f"""
trainer:
  accelerator: {device}
  strategy: auto
  devices: 1
  num_nodes: 1
  plugins: null
  logger: false
checkpoint_callback:
  dirpath: {wav_in.parent / "lightning"}
model:
  pretrained_checkpoints:
{ckpts}
  t_cutoffs: {cutoffs}
data:
  num_workers: 0
  batch_size: 1
  mix_dataset_config: {{}}
  predict_filelist:
    - filepath: {wav_in}
      output_subdir: "."{cutoff}
"""


def run_a2sb(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: int = 20,
    checkpoints: list[str] | None = None,
    device: str = "mps",
    cutoff_hz: float | None = None,
    predict_batch_size: int = 2,
) -> np.ndarray:
    """Restore `(channels, samples)` audio with A2SB.

    A2SB mono-sums on load, so each channel is a separate pass. That doubles the
    cost and means the two channels are synthesised independently above the
    cutoff — worth measuring inter-channel correlation up there before trusting
    the stereo image.

    Length is handled inside A2SB, which slides a 256-frame window over the
    spectrogram and runs `predict_batch_size` windows per forward pass. Their
    default of 16 exhausts MPS on anything past a few seconds, so this defaults
    low. It is a memory/throughput knob only and does not change the output.

    Slow regardless: tens of times slower than realtime on MPS, so full tracks
    are an overnight or rented-GPU job. Use excerpts locally.
    """
    from grooveback import audio as ga

    if sample_rate != A2SB_SAMPLE_RATE:
        raise ValueError(f"A2SB expects {A2SB_SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    if not A2SB_PYTHON.exists():
        raise FileNotFoundError(
            f"A2SB environment missing at {A2SB_PYTHON}. See docs/decisions/0005 — "
            "it needs its own venv because ssr_eval will not install."
        )
    checkpoints = checkpoints or a2sb_checkpoints()

    # A2SB regenerates everything above the cutoff it is given, and the shipped
    # config says 2000 Hz. Left alone it rebuilds most of the spectrum, which
    # overwrites real content and — because each channel is synthesised
    # separately — collapses the stereo image toward mono.
    if cutoff_hz is None:
        cutoff_hz = ga.bandwidth_hz(audio, sample_rate)

    channels = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for index in range(audio.shape[0]):
            wav_in = tmp / f"ch{index}_in.wav"
            wav_out = tmp / f"ch{index}_out.wav"
            ga.save(wav_in, audio[index : index + 1], sample_rate)
            config = tmp / f"ch{index}.yaml"
            config.write_text(_a2sb_config(wav_in, checkpoints, device, cutoff_hz))

            env = {**os.environ, "PYTHONPATH": str(A2SB_SHIMS)}
            result = subprocess.run(
                [
                    str(A2SB_PYTHON), "ensembled_inference_api.py", "predict",
                    "-c", "configs/ensemble_2split_sampling.yaml",
                    "-c", "configs/inference_files_upsampling.yaml",
                    "-c", str(config),
                    f"--model.predict_n_steps={n_steps}",
                    f"--model.predict_batch_size={predict_batch_size}",
                    f"--model.output_audio_filename={wav_out}",
                ],
                cwd=A2SB_REPO, env=env, capture_output=True, text=True,
            )
            if not wav_out.exists():
                raise RuntimeError(f"A2SB produced no output.\n{result.stderr[-2000:]}")
            restored, _ = ga.load(wav_out)
            channels.append(restored[0])

    # A2SB returns a few hundred samples short of its input; pad so the result
    # lines up with the original for any comparison.
    total = audio.shape[1]
    out = np.zeros((len(channels), total), dtype=np.float32)
    for index, channel in enumerate(channels):
        n = min(total, channel.size)
        out[index, :n] = channel[:n]
    return out
