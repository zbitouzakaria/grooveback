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


def a2sb_checkpoints(ensemble: bool = True) -> list[str]:
    """Fetch A2SB weights, 2.26 GB per checkpoint.

    The two-split ensemble is the default because it is what the paper
    evaluates and what upstream's own exp config uses — and the difference is
    not subtle. On identical input the single-split checkpoint paints a flat
    shelf (~-45 dB from 4 kHz to 19 kHz) where the ensemble rolls off like
    music (-46 down to -92). Verified bit-exact against upstream inference
    with the ensemble; `ensemble=False` halves download and memory for smoke
    tests only.
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
    segment_seconds: float | None = 30.0,
    overlap_seconds: float = 1.0,
) -> np.ndarray:
    """Restore `(channels, samples)` audio with A2SB.

    **A2SB is a mono model.** The input is mono-summed, restored in one pass, and
    the result copied across the output channels. So a stereo input comes back
    mono-in-stereo, narrower than it went in.

    That is the model's limitation, surfaced rather than hidden. Running the
    channels separately would keep two of them, but each would be synthesised
    independently: measured on this material, inter-channel correlation in the
    restored band fell from 0.99 to 0.56, which is invented stereo width rather
    than recovered width. It also doubles an already severe cost. If A2SB is
    being judged as a baseline, it should be judged as what it is.

    Long input is segmented here rather than passed whole. A2SB's `ddpm_sample`
    appends every diffusion step's full spectrogram to a list and returns all of
    them, though only the last is used — 482 MB per step for a seven-minute
    track, so 20 steps is ~10 GB of pure retention before the model or working
    tensors are counted. Segmenting caps that regardless of track length, and
    costs nothing in quality: A2SB's own multidiffusion already windows at 256
    frames (~3 s), so it never sees long context anyway. Segments are crossfaded
    with the same overlap-add used for Apollo, which is covered by the identity
    test.

    `predict_batch_size` sets how many spectrogram windows go through the UNet
    per forward pass. Their default of 16 exhausts MPS quickly, so this defaults
    low. Verified output-neutral: it only sets the chunk count in
    `get_multidiffusion_vf`, and the windows are recombined identically.

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

    mono = torch.from_numpy(
        np.ascontiguousarray(audio.mean(axis=0, keepdims=True))
    ).unsqueeze(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config = tmp / "run.yaml"
        env = {**os.environ, "PYTHONPATH": str(A2SB_SHIMS)}
        counter = {"n": 0}

        def a2sb_once(batch: torch.Tensor) -> torch.Tensor:
            counter["n"] += 1
            index = counter["n"]
            wav_in, wav_out = tmp / f"seg{index}_in.wav", tmp / f"seg{index}_out.wav"
            ga.save(wav_in, batch[0].numpy(), sample_rate)
            config.write_text(_a2sb_config(wav_in, checkpoints, device, cutoff_hz))
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
                raise RuntimeError(
                    f"A2SB produced no output.\n{result.stderr[-2000:]}"
                )
            restored, _ = ga.load(wav_out)
            wav_in.unlink(missing_ok=True)

            # A2SB returns a few hundred samples short; pad back so overlap-add
            # lines up with the segment it was given.
            out = torch.zeros_like(batch)
            n = min(batch.shape[-1], restored.shape[1])
            out[0, 0, :n] = torch.from_numpy(restored[0, :n])
            wav_out.unlink(missing_ok=True)
            return out

        segment_samples = (
            int(round(segment_seconds * sample_rate)) if segment_seconds else None
        )
        restored = chunked(
            a2sb_once,
            mono,
            segment_samples,
            int(round(overlap_seconds * sample_rate)),
            batch_size=1,
        )

    # The single restored channel is copied across, so the output is mono
    # carried in the input's shape.
    return np.repeat(restored[0].numpy(), audio.shape[0], axis=0)
