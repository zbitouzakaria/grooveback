"""Baseline restoration methods.

Standing baselines, not components — see ADR-0005.

**Apollo** (Li & Luo, ICASSP 2025) targets codec artifacts. Vendored as a
submodule at `third_party/apollo` because it ships no installable package, and
imported directly: its model needs only torch, numpy and huggingface_hub.
Chunking is ours, since their crossfade lives in a top-level `inference.py`
script rather than in the `look2hear` package.

**A2SB** (NVIDIA) targets missing bandwidth. It lives in a fork with its own
environment and one-command entry point; this module shells out to it and
nothing more. It is a mono model, so its output is mono carried across the
input's channels.
"""

from __future__ import annotations

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
    mode: str = "crossfade",
    join_samples: int = 441,
) -> torch.Tensor:
    """Apply `fn` over overlapping chunks and reassemble the results.

    `audio` is `(1, channels, samples)`. Output length always equals input
    length. Tested against an identity `fn`, which is what catches the
    off-by-one and windowing bugs this function exists to hide.

    Two reassembly modes:

    - `"crossfade"`: overlap-add with linear ramps across `overlap_samples`,
      normalized by its own weights so the crossfade cannot change the level.
      When `fn` invents content (Apollo synthesizes everything above the codec
      cutoff), the two chunks disagree in phase across the entire overlap and
      the blend comb-filters — audible as brief crackles at every seam.
    - `"discard"`: `overlap_samples` becomes *context per side*. Windows overlap
      by twice that, each window's outer context is thrown away, and only the
      core is kept, so every output sample comes from a single forward pass
      that saw at least `overlap_samples` of audio on both sides. Consecutive
      cores are joined with a `join_samples` crossfade (~10 ms) to absorb the
      residual discontinuity.
    """
    total = audio.shape[-1]
    if chunk_samples is None or total <= chunk_samples:
        return fn(audio)
    if mode not in ("crossfade", "discard"):
        raise ValueError(f"Unknown mode {mode!r}.")
    # In discard mode the context is thrown away, so it must leave a core.
    limit = chunk_samples - 1 if mode == "discard" else chunk_samples
    if overlap_samples * 2 > limit:
        raise ValueError("Overlap must not exceed half the chunk length.")

    if mode == "discard":
        context = overlap_samples
        join = min(join_samples, context) if context else 0
        # Core hop; each window is core + context on both sides, and cores are
        # widened by the join fade so neighbours overlap by exactly `join`.
        hop = chunk_samples - 2 * context
        padded = torch.nn.functional.pad(audio, (context, context))
        starts = list(range(0, total, hop))
    else:
        hop = chunk_samples - overlap_samples
        starts = [0]
        while starts[-1] + chunk_samples < total:
            starts.append(starts[-1] + hop)

    out_sum = torch.zeros_like(audio)
    weight_sum = torch.zeros((1, 1, total), dtype=audio.dtype)

    for i in range(0, len(starts), batch_size):
        batch = starts[i : i + batch_size]
        chunks, keeps = [], []
        for start in batch:
            if mode == "discard":
                # Window in padded coordinates; core [start, start+hop) in
                # original coordinates, widened by join/2 on interior edges.
                lo = max(start - (join // 2 if start else 0), 0)
                hi = min(start + hop + join - join // 2, total)
                chunk = padded[..., start : start + chunk_samples]
                keeps.append((lo, hi, lo - start + context))
            else:
                end = min(start + chunk_samples, total)
                chunk = audio[..., start:end]
                keeps.append((start, end, 0))
            if chunk.shape[-1] < chunk_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, chunk_samples - chunk.shape[-1])
                )
            chunks.append(chunk)

        processed = fn(torch.cat(chunks, dim=0))

        for offset, (lo, hi, src) in enumerate(keeps):
            length = hi - lo
            if mode == "discard":
                weights = _crossfade(
                    length,
                    join,
                    fade_in=lo > 0,
                    fade_out=hi < total,
                    dtype=audio.dtype,
                )
                piece = processed[offset : offset + 1, ..., src : src + length]
            else:
                weights = _crossfade(
                    length,
                    overlap_samples,
                    fade_in=i + offset > 0,
                    fade_out=i + offset < len(starts) - 1,
                    dtype=audio.dtype,
                )
                piece = processed[offset : offset + 1, ..., :length]
            out_sum[..., lo:hi] += piece * weights
            weight_sum[..., lo:hi] += weights

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
    chunk_seconds: float | None = 12.0,
    overlap_seconds: float = 2.0,
    batch_size: int = 1,
    mode: str = "discard",
) -> np.ndarray:
    """Restore `(channels, samples)` audio with Apollo.

    Chunking defaults to on. Apollo will happily try to process a seven-minute
    track in one pass and exhaust 16 GB of unified memory doing it.

    Default reassembly is `"discard"`: Apollo synthesizes the band above the
    codec cutoff, two chunks realise different phase there, and crossfading
    them comb-filters — the periodic crackle heard on full-track renders.
    Discard keeps one realisation everywhere. `overlap_seconds` is context per
    side in that mode.
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
            mode=mode,
        )
    return output.squeeze(0).numpy()


# --- A2SB ------------------------------------------------------------------
# Fork: github.com/zbitouzakaria/diffusion-audio-restoration, branch
# runnable-anywhere, cloned (gitignored) at third_party/a2sb.

A2SB_SAMPLE_RATE = 44_100
A2SB_DIR = _REPO / "third_party" / "a2sb"
A2SB_PYTHON = A2SB_DIR / ".venv" / "bin" / "python"


def a2sb_fork_sha() -> str:
    """Short HEAD of the fork clone, printed with every run for provenance."""
    result = subprocess.run(
        ["git", "-C", str(A2SB_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def run_a2sb(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: int = 20,
    cutoff_hz: float | None = None,
    single_split: bool = False,
    device: str = "mps",
) -> np.ndarray:
    """Restore `(channels, samples)` audio with the A2SB fork.

    The model is mono: output is the restored mono copied across the input's
    channel count. Cutoff detection happens fork-side on the audio it is
    handed, so when processing an excerpt of a longer file, pass the full
    file's `cutoff_hz` explicitly.
    """
    from grooveback import audio as ga

    if sample_rate != A2SB_SAMPLE_RATE:
        raise ValueError(f"A2SB expects {A2SB_SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    if not (A2SB_DIR / "restore.py").exists():
        raise FileNotFoundError(
            f"A2SB fork missing at {A2SB_DIR}. Clone it:\n"
            "  git clone -b runnable-anywhere "
            f"git@github.com:zbitouzakaria/diffusion-audio-restoration.git {A2SB_DIR}"
        )
    if not A2SB_PYTHON.exists():
        raise FileNotFoundError(
            f"A2SB environment missing at {A2SB_PYTHON}. Create it:\n"
            f"  {A2SB_DIR}/setup.sh"
        )
    print(f"a2sb: fork @ {a2sb_fork_sha()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav_in, wav_out = tmp / "in.wav", tmp / "out.wav"
        ga.save(wav_in, audio, sample_rate)
        cmd = [
            str(A2SB_PYTHON), "restore.py", str(wav_in), str(wav_out),
            f"--steps={n_steps}", f"--device={device}",
        ]
        if cutoff_hz is not None:
            cmd.append(f"--cutoff-hz={cutoff_hz}")
        if single_split:
            cmd.append("--single-split")
        result = subprocess.run(cmd, cwd=A2SB_DIR, capture_output=True, text=True)
        if result.returncode != 0 or not wav_out.exists():
            raise RuntimeError(
                f"A2SB failed.\n{(result.stderr or result.stdout)[-2000:]}"
            )
        for line in result.stdout.splitlines():
            if line.startswith("restore:"):
                print(f"a2sb: {line[9:]}")
        restored, _ = ga.load(wav_out)

    out = np.zeros((audio.shape[0], audio.shape[1]), dtype=np.float32)
    n = min(audio.shape[1], restored.shape[1])
    out[:, :n] = restored[0, :n]
    return out
