"""Baseline restoration methods.

Standing baselines, not components — see ADR-0005.

**Apollo** (Li & Luo, ICASSP 2025) targets codec artifacts. Vendored as a
submodule at `third_party/apollo` because it ships no installable package, and
imported directly: its model needs only torch, numpy and huggingface_hub.
Chunking is ours: upstream's lives in a top-level `inference.py` script rather
than in the `look2hear` package, and crossfades its overlaps, which is wrong
for a model that invents content.

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

# Apollo's rotary tables cover 10_000 positions at 100 frames/s. Past that it
# dies inside the model on a broadcast mismatch — architectural, not memory, so
# no GPU raises it. Attention is also quadratic, so the practical limit is lower.
APOLLO_MAX_SECONDS = 10_000 * (_APOLLO_ARCH["win"] // 2) / 1000
APOLLO_SAFE_CHUNK_SECONDS = 90.0


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
    context_samples: int = 0,
    batch_size: int = 1,
    join_samples: int = 441,
) -> torch.Tensor:
    """Apply `fn` over windows and stitch the results back together.

    `audio` is `(1, channels, samples)`; output length equals input length.

    Windows overlap by `context_samples` per side and that context is
    discarded, so every output sample comes from one forward pass. Cores meet
    with a short `join_samples` crossfade.

    Crossfading the whole overlap instead would be wrong for any `fn` that
    invents content: two windows produce the same band at different phase, and
    averaging them loses ~3 dB rather than blending. Keeping one realisation
    and confining the blend to ~10 ms is the fix.
    """
    total = audio.shape[-1]
    if chunk_samples is None or total <= chunk_samples:
        return fn(audio)
    # The context is thrown away, so it has to leave a core behind.
    if context_samples * 2 >= chunk_samples:
        raise ValueError("Context must be under half the chunk length.")

    join = min(join_samples, context_samples) if context_samples else 0
    hop = chunk_samples - 2 * context_samples
    padded = torch.nn.functional.pad(audio, (context_samples, context_samples))
    starts = list(range(0, total, hop))

    out_sum = torch.zeros_like(audio)
    weight_sum = torch.zeros((1, 1, total), dtype=audio.dtype)

    for i in range(0, len(starts), batch_size):
        batch = starts[i : i + batch_size]
        windows, keeps = [], []
        for start in batch:
            # Core is [start, start+hop) in original coordinates, widened by
            # half a join on interior edges so neighbours meet with a fade.
            lo = max(start - (join // 2 if start else 0), 0)
            hi = min(start + hop + join - join // 2, total)
            keeps.append((lo, hi, lo - start + context_samples))
            window = padded[..., start : start + chunk_samples]
            if window.shape[-1] < chunk_samples:
                window = torch.nn.functional.pad(
                    window, (0, chunk_samples - window.shape[-1])
                )
            windows.append(window)

        processed = fn(torch.cat(windows, dim=0))

        for offset, (lo, hi, src) in enumerate(keeps):
            length = hi - lo
            weights = _join_ramp(
                length, join, fade_in=lo > 0, fade_out=hi < total, dtype=audio.dtype
            )
            out_sum[..., lo:hi] += (
                processed[offset : offset + 1, ..., src : src + length] * weights
            )
            weight_sum[..., lo:hi] += weights

    return out_sum / weight_sum.clamp(min=1e-8)


def _join_ramp(
    length: int, join: int, fade_in: bool, fade_out: bool, dtype: torch.dtype
) -> torch.Tensor:
    """Linear edge ramps for the short join between cores.

    The ramp is open at both ends — `i/(fade+1)` for `i` in `1..fade`, never
    reaching 0 or 1. A ramp starting at 0 zeroes the shared sample from both
    sides when `join == 1`, leaving zero total weight and a hole in the output.
    Upstream Apollo's `linspace(0, 1, fade)` has this bug; normalization hides
    it for wider joins but still skews the blend at the seams.
    """
    weights = torch.ones(length, dtype=dtype)
    fade = min(join, length)
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
    context_seconds: float = 2.0,
    batch_size: int = 1,
) -> np.ndarray:
    """Restore `(channels, samples)` audio with Apollo.

    Chunking is mandatory, not an optimisation: Apollo cannot process more than
    `APOLLO_MAX_SECONDS` of audio at all, and runs out of memory well before
    that. Passing `chunk_seconds=None` is only valid for input already under
    the limit.

    `context_seconds` is extra audio given to the model on each side of a
    window and then thrown away; see `chunked`.
    """
    if sample_rate != APOLLO_SAMPLE_RATE:
        raise ValueError(
            f"Apollo expects {APOLLO_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains NaN or infinite values.")

    span = chunk_seconds if chunk_seconds else audio.shape[-1] / sample_rate
    if span > APOLLO_MAX_SECONDS:
        what = "chunk_seconds" if chunk_seconds else "input"
        raise ValueError(
            f"{what} is {span:.1f} s; Apollo's rotary embeddings only cover "
            f"{APOLLO_MAX_SECONDS:.0f} s. Past that it fails inside the model "
            f"on a shape mismatch. Set chunk_seconds to "
            f"{APOLLO_SAFE_CHUNK_SECONDS:.0f} or less."
        )

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
            int(round(context_seconds * sample_rate)),
            batch_size,
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
