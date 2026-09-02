"""Baseline restoration methods.

Standing baselines, not components — see ADR-0005.

**Apollo** (Li & Luo, ICASSP 2025) targets codec artifacts. Vendored as a
submodule at `third_party/apollo` because it ships no installable package, and
imported directly: its model needs only torch, numpy and huggingface_hub.

**A2SB** (NVIDIA) targets missing bandwidth. It lives in a fork with its own
environment and one-command entry point; this module shells out to it and
nothing more. The model is mono; the fork restores a stereo file one channel
at a time in a single inference run, so output keeps the input's channels.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
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


def select_device(requested: str = "auto") -> torch.device:
    """Resolve a device, preferring CUDA, then MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _apollo_import(name: str):
    """Import a module from the vendored Apollo repo.

    Apollo ships no package — `inference` and `look2hear/` are top-level names
    in its repo — so the repo root goes on `sys.path`, at the front so those
    names resolve to Apollo's own modules. Lazy because `baselines` must stay
    importable without the submodule: the A2SB path needs none of this.
    """
    if not (_APOLLO_REPO / "look2hear").is_dir():
        raise FileNotFoundError(
            f"Apollo submodule missing at {_APOLLO_REPO}. "
            "Run: git submodule update --init --recursive"
        )
    if str(_APOLLO_REPO) not in sys.path:
        sys.path.insert(0, str(_APOLLO_REPO))
    return importlib.import_module(name)


def load_apollo(checkpoint: str | Path = APOLLO_CHECKPOINT, device: str = "auto"):
    """Load the pretrained Apollo model.

    The default downloads the official checkpoint from Hugging Face and loads it
    with `torch.load`, which is arbitrary deserialization — the usual bargain
    with research checkpoints.
    """
    models = _apollo_import("look2hear.models")

    if str(checkpoint) == APOLLO_CHECKPOINT:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            repo_id=APOLLO_CHECKPOINT, filename="pytorch_model.bin"
        )

    model = models.BaseModel.from_pretrain(str(checkpoint), **_APOLLO_ARCH)
    return model.to(select_device(device)).eval()


def run_apollo(
    audio: np.ndarray,
    sample_rate: int,
    model=None,
    device: str = "auto",
    chunk_seconds: float | None = 12.0,
    overlap_seconds: float = 1.0,
    chunk_pad_seconds: float = 1.0,
    batch_size: int = 1,
) -> np.ndarray:
    """Restore `(channels, samples)` audio with Apollo.

    Each chunk is inferred with `chunk_pad_seconds` of surrounding audio per
    side, discarded from the output, because the model's output is wrong near
    the edges of its input.

    `chunk_seconds + 2 * chunk_pad_seconds` must stay within
    `APOLLO_MAX_SECONDS` — the rotary ceiling holds on every device, so real
    tracks are always chunked. The defaults are sized for a 16 GB MPS laptop;
    on CUDA, raise `chunk_seconds` toward the ceiling (90 is a round choice)
    and `batch_size` as memory allows. `chunk_seconds=None` runs a single
    pass, which only fits inputs shorter than `APOLLO_MAX_SECONDS`.
    """
    if sample_rate != APOLLO_SAMPLE_RATE:
        raise ValueError(
            f"Apollo expects {APOLLO_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains NaN or infinite values.")

    span = (
        chunk_seconds + 2 * chunk_pad_seconds
        if chunk_seconds
        else audio.shape[-1] / sample_rate
    )
    if span > APOLLO_MAX_SECONDS:
        what = "chunk plus padding" if chunk_seconds else "input"
        raise ValueError(
            f"{what} is {span:.1f} s; Apollo's rotary embeddings only cover "
            f"{APOLLO_MAX_SECONDS:.0f} s. Use chunk_seconds of at most "
            f"{APOLLO_MAX_SECONDS - 2 * chunk_pad_seconds:.0f} s with this "
            f"padding."
        )

    inference = _apollo_import("inference")
    if model is None:
        model = load_apollo(device=device)
    target = next(model.parameters()).device

    tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    with torch.inference_mode():
        output = inference.run_model(
            model,
            tensor,
            target,
            chunk_samples=(
                int(round(chunk_seconds * sample_rate)) if chunk_seconds else None
            ),
            overlap_samples=int(round(overlap_seconds * sample_rate)),
            chunk_batch_size=batch_size,
            chunk_pad_samples=int(round(chunk_pad_seconds * sample_rate)),
        )
    return output.squeeze(0).numpy()


# --- A2SB ------------------------------------------------------------------
# Fork: github.com/zbitouzakaria/diffusion-audio-restoration, branch
# runnable-anywhere, cloned (gitignored) at third_party/a2sb.

A2SB_SAMPLE_RATE = 44_100
A2SB_DIR = _REPO / "third_party" / "a2sb"


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

    Stereo is restored per channel fork-side (one model load, both channels),
    so output keeps the input's channel count. Cutoff detection happens
    fork-side on the audio it is handed, so when processing an excerpt of a
    longer file, pass the full file's `cutoff_hz` explicitly.
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
    python = A2SB_DIR / ".venv" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(
            f"A2SB environment missing at {python}. Create it:\n"
            f"  {A2SB_DIR}/setup.sh"
        )
    print(f"a2sb: fork @ {a2sb_fork_sha()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav_in, wav_out = tmp / "in.wav", tmp / "out.wav"
        ga.save(wav_in, audio, sample_rate)
        cmd = [
            str(python), "restore.py", str(wav_in), str(wav_out),
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
    # Broadcasting copies a mono render across the input's channels.
    out[:, :n] = restored[:, :n]
    return out
