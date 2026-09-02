"""SAME latent space: encode, decode, round-trip.

SAME is the autoencoder Stable Audio 3 generates into, so it is the space any
future prior will live in. It compresses 44.1 kHz stereo by 4096x in time into
256 channels — about 10.8 latent frames per second.

`stable-audio-3` is a project dependency (it pins torch to 2.7.1), so the
model runs in-process. Load once with `load_same` and pass the model around:
loading SAME-L costs more than encoding a clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from stable_audio_3.loading_utils import load_autoencoder

from grooveback.baselines import select_device

SAME_SAMPLE_RATE = 44_100
SAME_CHECKPOINTS = {"same-s": "stabilityai/SAME-S", "same-l": "stabilityai/SAME-L"}
"""Only two variants exist. Stable Audio 3 small generates into `same-s`."""


def load_same(variant: str = "same-s", device: str = "auto"):
    """Load a SAME autoencoder from its official Hugging Face checkpoint."""
    if variant not in SAME_CHECKPOINTS:
        raise ValueError(
            f"variant must be one of {sorted(SAME_CHECKPOINTS)}, got {variant!r}."
        )
    config_path = hf_hub_download(SAME_CHECKPOINTS[variant], "model_config.json")
    weights_path = hf_hub_download(SAME_CHECKPOINTS[variant], "model.safetensors")
    config_rate = json.loads(Path(config_path).read_text())["sample_rate"]
    if config_rate != SAME_SAMPLE_RATE:
        raise RuntimeError(
            f"{variant} checkpoint says {config_rate} Hz, expected "
            f"{SAME_SAMPLE_RATE} Hz — stale or wrong checkpoint."
        )
    model = load_autoencoder(config_path, weights_path, str(select_device(device)))
    return model.eval()


def encode(audio: np.ndarray, sample_rate: int, model) -> np.ndarray:
    """Audio `(channels, samples)` to latents `(256, frames)`."""
    if sample_rate != SAME_SAMPLE_RATE:
        raise ValueError(f"SAME expects {SAME_SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(device)
    with torch.inference_mode():
        latents = model.encode_audio(batch)
    return latents.squeeze(0).float().cpu().numpy()


def decode(latents: np.ndarray, model) -> np.ndarray:
    """Latents `(256, frames)` back to audio `(channels, samples)`.

    Output length is the latent count times 4096, so it can overrun the
    original by up to one frame. Callers comparing against a source should
    trim both to the shorter length.
    """
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.ascontiguousarray(latents)).unsqueeze(0).to(device)
    with torch.inference_mode():
        audio = model.decode_audio(batch)
    return audio.squeeze(0).float().cpu().numpy()


def roundtrip(audio: np.ndarray, sample_rate: int, model) -> np.ndarray:
    """`decode(encode(audio))` — what survives a trip through the latent space.

    Not an identity, and not even close on bandlimited input: the decoder
    re-realises phase and invents high-frequency content where the input has
    none.
    """
    return decode(encode(audio, sample_rate, model), model)
