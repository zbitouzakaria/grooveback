"""Generating audio with the Stable Audio 3 prior.

The prior is the artifact this project is about (ADR-0004): a generative
model over music, living in the SAME latent space. This module wraps the
`stable-audio-3` dependency's in-process API the way `latents.py` wraps the
autoencoder: load once, pass the model around, numpy out.
"""

from __future__ import annotations

import numpy as np

PRIOR_SAMPLE_RATE = 44_100

PRIOR_VARIANTS = {
    # Post-trained checkpoints sample in a few unguided steps; the -base
    # checkpoints need classic many-step guided sampling (the package's
    # docs/workflows/inference.md).
    "small-music": {"steps": 8, "cfg_scale": 1.0},
    "small-music-base": {"steps": 50, "cfg_scale": 7.0},
    "small-sfx": {"steps": 8, "cfg_scale": 1.0},
    "small-sfx-base": {"steps": 50, "cfg_scale": 7.0},
    "medium": {"steps": 8, "cfg_scale": 1.0},
    "medium-base": {"steps": 50, "cfg_scale": 7.0},
}
"""Every released model type, with the sampling settings its family needs."""


def load_prior(variant: str = "small-music", device: str = "auto"):
    """Load a Stable Audio 3 checkpoint by variant name.

    Every checkpoint is gated on Hugging Face: accept the license of each
    `stabilityai/stable-audio-3-*` repo, request access to
    `google/t5gemma-b-b-ul2` (the text encoder), and put `HF_TOKEN` in `.env`.
    """
    if variant not in PRIOR_VARIANTS:
        raise ValueError(
            f"variant must be one of {sorted(PRIOR_VARIANTS)}, got {variant!r}."
        )
    # Imported here so this module stays light to import; loading is the
    # heavy step anyway.
    from huggingface_hub.errors import HfHubHTTPError
    from stable_audio_3 import StableAudioModel

    try:
        return StableAudioModel.from_pretrained(
            variant, device=None if device == "auto" else device
        )
    except HfHubHTTPError as error:
        raise RuntimeError(
            f"Could not download {variant!r}. The checkpoints are gated: "
            "accept the license of each stabilityai/stable-audio-3-* repo, "
            "request access to google/t5gemma-b-b-ul2, and set HF_TOKEN in "
            ".env."
        ) from error


def generate(
    model, prompt: str, *, seconds: float, seed: int, **sampling
) -> np.ndarray:
    """One clip from a loaded prior, `(2, samples)` float32 at 44.1 kHz.

    Pass the variant's `PRIOR_VARIANTS` entry as `sampling` (`steps`,
    `cfg_scale`) so each family samples the way it was trained to.
    """
    audio = model.generate(prompt=prompt, duration=seconds, seed=seed, **sampling)
    return np.ascontiguousarray(
        audio.squeeze(0).float().cpu().numpy().astype(np.float32)
    )
