"""Restoring audio with the prior at inference time.

A solver takes degraded audio and uses the prior (ADR-0004) to propose a
clean version — no surrogate model, no fine-tuning, the damage is handled
entirely at inference. SDEdit is the first solver (ADR-0009).
"""

from __future__ import annotations

import numpy as np
import torch

from grooveback.priors import PRIOR_SAMPLE_RATE

SDEDIT_MAX_SECONDS = 360.0
"""Whole files, one model call. The `seconds_total` conditioner is calibrated
up to 384 s and the schedule pads 6 s past the duration (ADR-0008), so six
minutes is the ceiling this solver accepts."""


def sdedit(
    model,
    audio: np.ndarray,
    sample_rate: int,
    *,
    noise_level: float,
    steps: int,
    cfg_scale: float,
    prompt: str = "",
    seed: int = 0,
) -> np.ndarray:
    """Restore `(channels, samples)` audio by partial regeneration (SDEdit).

    The model encodes the input into SAME latents, mixes in noise at
    `noise_level` — the schedule's starting sigma, so latents begin at
    `input * (1 - noise_level) + noise * noise_level` — and denoises from
    there in `steps` iterations. 1.0 regenerates everything (the input is
    ignored); small values keep the input and reduce to the bare autoencoder
    round-trip.
    """
    if sample_rate != PRIOR_SAMPLE_RATE:
        raise ValueError(
            f"SDEdit expects {PRIOR_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    if not 0.0 < noise_level <= 1.0:
        raise ValueError(f"noise_level must be in (0, 1], got {noise_level}.")
    if steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps}.")
    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains NaN or infinite values.")
    seconds = audio.shape[-1] / sample_rate
    if seconds > SDEDIT_MAX_SECONDS:
        raise ValueError(
            f"Input is {seconds:.1f} s; this solver takes whole files up to "
            f"{SDEDIT_MAX_SECONDS:.0f} s. Cut the file first."
        )

    # A 2-D numpy array would be read as (samples, channels) upstream, so the
    # input goes in as a torch tensor, which passes through untransposed.
    tensor = torch.from_numpy(np.ascontiguousarray(audio))
    batch = model.generate(
        prompt=prompt,
        # One extra sample so the output truncation's int() floor cannot
        # shave the input's last sample.
        duration=(audio.shape[-1] + 1) / sample_rate,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        init_audio=(sample_rate, tensor),
        init_noise_level=noise_level,
        # The default sample_size silently truncates anything past 120 s
        # (ADR-0008). 10 s of headroom covers the schedule's 6 s duration
        # padding plus alignment rounding; the model itself takes the min.
        sample_size=audio.shape[-1] + 10 * sample_rate,
    )
    out = batch.squeeze(0).float().cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(out[:, : audio.shape[-1]])
