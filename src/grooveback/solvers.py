"""Restoring audio with a model at inference time.

A solver takes degraded audio and proposes a clean version — no surrogate
model, no fine-tuning, the damage is handled entirely at inference. SDEdit
was the first (ADR-0009, rejected); `roundtrip` and `latent_sub` run degraded
audio through an autoencoder's latent space (ADR-0011).

Every solver returns `(channels, samples)` float32 at the input's length and
integrated loudness — the waveform baselines' output tracks the input's level
by construction, while a decoder's native level drifts.
"""

from __future__ import annotations

import numpy as np
import torch

from grooveback import audio as ga
from grooveback import latents as gl
from grooveback.priors import PRIOR_SAMPLE_RATE

SDEDIT_MAX_SECONDS = 360.0
"""Whole files, one model call. The `seconds_total` conditioner is calibrated
up to 384 s and the schedule pads 6 s past the duration (ADR-0008), so six
minutes is the ceiling this solver accepts."""


def _fit_to_input(out: np.ndarray, samples: int) -> np.ndarray:
    """Trim a decoder overrun, or zero-pad a resampling shortfall, to `samples`."""
    out = out[:, :samples]
    if out.shape[-1] < samples:
        out = np.pad(out, ((0, 0), (0, samples - out.shape[-1])))
    return np.ascontiguousarray(out.astype(np.float32))


def _match_input_loudness(out: np.ndarray, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Gain `out` to the input's integrated loudness (peaks may pass full scale).

    Integrated loudness needs at least the meter's 400 ms block, and digital
    silence (or DC) measures -inf; in both cases the output stays at its
    native level instead of scaling by infinity.
    """
    if out.shape[-1] > int(0.4 * sample_rate):
        target = ga.loudness(audio, sample_rate)
        level = ga.loudness(out, sample_rate)
        if np.isfinite(target) and np.isfinite(level):
            out = out * np.float32(10.0 ** ((target - level) / 20.0))
    return out


def roundtrip(
    model,
    audio: np.ndarray,
    sample_rate: int,
    *,
    ae: str,
    sample: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """`decode(encode(x))` through one autoencoder — the cheapest restoration
    its latent space offers: the decoder invents plausible content where the
    input carries none (ADR-0011).

    `sample=True` draws the encoder's posterior where one exists (εar-VAE,
    εar-VAE2) instead of taking the deterministic encode.
    """
    latents = gl.ae_encode(ae, audio, sample_rate, model, sample=sample, seed=seed)
    out = gl.ae_decode(ae, latents, model)
    out = _fit_to_input(out, audio.shape[-1])
    return _match_input_loudness(out, audio, sample_rate)


def latent_sub(
    model,
    audio: np.ndarray,
    sample_rate: int,
    *,
    ae: str,
    damage: np.ndarray,
) -> np.ndarray:
    """Subtract a mean MP3 damage direction from the input's latents, then decode.

    `damage` is a `(channels,)` vector measured on a donor track:
    `mean_t(encode(mp3(donor)) - encode(donor))` at the input's bitrate — it
    points clean → damaged, so subtraction moves toward clean (ADR-0011).
    """
    latents = gl.ae_encode(ae, audio, sample_rate, model)
    out = gl.ae_decode(ae, gl.subtract_damage(latents, damage), model)
    out = _fit_to_input(out, audio.shape[-1])
    return _match_input_loudness(out, audio, sample_rate)


def sdedit(
    model,
    audio: np.ndarray,
    sample_rate: int,
    *,
    noise_level: float,
    steps: int,
    cfg_scale: float,
    theta: float | None = None,
    prompt: str = "",
    negative_prompt: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Restore `(channels, samples)` audio by partial regeneration (SDEdit).

    The model encodes the input into SAME latents, mixes in noise at
    `noise_level` — latents begin at
    `input * (1 - noise_level) + noise * noise_level` — and denoises from
    the schedule time `theta` down to 0 in `steps` iterations. With `theta`
    unset (classic SDEdit) the schedule starts at `noise_level` itself:
    1.0 regenerates everything, small values reduce to the bare autoencoder
    round-trip.

    Setting `theta` above `noise_level` makes the model remove more than was
    added — with `noise_level=0` nothing is added at all, and the θ of
    denoising budget is spent on what already deviates from clean music: the
    damage itself (ADR-0009). Deterministic on the -base checkpoints, whose
    euler sampler draws no noise of its own.

    Output comes back at the input's integrated loudness (gain only, so peaks
    can pass full scale) — the waveform baselines' output tracks the input's
    level by construction, while the decoder's native level drifts with
    `noise_level`.
    """
    if sample_rate != PRIOR_SAMPLE_RATE:
        raise ValueError(
            f"SDEdit expects {PRIOR_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    if theta is None:
        if not 0.0 < noise_level <= 1.0:
            raise ValueError(
                f"noise_level must be in (0, 1] when theta is not set, "
                f"got {noise_level}."
            )
    else:
        if not 0.0 < theta <= 1.0:
            raise ValueError(f"theta must be in (0, 1], got {theta}.")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"noise_level must be in [0, 1], got {noise_level}.")
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
        negative_prompt=negative_prompt,
        # One extra sample so the output truncation's int() floor cannot
        # shave the input's last sample.
        duration=(audio.shape[-1] + 1) / sample_rate,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        init_audio=(sample_rate, tensor),
        init_noise_level=theta if theta is not None else noise_level,
        init_mix_level=noise_level,
        # The default sample_size silently truncates anything past 120 s
        # (ADR-0008). 10 s of headroom covers the schedule's 6 s duration
        # padding plus alignment rounding; the model itself takes the min.
        sample_size=audio.shape[-1] + 10 * sample_rate,
    )
    out = batch.squeeze(0).float().cpu().numpy().astype(np.float32)
    out = np.ascontiguousarray(out[:, : audio.shape[-1]])
    return _match_input_loudness(out, audio, sample_rate)
