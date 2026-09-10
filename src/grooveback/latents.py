"""Autoencoder latent spaces: encode, decode, round-trip, damage arithmetic.

SAME is the space any future prior will live in (Stable Audio 3 generates into
it); εar-VAE, εar-VAE2 and CoDiCodec are the other open music autoencoders the
round-trip benchmark compares against (ADR-0011). Every wrapper speaks the same
contract — 44.1 kHz `(channels, samples)` float32 audio in and out, latents as
`(channels, frames)` float32 — and resampling to a model's native rate is the
wrapper's private business.

Load once with `load_ae` and pass the handle around: loading always costs more
than encoding a clip.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from grooveback import audio as ga
from grooveback.baselines import select_device

_REPO = Path(__file__).resolve().parents[2]

MAX_FRAME_SLACK = 2
"""Latent-grid tolerance: decoders and codecs round lengths by up to a frame
or two at the boundary; a larger frame-count mismatch is a real grid
disagreement, not an overrun, and is refused."""


def _require_finite(array: np.ndarray, what: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{what} contains NaN or infinite values.")


# --- SAME ------------------------------------------------------------------

SAME_SAMPLE_RATE = 44_100
SAME_CHECKPOINTS = {"same-s": "stabilityai/SAME-S", "same-l": "stabilityai/SAME-L"}
"""Only two variants exist. Stable Audio 3 small generates into `same-s`."""


def load_same(variant: str = "same-s", device: str = "auto"):
    """Load a SAME autoencoder from its official Hugging Face checkpoint."""
    if variant not in SAME_CHECKPOINTS:
        raise ValueError(
            f"variant must be one of {sorted(SAME_CHECKPOINTS)}, got {variant!r}."
        )
    from stable_audio_3.loading_utils import load_autoencoder

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
    """Audio `(channels, samples)` to SAME latents `(256, frames)`."""
    if sample_rate != SAME_SAMPLE_RATE:
        raise ValueError(f"SAME expects {SAME_SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    _require_finite(audio, "Input audio")
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(device)
    with torch.inference_mode():
        latents = model.encode_audio(batch)
    return latents.squeeze(0).float().cpu().numpy()


def decode(latents: np.ndarray, model) -> np.ndarray:
    """SAME latents `(256, frames)` back to audio `(channels, samples)`.

    Output length is the latent count times 4096, so it can overrun the
    original by up to one frame. Callers comparing against a source should
    trim both to the shorter length.
    """
    _require_finite(latents, "Latents")
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.ascontiguousarray(latents)).unsqueeze(0).to(device)
    with torch.inference_mode():
        audio = model.decode_audio(batch)
    out = audio.squeeze(0).float().cpu().numpy()
    _require_finite(out, "Decoded audio")
    return out


def roundtrip(audio: np.ndarray, sample_rate: int, model) -> np.ndarray:
    """`decode(encode(audio))` — what survives a trip through the latent space.

    Not an identity, and not even close on bandlimited input: the decoder
    re-realises phase and invents high-frequency content where the input has
    none.
    """
    return decode(encode(audio, sample_rate, model), model)


# --- εar-VAE ---------------------------------------------------------------
# Wang et al., arXiv:2509.14912. Vendored as a submodule because it ships no
# installable package; its model code needs dac and alias_free_torch, which
# resolve under the project pins (dependency group `aes`).

EARVAE_SAMPLE_RATE = 44_100
EARVAE_REPO = _REPO / "third_party" / "earvae"
EARVAE_CHECKPOINT = ("earlab/EAR_VAE", "pretrained_weight/ear_vae_44k.pyt")
"""The 44.1 kHz v1 weights (the paper's model). The repo also ships a 48 kHz
v2 (`pretrained_weight/ear_vae_v2_48k.pyt` + `config/ear_vae_v2.json`)."""


def _earvae_import():
    """Import the vendored εar-VAE model class.

    The repo's package is the generic top-level name `model`, so its root goes
    at the front of `sys.path` — nothing else in this stack imports a bare
    `model`, but keep it that way.
    """
    if not (EARVAE_REPO / "model").is_dir():
        raise FileNotFoundError(
            f"εar-VAE submodule missing at {EARVAE_REPO}. "
            "Run: git submodule update --init --recursive"
        )
    if str(EARVAE_REPO) not in sys.path:
        sys.path.insert(0, str(EARVAE_REPO))
    return importlib.import_module("model.ear_vae").EAR_VAE


def load_earvae(device: str = "auto"):
    """Load εar-VAE with the repo's config and the official 44.1 kHz weights.

    `torch.load` runs with `weights_only=False`, matching the upstream loader —
    arbitrary deserialization, the usual bargain with research checkpoints.
    """
    ear_vae_cls = _earvae_import()
    config = json.loads((EARVAE_REPO / "config" / "model_config.json").read_text())
    weights_path = hf_hub_download(*EARVAE_CHECKPOINT)
    model = ear_vae_cls(model_config=config)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(select_device(device)).eval()


def _earvae_encode(audio, sample_rate, model):
    """Audio to εar-VAE latents `(64, frames)` — always the posterior mean.

    A sampled-encode variant was benchmarked once and removed: the posterior
    is collapsed, sampled and mean encodes agree to 0.01 dB (ADR-0011).
    """
    if sample_rate != EARVAE_SAMPLE_RATE:
        raise ValueError(f"εar-VAE expects {EARVAE_SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    _require_finite(audio, "Input audio")
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(device)
    with torch.inference_mode():
        latents = model.encode(batch, use_sample=False)
    return latents.squeeze(0).float().cpu().numpy()


def _earvae_decode(latents, model):
    _require_finite(latents, "Latents")
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.ascontiguousarray(latents)).unsqueeze(0).to(device)
    with torch.inference_mode():
        audio = model.decode(batch)
    out = audio.squeeze(0).float().cpu().numpy()
    _require_finite(out, "Decoded audio")
    return out


# --- εar-VAE2 --------------------------------------------------------------
# Wang, Dai, Xu, arXiv:2608.19843. Natively 48 kHz; the wrapper resamples
# 44.1 kHz in and out, so the whole registry keeps one external rate.

EARVAE2_SAMPLE_RATE = 48_000
EARVAE2_REPO = _REPO / "third_party" / "earvae2"
EARVAE2_CHECKPOINT = ("earlab/EAR_VAE2", "weights/ear_vae2.pt")
EARVAE2_CHUNK = {"chunked": True, "chunk_size": 512, "overlap": 16}
"""The upstream inference defaults; chunk sizes are latent frames (40 ms each)."""


def _earvae2_import():
    if not (EARVAE2_REPO / "ear_vae2").is_dir():
        raise FileNotFoundError(
            f"εar-VAE2 submodule missing at {EARVAE2_REPO}. "
            "Run: git submodule update --init --recursive"
        )
    if str(EARVAE2_REPO) not in sys.path:
        sys.path.insert(0, str(EARVAE2_REPO))
    return importlib.import_module("ear_vae2").EarVAE2


def load_earvae2(device: str = "auto"):
    """Load εar-VAE2 with the repo's shipped config and open weights.

    The open weights are retrained on public data, not the paper's corpus.
    Checkpoint handling mirrors the upstream loader exactly: `weights_only=False`
    (the research-checkpoint bargain), EMA weights preferred, `strict=False`.
    """
    ear_vae2_cls = _earvae2_import()
    config = json.loads((EARVAE2_REPO / "configs" / "ear_vae2.json").read_text())
    config = config["model"]["gen"]["config"]
    weights_path = hf_hub_download(*EARVAE2_CHECKPOINT)
    model = ear_vae2_cls(config)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "gen_ema" in ckpt:
        state = {
            k[len("ema_model."):]: v
            for k, v in ckpt["gen_ema"].items()
            if k.startswith("ema_model.")
        }
    elif "gen" in ckpt:
        state = ckpt["gen"]
    else:
        state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"earvae2: state dict mismatch — {len(missing)} missing, "
              f"{len(unexpected)} unexpected")
    return model.to(select_device(device)).eval()


def _earvae2_encode(audio, sample_rate, model):
    """Audio to εar-VAE2 latents `(128, frames)`, deterministic encode."""
    if sample_rate != SAME_SAMPLE_RATE:
        raise ValueError(
            f"the registry expects {SAME_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    _require_finite(audio, "Input audio")
    device = next(model.parameters()).device
    native = ga.resample(audio, sr_in=sample_rate, sr_out=EARVAE2_SAMPLE_RATE)
    batch = torch.from_numpy(np.ascontiguousarray(native)).unsqueeze(0).to(device)
    pad = (-batch.shape[-1]) % model.samples_per_latent
    if pad:
        batch = torch.nn.functional.pad(batch, (0, pad))
    with torch.inference_mode():
        latents = model.encode_audio(batch, **EARVAE2_CHUNK, deterministic=True)
    return latents.squeeze(0).float().cpu().numpy()


def _earvae2_decode(latents, model):
    _require_finite(latents, "Latents")
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.ascontiguousarray(latents)).unsqueeze(0).to(device)
    with torch.inference_mode():
        native = model.decode_audio(batch, **EARVAE2_CHUNK)
    out = native.squeeze(0).float().cpu().numpy()
    out = ga.resample(out, sr_in=EARVAE2_SAMPLE_RATE, sr_out=SAME_SAMPLE_RATE)
    _require_finite(out, "Decoded audio")
    return out


# --- CoDiCodec -------------------------------------------------------------
# Pasini et al., arXiv:2509.09836, pip package. The shipped checkpoint runs at
# 48 kHz; latents come back as [frames, 16, 64] and are flattened to
# (1024, frames) so the frame axis sits last like every other model here.

CODICODEC_SAMPLE_RATE = 48_000
CODICODEC_MAX_BATCH = 4
"""Patches decoded at once. The package defaults (64 encode / 32 decode) are
sized for large GPUs; 180 s of latents OOMs a 22 GiB L4 at the default."""


class CodiHandle(NamedTuple):
    """A loaded CoDiCodec plus the latent grid measured at load time."""

    encdec: object
    tokens: int
    dim: int


def load_codicodec(device: str = "auto"):
    """Load CoDiCodec and probe its latent grid on a short silence.

    The import flips global torch backend flags (TF32, cudnn benchmark), which
    would silently change every other model's numerics — they are snapshotted
    and restored. The probe also enforces stereo output: a mono model would
    invalidate the whole comparison (ADR-0011).
    """
    flags = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    from codicodec import EncoderDecoder

    (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    ) = flags
    encdec = EncoderDecoder(device=select_device(device))
    probe = np.zeros((2, CODICODEC_SAMPLE_RATE), dtype=np.float32)
    with torch.inference_mode():
        latents = encdec.encode(probe, max_batch_size=CODICODEC_MAX_BATCH)
    if latents.ndim != 3:
        raise RuntimeError(
            f"codicodec returned {latents.ndim}-D latents; expected [frames, tokens, dim]."
        )
    decoded = _codi_to_channels_first(
        encdec.decode(latents, max_batch_size=CODICODEC_MAX_BATCH, denoising_steps=1)
    )
    if decoded.shape[0] != 2:
        raise RuntimeError(
            f"codicodec decoded {decoded.shape[0]} channels; the benchmark needs stereo."
        )
    return CodiHandle(encdec, tokens=int(latents.shape[-2]), dim=int(latents.shape[-1]))


def _codi_to_channels_first(waveform) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.float().cpu().numpy()
    if waveform.ndim != 2:
        raise RuntimeError(f"codicodec decoded shape {waveform.shape}; expected 2-D audio.")
    if waveform.shape[0] > waveform.shape[1]:  # (samples, channels) -> (channels, samples)
        waveform = waveform.T
    return np.ascontiguousarray(waveform.astype(np.float32))


def _codicodec_encode(audio, sample_rate, handle):
    """Audio to CoDiCodec latents `(tokens*dim, frames)` — joint stereo."""
    import einops

    if sample_rate != SAME_SAMPLE_RATE:
        raise ValueError(f"the registry expects {SAME_SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    _require_finite(audio, "Input audio")
    native = ga.resample(audio, sr_in=sample_rate, sr_out=CODICODEC_SAMPLE_RATE)
    with torch.inference_mode():
        latents = handle.encdec.encode(
            np.ascontiguousarray(native), max_batch_size=CODICODEC_MAX_BATCH
        )
    if isinstance(latents, torch.Tensor):
        latents = latents.float().cpu().numpy()
    return np.ascontiguousarray(einops.rearrange(latents, "t l d -> (l d) t"))


def _codicodec_decode(latents, handle):
    import einops

    _require_finite(latents, "Latents")
    grid = einops.rearrange(latents, "(l d) t -> t l d", l=handle.tokens, d=handle.dim)
    # The decoder synthesizes from noise; a fixed seed keeps renders repeatable.
    torch.manual_seed(0)
    with torch.inference_mode():
        waveform = handle.encdec.decode(
            torch.from_numpy(np.ascontiguousarray(grid)),
            max_batch_size=CODICODEC_MAX_BATCH,
        )
    out = _codi_to_channels_first(waveform)
    out = ga.resample(out, sr_in=CODICODEC_SAMPLE_RATE, sr_out=SAME_SAMPLE_RATE)
    _require_finite(out, "Decoded audio")
    return out


# --- Registry --------------------------------------------------------------


class _AE(NamedTuple):
    load: object
    encode: object
    decode: object


AUTOENCODERS = {
    "same-l": _AE(lambda device: load_same("same-l", device), encode, decode),
    "earvae": _AE(load_earvae, _earvae_encode, _earvae_decode),
    "earvae2": _AE(load_earvae2, _earvae2_encode, _earvae2_decode),
    "codicodec": _AE(load_codicodec, _codicodec_encode, _codicodec_decode),
}


def _entry(name: str) -> _AE:
    if name not in AUTOENCODERS:
        raise ValueError(f"autoencoder must be one of {sorted(AUTOENCODERS)}, got {name!r}.")
    return AUTOENCODERS[name]


def load_ae(name: str, device: str = "auto"):
    """Load one autoencoder by registry name; the handle goes to ae_encode/ae_decode."""
    return _entry(name).load(device)


def ae_encode(name: str, audio: np.ndarray, sample_rate: int, model) -> np.ndarray:
    """Audio `(channels, samples)` at 44.1 kHz to `(channels, frames)` latents.

    Always the deterministic encode (mean posterior where the model is a
    VAE) — a sampled variant was benchmarked once and removed as redundant
    (ADR-0011).
    """
    return _entry(name).encode(audio, sample_rate, model)


def ae_decode(name: str, latents: np.ndarray, model) -> np.ndarray:
    """Latents `(channels, frames)` back to `(channels, samples)` at 44.1 kHz.

    Decoders round the length up to whole frames, so output can overrun the
    encoded audio; callers trim against their source.
    """
    return _entry(name).decode(latents, model)


# --- Damage arithmetic (ADR-0011) ------------------------------------------


def mean_damage(clean_latents: np.ndarray, degraded_latents: np.ndarray) -> np.ndarray:
    """Mean over frames of (degraded − clean): the damage direction.

    Both encodings come from the same donor audio, so their frame grids must
    agree up to boundary rounding (`MAX_FRAME_SLACK`); anything larger means
    the two latents do not describe the same audio.
    """
    if clean_latents.shape[0] != degraded_latents.shape[0]:
        raise ValueError(
            f"channel counts differ: clean {clean_latents.shape[0]} vs "
            f"degraded {degraded_latents.shape[0]} — different autoencoders?"
        )
    gap = abs(clean_latents.shape[1] - degraded_latents.shape[1])
    if gap > MAX_FRAME_SLACK:
        raise ValueError(
            f"frame counts differ by {gap} (clean {clean_latents.shape[1]}, "
            f"degraded {degraded_latents.shape[1]}); these latents do not "
            "describe the same audio."
        )
    frames = min(clean_latents.shape[1], degraded_latents.shape[1])
    return (degraded_latents[:, :frames] - clean_latents[:, :frames]).mean(axis=1)


def subtract_damage(latents: np.ndarray, damage: np.ndarray) -> np.ndarray:
    """Subtract a `(channels,)` damage direction from every frame of `latents`.

    The direction points clean → damaged, so subtraction moves toward clean.
    """
    if damage.ndim != 1:
        raise ValueError(f"damage must be one vector (channels,), got shape {damage.shape}.")
    if latents.shape[0] != damage.shape[0]:
        raise ValueError(
            f"{latents.shape[0]}-channel latents vs {damage.shape[0]}-channel "
            "damage — measured with a different autoencoder?"
        )
    return latents - damage[:, None]
