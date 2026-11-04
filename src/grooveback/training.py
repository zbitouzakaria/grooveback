"""LoRA fine-tuning of Stable Audio 3 to restore MP3s.

The model learns to turn the latents of an MP3 into the latents of the clean
master. Everything is built on Stable Audio 3's own code:

- The MP3 enters through a new per-frame input, `mp3_latents`, made with
  SA3's `modular_local_cond` mechanism: in every transformer block a small
  two-layer network maps MP3 frame i to a vector that is added to token i.
  Its last layer starts at zero, so before training the model behaves
  exactly like base SA3 (the ControlNet recipe: new zero-initialised layers
  next to a frozen pretrained model).
- A LoRA on the frozen transformer lets its attention learn to use that
  input. The new input layers train in full.
- The loss is SA3's own rectified-flow objective, with the clean latents as
  the target. The inpainting input says "generate everything".

The comments follow one worked example: small-music-base, batch 8, 10 s
windows. Its autoencoder, SAME-S, turns 4096 samples into one latent frame
of 256 channels, so a 10 s window is 107 frames, which hold 9.94 s.
"""

from __future__ import annotations

import copy
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

MP3_INPUT = "mp3_latents"
"""Conditioning id of the new per-frame input that carries the MP3."""


# --- The model ---------------------------------------------------------------


def base_checkpoint(variant: str) -> tuple[dict, dict]:
    """The released config and weights of a Stable Audio 3 base checkpoint.

    Kept on its own so tests can swap in a tiny model. The -base weights are
    open, but their config loads the T5Gemma text encoder from the gated
    stabilityai/stable-audio-3-small-music (or -medium) repo: accept that
    licence on Hugging Face and export HF_TOKEN.
    """
    from stable_audio_3.model_configs import base_models

    if variant not in base_models:
        raise ValueError(f"variant must be one of {sorted(base_models)}, got {variant!r}.")
    config_path, weights_path = base_models[variant].resolve()
    return json.loads(Path(config_path).read_text()), load_file(weights_path)


def load_model(variant: str, *, device: torch.device):
    """Base SA3 with the zero-initialised MP3 input, frozen, in eval mode.

    Refuses a checkpoint that leaves any base weight unfilled. SA3's own
    loader only prints a line per skipped key, so a mismatched checkpoint
    would train from random weights without an error.
    """
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.loading_utils import remap_state_dict_keys

    config, weights = base_checkpoint(variant)
    config = copy.deepcopy(config)
    diffusion = config["model"]["diffusion"]
    # The new input has the latents' width: 256 channels per frame.
    latent_channels = diffusion["config"]["io_channels"]
    diffusion["modular_local_cond_configs"] = [{"id": MP3_INPUT, "dim": latent_channels}]

    model = create_diffusion_cond_from_config(config)
    weights = remap_state_dict_keys(weights, model.state_dict())
    missing, unexpected = model.load_state_dict(weights, strict=False)
    # The new input is the only thing the checkpoint may lack.
    missing_base = [key for key in missing if ".modular_local_embeds." not in key]
    if missing_base or unexpected:
        raise RuntimeError(
            f"{variant}: {len(missing_base)} base weights missing and {len(unexpected)} "
            "unexpected in the checkpoint; it does not match the model."
        )
    return model.to(device).eval().requires_grad_(False)


def lora_config(*, rank: int, lora_alpha: float) -> dict:
    """What the LoRA looks like, stored in every checkpoint so a reload can check it."""
    # The new input trains in full, so it gets no LoRA of its own.
    return {
        "rank": rank,
        "alpha": lora_alpha,
        "adapter_type": "lora",
        "exclude": ["modular_local_embeds"],
    }


def add_lora(model, config: dict) -> list[torch.nn.Parameter]:
    """Put a LoRA on the frozen transformer; return every trainable weight.

    A LoRA keeps a frozen weight W and learns W + (alpha / rank) * B @ A,
    with A of shape (rank, fan_in) and B of shape (fan_out, rank). B starts
    at zero, so the LoRA starts as a no-op. On small-music-base at rank 16, a
    1024 x 1024 attention projection gets 2 * 16 * 1024 = 32,768 trainable
    weights next to its 1,048,576 frozen ones.
    """
    from stable_audio_3.models import lora as sa3_lora

    kinds = {
        torch.nn.Linear: sa3_lora.LoRAParametrization.from_linear,
        torch.nn.Conv1d: sa3_lora.LoRAParametrization.from_conv1d,
    }
    layers = {
        kind: {
            "weight": partial(
                build,
                rank=config["rank"],
                lora_alpha=config["alpha"],
                adapter_type=config["adapter_type"],
            )
        }
        for kind, build in kinds.items()
    }
    sa3_lora.add_lora(model.model, layers, exclude=config["exclude"])
    new_input = [
        param for name, param in model.model.named_parameters() if ".modular_local_embeds." in name
    ]
    for param in new_input:
        param.requires_grad_(True)
    return list(sa3_lora.get_lora_params(model.model)) + new_input


class DecoderSkips(torch.nn.Module):
    """Two linear maps that carry the MP3 into SAME's decoder (ADR-0018).

    `queries` maps the encoder's first layer, run on the MP3, onto the decoder's
    query tokens. `patches` maps the MP3's patches onto the decoder's output
    patches. Both start at zero, so the decoder starts unchanged.
    """

    def __init__(self, *, features: int, tokens: int, patches: int):
        super().__init__()
        self.queries = torch.nn.Linear(features, tokens)
        self.patches = torch.nn.Conv1d(patches, patches, kernel_size=1)
        for layer in (self.queries, self.patches):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


def add_skips(model) -> list[torch.nn.Parameter]:
    """Put zero-initialised skips on the model; return their weights.

    They sit next to `model.model`, not in it, so a LoRA checkpoint without
    skips still loads: `_trainable_state` only reads `model.model`. The decoder
    stays in fp32, as in the plain decode, because in bf16 it lands 21 to 23 dB
    SNR from it on SAME-S. Its block recomputes its layers in the backward pass
    to make room for that.
    """
    autoencoder = model.pretransform.model
    block = autoencoder.decoder.layers[-1]
    block.checkpointing = True
    model.skips = DecoderSkips(
        features=autoencoder.encoder.layers[0].out_channels,
        tokens=block.in_channels,
        patches=block.out_channels,
    ).to(_device(model))
    return list(model.skips.parameters())


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _seconds(model, frames: int) -> float:
    """How long `frames` latent frames last: 107 frames of 4096 samples are 9.94 s."""
    return frames * model.pretransform.downsampling_ratio / model.sample_rate


@torch.no_grad()
def encode(model, audio: np.ndarray) -> torch.Tensor:
    """Audio (2, samples) at the model's rate to latents (channels, frames).

    A 4-minute song, (2, 10_584_000), becomes (256, 2_584) on SAME-S, of which
    the first 2_583 frames hold real audio (`samples // downsampling_ratio`).
    The audio is padded to whole frames first: SA3's chunked encoder places
    its last chunk off the frame grid otherwise, which shifts the song's last
    112 frames in time.
    """
    batch = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(_device(model))
    batch = F.pad(batch, (0, -batch.shape[-1] % model.pretransform.downsampling_ratio))
    return model.pretransform.encode(batch)[0]


# --- One step ----------------------------------------------------------------


def _conditioning(model, *, mp3: torch.Tensor, inpaint_mask_value: float) -> dict:
    """Everything the model is told besides the noised latents, for a batch of MP3 windows."""
    batch, channels, frames = mp3.shape
    device = mp3.device
    with torch.no_grad():
        # Empty prompt; seconds_total tells the model how long the clip is.
        clip = {"prompt": "", "seconds_total": _seconds(model, frames)}
        cond = model.conditioner([clip] * batch, device)
    # Inpainting input: nothing is given, so the model generates every frame.
    # 0 is upstream's convention (stable_audio_3/model.py generate); SA3's MLX
    # trainer (optimized/mlx/scripts/lora_train_mlx.py) uses an all-ones mask
    # with empty content instead, hence the config value.
    cond["inpaint_mask"] = [torch.full((batch, 1, frames), inpaint_mask_value, device=device)]
    cond["inpaint_masked_input"] = [torch.zeros_like(mp3)]
    cond[MP3_INPUT] = [mp3]
    return cond


def predict_velocity(
    model, noised: torch.Tensor, t: torch.Tensor, *, mp3: torch.Tensor, inpaint_mask_value: float
) -> torch.Tensor:
    """The model's estimate of noise - clean, for a batch of noised latents and their MP3.

    noised (8, 256, 107), t (8,), mp3 (8, 256, 107) -> (8, 256, 107).
    """
    cond = _conditioning(model, mp3=mp3, inpaint_mask_value=inpaint_mask_value)
    # Every frame is real audio: no padding to mask out of attention.
    padding_mask = torch.ones(
        noised.shape[0], noised.shape[-1], dtype=torch.bool, device=noised.device
    )
    with torch.autocast(noised.device.type, dtype=torch.bfloat16, enabled=noised.is_cuda):
        return model(noised, t, cond=cond, padding_mask=padding_mask).float()


def training_step(
    model, clean: torch.Tensor, *, mp3: torch.Tensor, inpaint_mask_value: float
) -> torch.Tensor:
    """Rectified-flow loss for one batch of (clean, MP3) windows.

    Training shows the model a clean window mixed with noise and asks for the
    direction from the clean window to the noise. At sampling time it starts
    from pure noise and follows that direction backwards, step by step.
    """
    # One noise level per clip, t in (0, 1): 0 is the clean window, 1 is pure
    # noise. Logit-normal draws favour the middle. SA3's shift then moves every
    # draw toward noise (t = 0.5 becomes 0.62); below 256 frames (23.8 s) it
    # moves every window by the same amount.
    t = torch.sigmoid(torch.randn(clean.shape[0], device=clean.device))
    t = model.dist_shift.shift(t, clean.shape[-1])
    noise = torch.randn_like(clean)
    noised = clean * (1 - t[:, None, None]) + noise * t[:, None, None]
    velocity = predict_velocity(model, noised, t, mp3=mp3, inpaint_mask_value=inpaint_mask_value)
    return F.mse_loss(velocity, noise - clean)


def decode_with_skips(model, latents: torch.Tensor, *, mp3_audio: torch.Tensor) -> torch.Tensor:
    """Decode latents to audio with the decoder also reading the MP3 (ADR-0018).

    latents (8, 256, 107), mp3_audio (8, 2, 438_272) -> audio (8, 2, 438_272).
    This repeats AudioAutoencoder.decode, which decodes a window under 128 frames
    in one piece, because SAMEDecoder.forward cannot pass tokens to its block.
    """
    autoencoder = model.pretransform.model
    block = autoencoder.decoder.layers[-1]
    # 256 samples of 2 channels per patch, 16 patches per latent frame: (8, 512, 1712).
    patches = autoencoder.pretransform.encode(mp3_audio)
    features = autoencoder.encoder.layers[0].mapping(patches)  # (8, 768, 1712)
    x = autoencoder.bottleneck.decode(latents * model.pretransform.scale)
    for layer in autoencoder.decoder.layers[:-1]:
        x = layer(x)  # (8, 768, 107)
    tokens = model.skips.queries(features.transpose(1, 2))  # (8, 1712, 768)
    # The block wants a token for every patch of its frames, in whole chunks without a
    # sliding window. Those frames can outnumber the audio's patches: SAME-S's encoder
    # pads 107 frames to 108, and a song rarely fills its last frame.
    chunk = 1 if block.sliding_window_seq else block.chunk_size // block.stride
    frames = x.shape[-1] + -x.shape[-1] % chunk
    tokens = F.pad(tokens, (0, 0, 0, frames * block.stride - tokens.shape[1]))
    # Given tokens, the block no longer adds its noise to its queries, so it is added here.
    tokens = tokens + block.mask_noise * torch.randn_like(tokens)
    decoded = block(x, override_new_tokens=tokens)[..., : patches.shape[-1]]
    decoded = decoded + model.skips.patches(patches)
    audio = autoencoder.pretransform.decode(decoded)
    return torch.tanh(audio) if autoencoder.soft_clip else audio


def audio_loss(audio: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
    """L1 on the waveform plus L1 between STFT magnitudes at three resolutions.

    audio, master (8, 2, 438_272) -> a scalar. `normalized` keeps the magnitudes
    on the waveform's scale, whatever the FFT size.
    """
    loss = F.l1_loss(audio, master)
    for n_fft in (512, 2048, 8192):
        window = torch.hann_window(n_fft, device=audio.device)
        audio_stft, master_stft = (
            torch.stft(x.flatten(0, 1), n_fft, window=window, normalized=True, return_complex=True)
            for x in (audio, master)
        )
        loss = loss + F.l1_loss(audio_stft.abs(), master_stft.abs())
    return loss


def last_noise_level(model, *, steps: int, frames: int) -> float:
    """The noise level of the sampler's last step: 0.138 at 50 steps on medium-base.

    A rectified-flow sampler's last step goes from this level to 0, so what it
    returns is `noised - level * velocity`, the estimate `audio_step` decodes.
    """
    from stable_audio_3.inference.sampling import build_schedule

    levels = build_schedule(
        steps=steps,
        sigma_max=1.0,
        dist_shift=model.sampling_dist_shift,
        effective_seq_len=None,
        fallback_seq_len=frames,
        include_endpoint=True,
        device="cpu",
    )
    return float(levels[-2])


def audio_step(
    model,
    clean: torch.Tensor,
    *,
    mp3: torch.Tensor,
    master: torch.Tensor,
    mp3_audio: torch.Tensor,
    level: float,
    inpaint_mask_value: float,
) -> torch.Tensor:
    """Audio loss for one batch at noise level `level`, the second pass of ADR-0018.

    The model's estimate of the clean latents is decoded with the skips and
    compared with the master, so the gradient reaches the skips and the LoRA.
    """
    t = torch.full((clean.shape[0],), level, device=clean.device)
    noised = clean * (1 - level) + torch.randn_like(clean) * level
    velocity = predict_velocity(model, noised, t, mp3=mp3, inpaint_mask_value=inpaint_mask_value)
    audio = decode_with_skips(model, noised - level * velocity, mp3_audio=mp3_audio)
    return audio_loss(audio, master)


@torch.no_grad()
def validation_loss(
    model,
    dataset: torch.utils.data.Dataset,
    *,
    timesteps: list[float],
    batch_size: int,
    inpaint_mask_value: float,
) -> float:
    """Mean rectified-flow loss over every window, at fixed noise levels.

    The noise comes from the same seed on every call, so two calls differ
    only because the model changed.
    """
    device = _device(model)
    generator = torch.Generator(device=device).manual_seed(0)
    total, count = 0.0, 0
    for clean, mp3, *_ in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        for level in timesteps:
            t = torch.full((clean.shape[0],), level, device=device)
            noise = torch.randn(clean.shape, generator=generator, device=device)
            noised = clean * (1 - level) + noise * level
            velocity = predict_velocity(
                model, noised, t, mp3=mp3, inpaint_mask_value=inpaint_mask_value
            )
            total += F.mse_loss(velocity, noise - clean, reduction="sum").item()
            count += clean.numel()
    return total / count


@torch.no_grad()
def validation_audio_loss(
    model,
    dataset: torch.utils.data.Dataset,
    *,
    level: float,
    batch_size: int,
    inpaint_mask_value: float,
) -> float:
    """Mean audio loss over every window at noise level `level`, decoded with the skips.

    The latents' noise and the decoder's own noise come from the same seed on
    every call, so two calls differ only because the model changed. The seed
    stays inside this call: training's draws carry on as if it had not run.
    """
    device = _device(model)
    total, count = 0.0, 0
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        for clean, mp3, master, mp3_audio in torch.utils.data.DataLoader(
            dataset, batch_size=batch_size
        ):
            loss = audio_step(
                model,
                clean,
                mp3=mp3,
                master=master.to(device),
                mp3_audio=mp3_audio.to(device),
                level=level,
                inpaint_mask_value=inpaint_mask_value,
            )
            total += loss.item() * clean.shape[0]
            count += clean.shape[0]
    return total / count


@torch.no_grad()
def restore(
    model,
    mp3: torch.Tensor,
    *,
    steps: int,
    inpaint_mask_value: float,
    sampler_type: str = "euler",
    mp3_audio: np.ndarray | None = None,
) -> np.ndarray:
    """Generate the clean version of one window of MP3 latents, as audio.

    mp3 (256, 107) -> audio (2, 438_272): 107 frames of 4096 samples, starting
    from seeded noise and denoising in `steps` steps with one of SA3's samplers
    for rectified flow. "euler" (SA3's default) and "dpmpp" call the model once
    per step, "rk4" four times; all three are deterministic given the noise.
    A model with skips also reads `mp3_audio`, the window's coded audio, when
    it decodes.
    """
    from stable_audio_3.inference.sampling import sample_diffusion

    skips = hasattr(model, "skips")
    if skips and mp3_audio is None:
        raise ValueError("This model has decoder skips: give restore the window's mp3_audio.")
    device = _device(model)
    channels, frames = mp3.shape
    cond = _conditioning(model, mp3=mp3[None], inpaint_mask_value=inpaint_mask_value)
    generator = torch.Generator(device=device).manual_seed(0)
    audio = sample_diffusion(
        model=model.model,
        noise=torch.randn((1, channels, frames), generator=generator, device=device),
        cond_inputs=model.get_conditioning_inputs(cond),
        diffusion_objective=model.diffusion_objective,
        steps=steps,
        cfg_scale=1.0,
        sample_rate=model.sample_rate,
        pretransform=model.pretransform,
        padding_mask=torch.ones(1, frames, dtype=torch.bool, device=device),
        dist_shift=model.sampling_dist_shift,
        sampler_type=sampler_type,
        disable_tqdm=True,
        decode=not skips,
    )
    if skips:
        audio = decode_with_skips(
            model, audio, mp3_audio=torch.from_numpy(mp3_audio)[None].to(device)
        )
    return audio[0].float().cpu().numpy()


# --- Checkpoints -------------------------------------------------------------


def _trainable_state(model) -> dict[str, torch.Tensor]:
    """The LoRA and new-input weights, the only ones training changes."""
    from stable_audio_3.models.lora import name_is_lora

    return {
        name: tensor
        for name, tensor in model.model.state_dict().items()
        if name_is_lora(name) or ".modular_local_embeds." in name
    }


def save_checkpoint(directory: Path, model, optimizer, *, config: dict, step: int, epoch: int):
    """Write `lora.safetensors` (trained weights, fp32) and `state.pt` (optimizer, step, epoch),
    and `skips.safetensors` for a model with skips.

    The weights stay fp32: SA3's own save_lora_safetensors rounds them to
    fp16, which would make a resumed run start from slightly different
    weights.
    """
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: tensor.contiguous().cpu() for name, tensor in _trainable_state(model).items()}
    save_file(weights, directory / "lora.safetensors", metadata={"lora_config": json.dumps(config)})
    if hasattr(model, "skips"):
        skips = {name: tensor.cpu() for name, tensor in model.skips.state_dict().items()}
        save_file(skips, directory / "skips.safetensors")
    state = {"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch}
    torch.save(state, directory / "state.pt")


def load_lora_weights(model, path: Path, *, config: dict) -> None:
    """Load weights written by `save_checkpoint` into a model with the same LoRA.

    Every key must match, and the LoRA config must be the one in the file: a
    different alpha would silently rescale every update.
    """
    from stable_audio_3.models.lora import load_lora_checkpoint

    weights, saved_config = load_lora_checkpoint(path)
    if saved_config != config:
        raise ValueError(f"{path} was trained with {saved_config}, the config asks for {config}.")
    expected = set(_trainable_state(model))
    if set(weights) != expected:
        raise ValueError(
            f"{path}: {len(expected - set(weights))} weights missing, "
            f"{len(set(weights) - expected)} unexpected."
        )
    model.model.load_state_dict(weights, strict=False)


def load_skips(model, path: Path) -> None:
    """Load skips written by `save_checkpoint` into a model that has skips."""
    model.skips.load_state_dict(load_file(path))


def load_trained_model(variant: str, checkpoint: Path, *, device: torch.device):
    """Base SA3 plus the LoRA and MP3 input saved in `checkpoint/lora.safetensors`,
    and the skips of `checkpoint/skips.safetensors` if it has any, ready to
    restore. The LoRA's shape is read from the file itself."""
    from stable_audio_3.models.lora import load_lora_checkpoint

    path = Path(checkpoint) / "lora.safetensors"
    _, config = load_lora_checkpoint(path)
    model = load_model(variant, device=device)
    add_lora(model, config)
    load_lora_weights(model, path, config=config)
    skips = Path(checkpoint) / "skips.safetensors"
    if skips.exists():
        add_skips(model)
        load_skips(model, skips)
    return model


def load_training_state(directory: Path, optimizer) -> tuple[int, int]:
    """Restore Adam's running averages and the step count; return (step, epoch).

    torch's Optimizer.load_state_dict also restores the saved lr, betas and
    weight decay, so a new learning rate in the config would be silently
    ignored. The optimizer was built from the config, so its own defaults
    are written back over them: the config always wins.
    """
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group.update(optimizer.defaults)
    return state["step"], state["epoch"]
