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

from grooveback import audio as ga

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


# --- The data ----------------------------------------------------------------


class LatentPairDataset(torch.utils.data.Dataset):
    """Aligned windows of clean and MP3 latents, one item per (song, bitrate, window).

    Reads the layout `scripts/prepare_dataset.py` writes, `{song}/original.wav`
    and `{song}/{kbps}k.wav`, encodes every file once and keeps the latents on
    the model's device. Windows do not overlap: a 4-minute song holds 24
    windows of 107 frames, so at 6 bitrates it gives 144 items.
    """

    def __init__(self, data_dir: str | Path, model, *, bitrates: list[int], window_frames: int):
        self.window_frames = window_frames
        self.songs = sorted(path for path in Path(data_dir).iterdir() if path.is_dir())
        if not self.songs:
            raise FileNotFoundError(f"No song folders in {data_dir}.")
        self.clean: dict[str, torch.Tensor] = {}
        self.mp3: dict[tuple[str, int], torch.Tensor] = {}
        self.items: list[tuple[str, int, int]] = []
        for song in self.songs:
            original, sample_rate = ga.load(song / "original.wav")
            if sample_rate != model.sample_rate:
                raise ValueError(
                    f"{song.name}: {sample_rate} Hz, the model needs {model.sample_rate}."
                )
            self.clean[song.name] = encode(model, original)
            for kbps in bitrates:
                mp3, _ = ga.load(song / f"{kbps}k.wav")
                if mp3.shape != original.shape:
                    raise ValueError(f"{song.name}: {kbps}k.wav is not the original's length.")
                self.mp3[(song.name, kbps)] = encode(model, mp3)
            # Whole frames of real audio, not the encoder's padding past the end.
            frames = original.shape[-1] // model.pretransform.downsampling_ratio
            starts = range(0, frames - window_frames + 1, window_frames)
            self.items += [(song.name, kbps, start) for kbps in bitrates for start in starts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean and MP3 latents of one window, both (256, 107)."""
        song, kbps, start = self.items[index]
        window = slice(start, start + self.window_frames)
        return self.clean[song][:, window], self.mp3[(song, kbps)][:, window]


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


@torch.no_grad()
def validation_loss(
    model,
    dataset: LatentPairDataset,
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
    for clean, mp3 in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
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
def restore(model, mp3: torch.Tensor, *, steps: int, inpaint_mask_value: float) -> np.ndarray:
    """Generate the clean version of one window of MP3 latents, as audio.

    mp3 (256, 107) -> audio (2, 438_272): 107 frames of 4096 samples, starting
    from seeded noise and denoising in `steps` steps with SA3's own sampler.
    """
    from stable_audio_3.inference.sampling import sample_diffusion

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
        disable_tqdm=True,
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
    """Write `lora.safetensors` (trained weights, fp32) and `state.pt` (optimizer, step, epoch).

    The weights stay fp32: SA3's own save_lora_safetensors rounds them to
    fp16, which would make a resumed run start from slightly different
    weights.
    """
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: tensor.contiguous().cpu() for name, tensor in _trainable_state(model).items()}
    save_file(weights, directory / "lora.safetensors", metadata={"lora_config": json.dumps(config)})
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
