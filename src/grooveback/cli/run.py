"""Run the configured mode over a file or a folder of files.

    uv run python -m grooveback.cli.run input=data/degraded output_dir=artifacts/sdedit \\
        model.noise_level=0.3
    uv run python -m grooveback.cli.run -m input=data/degraded output_dir=artifacts/roundtrip \\
        model=roundtrip "model.ae=same-l,earvae,earvae2,codicodec"
    uv run python -m grooveback.cli.run model=audiosr input=data/degraded output_dir=artifacts/audiosr

Hydra composes `configs/config.yaml`: `model` selects the restoration method
and its parameters, `mode` the run type. An output that already exists is
skipped, so sweeps can share one output_dir — the method's parameters are in
every filename.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from grooveback import audio as ga
from grooveback import latents as gl
from grooveback.baselines import load_apollo, run_apollo, run_audiosr
from grooveback.priors import PRIOR_VARIANTS, load_prior
from grooveback.solvers import latent_sub, roundtrip, sdedit

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac")


def audio_files(input_path: Path) -> list[Path]:
    """The file itself, or every audio file directly inside the folder."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            path for path in input_path.iterdir()
            if path.suffix.lower() in AUDIO_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No audio files in {input_path}.")
        return files
    raise FileNotFoundError(f"{input_path} does not exist.")


def _sdedit_setup(cfg: DictConfig):
    variant_defaults = PRIOR_VARIANTS[cfg.model.variant]
    steps = cfg.model.steps if cfg.model.steps is not None else variant_defaults["steps"]
    cfg_scale = (
        cfg.model.cfg_scale
        if cfg.model.cfg_scale is not None
        else variant_defaults["cfg_scale"]
    )
    # t-runs are named by theta, n-runs by noise level; a run that sets both
    # (e.g. a prompted variant) gets its own output_dir instead.
    if cfg.model.theta is not None:
        suffix = f"_{cfg.model.name}_t{cfg.model.theta:.2f}.wav"
    else:
        suffix = f"_{cfg.model.name}_n{cfg.model.noise_level:.2f}.wav"

    def load():
        return load_prior(cfg.model.variant, device=cfg.device)

    def restore(model, signal, sample_rate):
        return sdedit(
            model,
            signal,
            sample_rate,
            noise_level=cfg.model.noise_level,
            steps=steps,
            cfg_scale=cfg_scale,
            theta=cfg.model.theta,
            prompt=cfg.model.prompt,
            negative_prompt=cfg.model.negative_prompt,
            seed=cfg.model.seed,
        )

    return suffix, load, restore


def _roundtrip_setup(cfg: DictConfig):
    suffix = f"_{cfg.model.name}_{cfg.model.ae}.wav"

    def load():
        return gl.load_ae(cfg.model.ae, device=cfg.device)

    def restore(model, signal, sample_rate):
        return roundtrip(model, signal, sample_rate, ae=cfg.model.ae)

    return suffix, load, restore


def _latent_sub_setup(cfg: DictConfig):
    suffix = f"_{cfg.model.name}_{cfg.model.ae}.wav"
    # Loaded before anything heavy so a wrong path fails immediately.
    damage = np.load(cfg.model.damage_npy)

    def load():
        return gl.load_ae(cfg.model.ae, device=cfg.device)

    def restore(model, signal, sample_rate):
        return latent_sub(model, signal, sample_rate, ae=cfg.model.ae, damage=damage)

    return suffix, load, restore


def _apollo_setup(cfg: DictConfig):
    suffix = f"_{cfg.model.name}.wav"

    def load():
        return load_apollo(device=cfg.device)

    def restore(model, signal, sample_rate):
        return run_apollo(
            signal,
            sample_rate,
            model=model,
            chunk_seconds=cfg.model.chunk_seconds,
            overlap_seconds=cfg.model.overlap_seconds,
            chunk_pad_seconds=cfg.model.chunk_pad_seconds,
            batch_size=cfg.model.batch_size,
        )

    return suffix, load, restore


def _audiosr_setup(cfg: DictConfig):
    suffix = f"_{cfg.model.name}.wav"

    def load():
        return None  # the subprocess owns the model (baselines.run_audiosr)

    def restore(model, signal, sample_rate):
        return run_audiosr(
            signal,
            sample_rate,
            ddim_steps=cfg.model.ddim_steps,
            guidance_scale=cfg.model.guidance_scale,
            seed=cfg.model.seed,
            device=None if cfg.device == "auto" else cfg.device,
        )

    return suffix, load, restore


SETUPS = {
    "sdedit": _sdedit_setup,
    "roundtrip": _roundtrip_setup,
    "latent-sub": _latent_sub_setup,
    "apollo": _apollo_setup,
    "audiosr": _audiosr_setup,
}


def inference(cfg: DictConfig) -> None:
    if cfg.model.name not in SETUPS:
        raise ValueError(
            f"Unknown model {cfg.model.name!r}; models: {', '.join(sorted(SETUPS))}."
        )
    suffix, load, restore = SETUPS[cfg.model.name](cfg)

    output_dir = Path(cfg.output_dir)
    pending = []
    for source in audio_files(Path(cfg.input)):
        target = output_dir / (source.stem + suffix)
        if target.exists():
            print(f"skip {target.name}: exists")
        else:
            pending.append((source, target))
    if not pending:
        return

    model = load()
    for source, target in tqdm(pending, unit="file"):
        signal, sample_rate = ga.load(source)
        ga.save(target, restore(model, signal, sample_rate), sample_rate)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.mode != "inference":
        raise ValueError(f"Unknown mode {cfg.mode!r}; modes: inference.")
    inference(cfg)


if __name__ == "__main__":
    main()
