"""Run the configured mode over a file or a folder of files.

    uv run python -m grooveback.cli.run input=data/degraded output_dir=artifacts/sdedit \\
        model.noise_level=0.3

Hydra composes `configs/config.yaml`: `model` selects the restoration method
and its parameters, `mode` the run type. An output that already exists is
skipped, so sweeps (`-m 'model.noise_level=0.1,0.3'`) can share one
output_dir — the noise level is in every filename.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from grooveback import audio as ga
from grooveback.priors import PRIOR_VARIANTS, load_prior
from grooveback.solvers import sdedit

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


def inference(cfg: DictConfig) -> None:
    if cfg.model.name != "sdedit":
        raise ValueError(f"Unknown model {cfg.model.name!r}; models: sdedit.")
    variant_defaults = PRIOR_VARIANTS[cfg.model.variant]
    steps = cfg.model.steps if cfg.model.steps is not None else variant_defaults["steps"]
    cfg_scale = (
        cfg.model.cfg_scale
        if cfg.model.cfg_scale is not None
        else variant_defaults["cfg_scale"]
    )

    output_dir = Path(cfg.output_dir)
    suffix = f"_{cfg.model.name}_{cfg.model.noise_level:.2f}.wav"
    pending = []
    for source in audio_files(Path(cfg.input)):
        target = output_dir / (source.stem + suffix)
        if target.exists():
            print(f"skip {target.name}: exists")
        else:
            pending.append((source, target))
    if not pending:
        return

    model = load_prior(cfg.model.variant, device=cfg.device)
    for source, target in tqdm(pending, unit="file"):
        signal, sample_rate = ga.load(source)
        restored = sdedit(
            model,
            signal,
            sample_rate,
            noise_level=cfg.model.noise_level,
            steps=steps,
            cfg_scale=cfg_scale,
            prompt=cfg.model.prompt,
            seed=cfg.model.seed,
        )
        ga.save(target, restored, sample_rate)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.mode != "inference":
        raise ValueError(f"Unknown mode {cfg.mode!r}; modes: inference.")
    inference(cfg)


if __name__ == "__main__":
    main()
