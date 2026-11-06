"""Score a LoRA checkpoint on the MUSDB18-HQ test clips, configured by configs/evaluate.yaml.

    just evaluate model.checkpoint=artifacts/lora/run1/checkpoint-46765
    just evaluate model.checkpoint=... clips=4 'bitrates=[32]' 'seeds=[0]'

At each of `bitrates` it writes the MP3 of the `clips` clips `musdb_eval` draws,
restores them once per seed, many clips per sampling run, and scores the
untouched MP3 (`input`) and each seed's restoration (`seed{n}`) as `musdb_eval`
does. `lora` is the mean of the seeds' scores, the number to compare checkpoints
on. The scores come before level matching, as Apollo's protocol makes them.

`output_dir` receives:

    mp3/{kbps}k/{track}/{start}.wav
    render/seed{n}/{kbps}k/{track}/{start}.wav
    scores.csv                                  one row per clip and bitrate

A render that already exists is kept and everything is scored again, so an
interrupted run can be restarted, and adding a seed costs only that seed. A run
with other sampling settings needs its own `output_dir`, as the default gives it.
scores.csv holds the bitrates and seeds of the last run.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from grooveback import audio as ga
from grooveback import training as gt
from grooveback.baselines import select_device
from grooveback.cli import musdb_eval
from grooveback.solvers import lora_batch


def render(
    model,
    mp3_dir: Path,
    render_dir: Path,
    *,
    batch_size: int,
    steps: int,
    inpaint_mask_value: float,
    sampler_type: str,
    seed: int,
) -> None:
    """Restore every `{track}/{start}.wav` of `mp3_dir` that `render_dir` does not hold yet."""
    paths = [
        path
        for path in sorted(mp3_dir.glob("*/*.wav"))
        if not (render_dir / path.relative_to(mp3_dir)).exists()
    ]
    for first in range(0, len(paths), batch_size):
        batch = paths[first : first + batch_size]
        restored = lora_batch(
            model,
            [ga.load(path)[0] for path in batch],
            model.sample_rate,
            steps=steps,
            inpaint_mask_value=inpaint_mask_value,
            sampler_type=sampler_type,
            seed=seed,
        )
        for path, audio in zip(batch, restored, strict=True):
            # _match_input_loudness leaves a NaN in place, and the scores would only read nan.
            if not np.isfinite(audio).all():
                raise ValueError(f"The restoration of {path} at seed {seed} is not finite.")
            ga.save(render_dir / path.relative_to(mp3_dir), audio, model.sample_rate)
    print(f"{render_dir}: {len(paths)} clips restored", flush=True)


def evaluate(cfg: DictConfig) -> None:
    # Read before output_dir, which is built on it: an unset checkpoint is named as such.
    checkpoint = Path(cfg.model.checkpoint)
    output_dir, musdb_test = Path(cfg.output_dir), Path(cfg.musdb_test)
    if not musdb_eval.mixtures(musdb_test):
        raise FileNotFoundError(f"No {{track}}/mixture.wav or .flac under {musdb_test}.")

    model = gt.load_trained_model(
        cfg.model.variant, checkpoint, device=select_device(cfg.device)
    )
    rows = []
    for kbps in cfg.bitrates:
        mp3_dir = output_dir / "mp3" / f"{kbps}k"
        musdb_eval.evaluate(musdb_test, kbps=kbps, clips=cfg.clips, mp3_dir=mp3_dir)
        renders = {
            f"seed{seed}": output_dir / "render" / f"seed{seed}" / f"{kbps}k" for seed in cfg.seeds
        }
        for seed in cfg.seeds:
            render(
                model,
                mp3_dir,
                renders[f"seed{seed}"],
                batch_size=cfg.batch_size,
                steps=cfg.model.steps,
                inpaint_mask_value=cfg.model.inpaint_mask_value,
                sampler_type=cfg.model.sampler_type,
                seed=seed,
            )
        scores = musdb_eval.evaluate(musdb_test, kbps=kbps, clips=cfg.clips, renders=renders)
        for row in scores:
            row |= {
                f"lora_{metric}": np.mean([row[f"{name}_{metric}"] for name in renders])
                for metric in musdb_eval.METRICS
            }
        rows += [{"kbps": kbps} | row for row in scores]
        print(f"{kbps} kbps", flush=True)
        musdb_eval.print_means(scores)
    musdb_eval.write_scores(rows, output_dir / "scores.csv")


@hydra.main(config_path="../../../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
