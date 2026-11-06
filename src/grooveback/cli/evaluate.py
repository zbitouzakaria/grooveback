"""Score a LoRA checkpoint on the MUSDB18-HQ test clips, configured by configs/evaluate.yaml.

    just evaluate model.checkpoint=artifacts/lora/run1/checkpoint-46765
    just evaluate model.checkpoint=... output_dir=artifacts/quick clips=4 'seeds=[0]'

At each of `bitrates` it writes the MP3 of the `clips` clips `musdb_eval` draws,
restores them once per seed, many clips per sampling run, and scores the
untouched MP3 (`input`) and each seed's restoration (`seed{n}`) as `musdb_eval`
does. `lora` is the mean of the seeds' scores, the number to compare checkpoints
on. Renders keep their MP3's loudness: the scores skip the match to -14 LUFS, as
Apollo's protocol does.

`output_dir` receives:

    mp3/{kbps}k/{track}/{start}.wav
    render/seed{n}/{kbps}k/{track}/{start}.wav
    scores.csv                                  one row per clip and bitrate

A render that already exists is kept and everything is scored again, so an
interrupted run can be restarted, and adding a seed costs only that seed. A run
with other sampling settings needs its own `output_dir`, as the default gives it.
scores.csv holds this run's clips, bitrates and seeds alone.
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
    clips: list[Path],
    batch_size: int,
    steps: int,
    inpaint_mask_value: float,
    sampler_type: str,
    seed: int,
) -> None:
    """Restore `clips`, `{track}/{start}.wav` in `mp3_dir`, that `render_dir` does not hold yet."""
    pending = [clip for clip in clips if not (render_dir / clip).exists()]
    for first in range(0, len(pending), batch_size):
        batch = pending[first : first + batch_size]
        restored = lora_batch(
            model,
            [ga.load(mp3_dir / clip)[0] for clip in batch],
            model.sample_rate,
            steps=steps,
            inpaint_mask_value=inpaint_mask_value,
            sampler_type=sampler_type,
            seed=seed,
        )
        for clip, audio in zip(batch, restored, strict=True):
            # _match_input_loudness leaves a NaN in place. Scoring would stop on it too, but
            # only after every render, and without naming the clip.
            if not np.isfinite(audio).all():
                raise ValueError(f"The restoration of {clip} at seed {seed} is not finite.")
            ga.save(render_dir / clip, audio, model.sample_rate)
    print(f"{render_dir}: {len(pending)} clips restored", flush=True)


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
        drawn = musdb_eval.evaluate(musdb_test, kbps=kbps, clips=cfg.clips, mp3_dir=mp3_dir)
        clips = [Path(row["track"]) / f"{row['start']}.wav" for row in drawn]
        renders = {
            f"seed{seed}": output_dir / "render" / f"seed{seed}" / f"{kbps}k" for seed in cfg.seeds
        }
        for seed in cfg.seeds:
            render(
                model,
                mp3_dir,
                renders[f"seed{seed}"],
                clips=clips,
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
