"""LoRA fine-tune of Stable Audio 3 to restore MP3s, configured by configs/train_lora.yaml.

    uv run python -m grooveback.cli.train_lora train_dir=data/lora/train \\
        valid_dir=data/lora/valid save_dir=artifacts/lora/run1
    # continue that run with a new learning rate (the config always wins):
    uv run python -m grooveback.cli.train_lora ... num_train_epochs=2 lr=5e-5 \\
        resume_from_checkpoint=artifacts/lora/run1/checkpoint-{step}

`train_dir` and `valid_dir` are `scripts/prepare_dataset.py` outputs. After
every epoch the run writes `save_dir/checkpoint-{step}/`: the trained weights
(`lora.safetensors`), the optimizer state (`state.pt`), and a level-matched
listening pack (`listen/`) with one restored validation window next to its
original, its MP3, and the original's own round trip through the autoencoder.
Training and validation loss per epoch go to MLflow (`mlflow_db`); browse them with

    uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from grooveback import audio as ga
from grooveback import training as gt
from grooveback.baselines import select_device
from grooveback.evaluation import write_listening_pack


def write_listening_window(
    model,
    dataset: gt.LatentPairDataset,
    out_dir: Path,
    *,
    kbps: int,
    steps: int,
    inpaint_mask_value: float,
) -> None:
    """Restore the first window of the first validation song at `kbps`, and
    write it level-matched next to the original, the MP3, and the original's
    round trip through the autoencoder (the best the decoder can do with the
    encoder's latents)."""
    song = dataset.songs[0]
    window = slice(0, dataset.window_frames)
    samples = dataset.window_frames * model.pretransform.downsampling_ratio
    clean, mp3 = dataset.clean[song.name][:, window], dataset.mp3[(song.name, kbps)][:, window]
    with torch.no_grad():
        roundtrip = model.pretransform.decode(clean[None])[0].float().cpu().numpy()
    restored = gt.restore(model, mp3, steps=steps, inpaint_mask_value=inpaint_mask_value)
    pack = {
        "original": ga.load(song / "original.wav")[0][:, :samples],
        "input": ga.load(song / f"{kbps}k.wav")[0][:, :samples],
        "original-roundtrip": roundtrip[:, :samples],
        "restored": restored[:, :samples],
    }
    write_listening_pack(pack, model.sample_rate, out_dir, suffix=".flac")


def train(cfg: DictConfig) -> None:
    # MLflow keeps one run per call, in a local SQLite file: its parameters are
    # this config, and its metrics are indexed by epoch (0 = before training).
    # A resumed run is a new MLflow run whose epochs continue the old one's.
    # Opened first, so a store that cannot be written fails before any loading.
    mlflow.set_tracking_uri(f"sqlite:///{Path(cfg.mlflow_db).resolve()}")
    mlflow.set_experiment("train_lora")

    torch.manual_seed(cfg.seed)
    model = gt.load_model(cfg.variant, device=select_device(cfg.device))
    lora_config = gt.lora_config(rank=cfg.rank, lora_alpha=cfg.lora_alpha)
    params = gt.add_lora(model, lora_config)
    print(f"{cfg.variant}: {sum(p.numel() for p in params):,} trainable weights", flush=True)

    # 10 s -> 441_000 samples -> 107 whole frames of 4096 samples.
    window_frames = (
        int(cfg.window_seconds * model.sample_rate) // model.pretransform.downsampling_ratio
    )
    bitrates = list(cfg.bitrates)
    train_set = gt.LatentPairDataset(
        cfg.train_dir, model, bitrates=bitrates, window_frames=window_frames
    )
    valid_set = gt.LatentPairDataset(
        cfg.valid_dir, model, bitrates=bitrates, window_frames=window_frames
    )
    print(f"{len(train_set)} training windows, {len(valid_set)} validation windows", flush=True)
    if len(train_set) < cfg.batch_size:
        raise ValueError(
            f"{len(train_set)} training windows, fewer than one batch of {cfg.batch_size}."
        )

    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
    )
    step, start_epoch = 0, 0
    if cfg.resume_from_checkpoint:
        directory = Path(cfg.resume_from_checkpoint)
        gt.load_lora_weights(model, directory / "lora.safetensors", config=lora_config)
        step, start_epoch = gt.load_training_state(directory, optimizer)

    def validate() -> float:
        return gt.validation_loss(
            model,
            valid_set,
            timesteps=list(cfg.validation_timesteps),
            batch_size=cfg.batch_size,
            inpaint_mask_value=cfg.inpaint_mask_value,
        )

    with mlflow.start_run(run_name=Path(cfg.save_dir).name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        if step == 0:
            # The baseline every later validation loss is read against.
            baseline = validate()
            print(f"step 0 validation loss {baseline:.4f}", flush=True)
            mlflow.log_metric("validation_loss", baseline, step=0)

        for epoch in range(start_epoch, cfg.num_train_epochs):
            # A fixed seed per epoch, so a resumed run shuffles like an uninterrupted one.
            loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=cfg.batch_size,
                shuffle=True,
                drop_last=True,
                generator=torch.Generator().manual_seed(cfg.seed + epoch),
            )
            losses = []
            for clean, mp3 in loader:
                loss = gt.training_step(
                    model, clean, mp3=mp3, inpaint_mask_value=cfg.inpaint_mask_value
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                step += 1
                losses.append(loss.item())
                if step % cfg.log_every == 0:
                    print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f}", flush=True)

            valid = validate()
            print(f"epoch {epoch + 1} step {step} validation loss {valid:.4f}", flush=True)
            training = sum(losses) / len(losses)
            mlflow.log_metrics(
                {"training_loss": training, "validation_loss": valid}, step=epoch + 1
            )
            directory = Path(cfg.save_dir) / f"checkpoint-{step}"
            gt.save_checkpoint(
                directory, model, optimizer, config=lora_config, step=step, epoch=epoch + 1
            )
            write_listening_window(
                model,
                valid_set,
                directory / "listen",
                kbps=min(bitrates),
                steps=cfg.sample_steps,
                inpaint_mask_value=cfg.inpaint_mask_value,
            )
            print(f"wrote {directory}", flush=True)


@hydra.main(config_path="../../../configs", config_name="train_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
