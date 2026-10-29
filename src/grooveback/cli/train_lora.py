"""LoRA fine-tune of Stable Audio 3 to restore MP3s, configured by configs/train_lora.yaml.

    uv run python -m grooveback.cli.train_lora sources.tracks.dir=data/lora/musdb-train \\
        sources.remixes.dir=data/lora/musdb-train valid_dir=data/lora/valid \\
        save_dir=artifacts/lora/run1
    # continue that run with a new learning rate (the config always wins):
    uv run python -m grooveback.cli.train_lora ... num_train_epochs=2 lr=5e-5 \\
        resume_from_checkpoint=artifacts/lora/run1/checkpoint-{step}

Every training item is a fresh window, read from one of the `sources` picked
by its weight: `tracks`, real tracks at a random start (`TrackDataset`), or
`remixes`, one MUSDB18-HQ stem of each kind from random songs (`RemixDataset`).
Apollo splits them half and half, as the config does. `augmentations` sets
ADR-0017's random EQs, left/right swap and chain of one or two codecs.
`valid_dir` is a `scripts/prepare_dataset.py` output: fixed windows, each with
a sox MP3 at a fixed one of `bitrates`. The split between training and
validation is in the folders: the training ones must not hold the validation
songs. MUSDB18-HQ's own train/ folder holds musdb's 14 validation songs, so
they are moved out of it.

After every epoch the run writes `save_dir/checkpoint-{step}/`: the trained
weights (`lora.safetensors`) and the optimizer state (`state.pt`). Of the
checkpoints it writes, it keeps two, the one with the lowest validation loss
and the last, and deletes the others: on medium-base each is about 1 GB.
Checkpoints of earlier runs, such as the one a run resumes from, are never
deleted, and a save_dir that holds a checkpoint past the step a run starts
from is refused, since the run would write over it. Each new best also gets a
level-matched listening pack (`listen/`): one restored validation window next
to its original, its MP3, and the original's own round trip through the
autoencoder. It costs about 30 s on medium-base, paid only when the validation
loss improves.

Ctrl-C or `kill <pid>` (SIGINT or SIGTERM) ends a run at once while it trains;
a run resumed from its last checkpoint repeats the interrupted epoch. While an
epoch is validated and saved, the signal waits until that is done.

MLflow (`mlflow_db`) gets the losses per epoch (epoch 0 = before training),
the batch loss and seconds per step every `log_every` steps, peak GPU memory
per epoch, and GPU/CPU system metrics once a minute. Browse them with

    uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from grooveback import audio as ga
from grooveback import dataset as gd
from grooveback import training as gt
from grooveback.baselines import select_device
from grooveback.evaluation import write_listening_pack

# MLflow sends usage telemetry unless this is set when it is first imported.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
import mlflow  # noqa: E402


@contextlib.contextmanager
def signals_deferred():
    """Hold SIGINT (Ctrl-C) and SIGTERM (`kill`) until the block ends, then act on
    the first one with the handlers that were there before: Ctrl-C raises
    KeyboardInterrupt, a `kill` ends the process. The handler only records the
    signal, so a second one arriving while it runs is safe."""
    held = []

    def hold(signum, frame):
        held.append(signum)

    sigint = signal.signal(signal.SIGINT, hold)
    sigterm = signal.signal(signal.SIGTERM, hold)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, sigint)
        signal.signal(signal.SIGTERM, sigterm)
        if held:
            print(f"{signal.Signals(held[0]).name} held until the checkpoint was saved", flush=True)
            signal.raise_signal(held[0])


def write_listening_window(
    model,
    dataset: gd.LatentPairDataset,
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
    # A source at weight 0 is not loaded.
    sources = {name: spec for name, spec in cfg.sources.items() if spec.weight > 0}
    if not sources:
        raise ValueError("Give at least one of the training sources a weight above 0.")
    unset = [name for name, spec in sources.items() if spec.dir is None]
    if unset:
        raise ValueError(f"sources.{unset[0]}.dir is not set, though its weight is above 0.")
    augmentations = gd.Augmentations(**OmegaConf.to_container(cfg.augmentations))
    # MLflow keeps one run per call, in a local SQLite file: its parameters are
    # this config; epoch metrics are indexed by epoch (0 = before training), batch
    # metrics by step. A resumed run is a new MLflow run whose epochs continue the
    # old one's. Opened first, so a store that cannot be written fails before any
    # loading; likewise the GPU metrics, which MLflow would skip with one log line.
    mlflow.set_tracking_uri(f"sqlite:///{Path(cfg.mlflow_db).resolve()}")
    mlflow.set_experiment("train_lora")
    if cfg.log_system_metrics and torch.cuda.is_available():
        import pynvml

        pynvml.nvmlInit()

    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    model = gt.load_model(cfg.variant, device=device)
    lora_config = gt.lora_config(rank=cfg.rank, lora_alpha=cfg.lora_alpha)
    params = gt.add_lora(model, lora_config)
    print(f"{cfg.variant}: {sum(p.numel() for p in params):,} trainable weights", flush=True)

    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
    )
    step, start_epoch = 0, 0
    if cfg.resume_from_checkpoint:
        directory = Path(cfg.resume_from_checkpoint)
        gt.load_lora_weights(model, directory / "lora.safetensors", config=lora_config)
        step, start_epoch = gt.load_training_state(directory, optimizer)
    # This run writes checkpoint-{n} for steps n past the one it starts from. One already
    # in save_dir is an earlier run's: this run would write over it, then could delete it.
    later = sorted(
        path.name
        for path in Path(cfg.save_dir).glob("checkpoint-*")
        if int(path.name.removeprefix("checkpoint-")) > step
    )
    if later:
        raise ValueError(
            f"{cfg.save_dir} already holds {later[0]}, past step {step}, from an earlier "
            "run; this run would write over it. Choose another save_dir."
        )

    # 10 s -> 441_000 samples -> 107 whole frames of 4096 samples.
    window_frames = (
        int(cfg.window_seconds * model.sample_rate) // model.pretransform.downsampling_ratio
    )
    bitrates = list(cfg.bitrates)
    window_samples = window_frames * model.pretransform.downsampling_ratio
    datasets = {
        "tracks": lambda spec: gd.TrackDataset(
            spec.dir, glob=spec.glob, window_samples=window_samples, augmentations=augmentations
        ),
        "remixes": lambda spec: gd.RemixDataset(
            spec.dir, window_samples=window_samples, augmentations=augmentations
        ),
    }
    train_set = gd.InterleavedDataset(
        [datasets[name](spec) for name, spec in sources.items()],
        probabilities=[spec.weight for spec in sources.values()],
        windows=cfg.epoch_windows,
    )
    valid_set = gd.LatentPairDataset(
        cfg.valid_dir,
        model,
        bitrates=bitrates,
        window_frames=window_frames,
        bitrates_file=Path(cfg.valid_dir) / "validation_bitrates.json",
    )
    print(
        f"{len(train_set)} training windows an epoch from {', '.join(sources)}, "
        f"{len(valid_set)} validation windows",
        flush=True,
    )
    if len(train_set) < cfg.batch_size:
        raise ValueError(
            f"epoch_windows={len(train_set)} is fewer than one batch of {cfg.batch_size}."
        )

    def validate() -> float:
        return gt.validation_loss(
            model,
            valid_set,
            timesteps=list(cfg.validation_timesteps),
            batch_size=cfg.batch_size,
            inpaint_mask_value=cfg.inpaint_mask_value,
        )

    # System metrics: GPU utilisation, GPU memory and power, CPU and RAM, sampled
    # every 10 s and logged as the mean of 6 samples, one point per minute, on
    # their own time axis (psutil and nvidia-ml-py).
    mlflow.set_system_metrics_sampling_interval(10)
    mlflow.set_system_metrics_samples_before_logging(6)
    with mlflow.start_run(
        run_name=Path(cfg.save_dir).name, log_system_metrics=cfg.log_system_metrics
    ):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        if step == 0:
            # The baseline every later validation loss is read against.
            baseline = validate()
            print(f"step 0 validation loss {baseline:.4f}", flush=True)
            mlflow.log_metric("validation_loss", baseline, step=0)

        def listen(directory: Path) -> None:
            write_listening_window(
                model,
                valid_set,
                directory / "listen",
                kbps=min(bitrates),
                steps=cfg.sample_steps,
                inpaint_mask_value=cfg.inpaint_mask_value,
            )

        started = time.perf_counter()
        # This run's checkpoint with the lowest validation loss, and its last one.
        best, best_loss, last = None, float("inf"), None
        for epoch in range(start_epoch, cfg.num_train_epochs):
            # Windows are drawn and coded on the CPU in worker processes. The generator
            # seeds the workers' draws: a fixed seed per epoch, so a resumed run's workers
            # draw what an uninterrupted run's would.
            loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=cfg.batch_size,
                drop_last=True,
                generator=torch.Generator().manual_seed(cfg.seed + epoch),
                num_workers=cfg.num_workers,
            )
            losses = []
            since, since_step = time.perf_counter(), step
            for clean, mp3 in loader:
                # Fresh audio from the workers, (8, 2, 438_272) on medium-base, to
                # (8, 256, 107) latents. Whole frames already, so nothing to pad.
                with torch.no_grad():
                    clean = model.pretransform.encode(clean.to(device))
                    mp3 = model.pretransform.encode(mp3.to(device))
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
                    # Per-step metrics, indexed by step, so the curves move during an epoch.
                    # Divided by the steps actually timed: an epoch's first log can
                    # come after fewer than log_every steps.
                    seconds = (time.perf_counter() - since) / (step - since_step)
                    since, since_step = time.perf_counter(), step
                    print(
                        f"epoch {epoch + 1} step {step} loss {loss.item():.4f} "
                        f"{seconds:.2f} s/step",
                        flush=True,
                    )
                    mlflow.log_metrics(
                        {"batch_loss": loss.item(), "seconds_per_step": seconds}, step=step
                    )

            # A Ctrl-C or kill during the steps above ends the run at once. From here to
            # the next epoch it waits, so the checkpoints on disk are always whole.
            with signals_deferred():
                valid = validate()
                print(f"epoch {epoch + 1} step {step} validation loss {valid:.4f}", flush=True)
                training = sum(losses) / len(losses)
                mlflow.log_metrics(
                    {"training_loss": training, "validation_loss": valid}, step=epoch + 1
                )
                if torch.cuda.is_available():
                    # Peaks since the process started, so they include the encoding.
                    # Allocated is what the tensors needed; reserved adds PyTorch's
                    # cache, and is the number a smaller card must hold.
                    peaks = {
                        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                    }
                    mlflow.log_metrics(peaks, step=epoch + 1)
                directory = Path(cfg.save_dir) / f"checkpoint-{step}"
                gt.save_checkpoint(
                    directory, model, optimizer, config=lora_config, step=step, epoch=epoch + 1
                )
                print(f"wrote {directory}", flush=True)
                # Only the best and the last of this run's checkpoints are kept, and a
                # checkpoint is deleted only once a newer one has been written in full.
                stale = {best, last}
                if valid < best_loss:
                    best, best_loss = directory, valid
                    # The model holds the best weights right now.
                    listen(directory)
                last = directory
                for old in stale - {best, last, None}:
                    shutil.rmtree(old)
            if cfg.max_train_minutes is not None and (
                time.perf_counter() - started >= 60 * cfg.max_train_minutes
            ):
                print(
                    f"stopping after epoch {epoch + 1}: {cfg.max_train_minutes} min budget",
                    flush=True,
                )
                break

        if last is not None:
            # best stays None when no validation loss was finite: a NaN is never below another.
            print(f"best {best} (validation loss {best_loss:.4f}), last {last}", flush=True)


@hydra.main(config_path="../../../configs", config_name="train_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
