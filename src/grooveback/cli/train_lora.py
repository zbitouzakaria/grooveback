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
weights (`lora.safetensors`), the optimizer state (`state.pt`) and the id of
the MLflow run that wrote them (`mlflow_run_id`). Of the
checkpoints it writes, it keeps two, the one with the lowest validation loss
and the last, and deletes the others: on medium-base each is about 1 GB.
Checkpoints of earlier runs, such as the one a run resumes from, are never
deleted, and a save_dir that holds a checkpoint past the step a run starts
from is refused, since the run would write over it. Each new best also gets
level-matched listening packs (`listen/{song}/`): the first window of each of
the first `listening_songs` validation songs, restored from the coded version
validation scores it with, next to its original and that coded input. Each
costs about 30 s on medium-base, paid only when the validation loss improves.

An `audio_loss_weight` above 0 adds ADR-0018's decoder skips. Every step then
makes a second pass, at the sampler's last noise level, that decodes the
model's estimate with the skips and trains them and the LoRA on an audio loss.
Validation reports that loss too, and it picks the best checkpoint in place of
the validation loss. Checkpoints gain `skips.safetensors`. A resume from a
checkpoint without skips starts them at zero.

Ctrl-C or `kill <pid>` (SIGINT or SIGTERM) ends a run at once while it trains;
a run resumed from its last checkpoint repeats the interrupted epoch. While an
epoch is validated and saved, the signal waits until that is done.

MLflow (`mlflow_db`) gets the losses per epoch (epoch 0 = before training),
the batch loss and seconds per step every `log_every` steps, peak GPU memory
per epoch, and GPU/CPU system metrics once a minute. A run resumed from a
checkpoint logs into the MLflow run that wrote it, when `mlflow_db` holds that
run, and opens a new one otherwise. MLflow refuses to change a parameter, so a
continued run keeps its name and the values it was first given, and the trainer
prints the ones this call changed. Browse them with

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
    song: Path,
    steps: int,
    inpaint_mask_value: float,
) -> None:
    """Restore the first window of validation song `song` from the coded
    version validation scores it with, and write it level-matched next to the
    original and that coded input."""
    kbps = dataset.fixed[dataset.windows.index((song.name, 0))]
    samples = dataset.window_frames * model.pretransform.downsampling_ratio
    mp3 = dataset.mp3[(song.name, kbps)][:, : dataset.window_frames]
    coded = ga.load(song / f"{kbps}k.wav", stop=samples)[0]
    restored = gt.restore(
        model, mp3, steps=steps, inpaint_mask_value=inpaint_mask_value, mp3_audio=coded
    )
    pack = {
        "original": ga.load(song / "original.wav")[0][:, :samples],
        "input": coded,
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
    # metrics by step. A resume continues its checkpoint's run, whose epochs and
    # steps it extends. Opened first, so a store that cannot be written fails before any
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
    skip_params = gt.add_skips(model) if cfg.audio_loss_weight > 0 else []
    step, start_epoch, run_id, resumed_skips = 0, 0, None, False
    if cfg.resume_from_checkpoint:
        directory = Path(cfg.resume_from_checkpoint)
        gt.load_lora_weights(model, directory / "lora.safetensors", config=lora_config)
        resumed_skips = bool(skip_params) and (directory / "skips.safetensors").exists()
        if resumed_skips:
            gt.load_skips(model, directory / "skips.safetensors")
            optimizer.add_param_group({"params": skip_params})
        step, start_epoch = gt.load_training_state(directory, optimizer)
        # The MLflow run that wrote the checkpoint, continued if this store holds it.
        # A checkpoint from before runs were recorded, or a new store, gets a new run.
        run_file = directory / "mlflow_run_id"
        written_by = run_file.read_text() if run_file.exists() else None
        if written_by and mlflow.search_runs(
            filter_string=f"attributes.run_id = '{written_by}'", output_format="list"
        ):
            run_id = written_by
    if skip_params:
        # New skips join after the load: a checkpoint without them holds one optimizer group.
        if not resumed_skips:
            optimizer.add_param_group({"params": skip_params})
        # A map that starts at zero and must reach weights near 1 needs far larger steps
        # than the LoRA's: Adam moves each weight by about the learning rate per step.
        optimizer.param_groups[-1]["lr"] = cfg.skip_lr
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
        keep_audio=bool(skip_params),
    )
    level = gt.last_noise_level(model, steps=cfg.sample_steps, frames=window_frames)
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

    def validate_audio() -> float:
        return gt.validation_audio_loss(
            model,
            valid_set,
            level=level,
            batch_size=cfg.batch_size,
            inpaint_mask_value=cfg.inpaint_mask_value,
        )

    # System metrics: GPU utilisation, GPU memory and power, CPU and RAM, sampled
    # every 10 s and logged as the mean of 6 samples, one point per minute, on
    # their own time axis (psutil and nvidia-ml-py).
    mlflow.set_system_metrics_sampling_interval(10)
    mlflow.set_system_metrics_samples_before_logging(6)
    # A new run is named after save_dir. A continued run keeps its name: given one,
    # MLflow renames it.
    with mlflow.start_run(
        run_id=run_id,
        run_name=None if run_id else Path(cfg.save_dir).name,
        log_system_metrics=cfg.log_system_metrics,
    ) as run:
        # A continued run already holds the parameters it was first given, which MLflow
        # will not change: it gets only the ones it lacks, and the changed ones are named.
        values = OmegaConf.to_container(cfg, resolve=True)
        first = run.data.params
        mlflow.log_params({key: value for key, value in values.items() if key not in first})
        changed = [
            key for key, value in values.items() if key in first and first[key] != str(value)
        ]
        if changed:
            print(f"MLflow run {run.info.run_id} keeps its first {', '.join(changed)}", flush=True)
        if step == 0:
            # The baseline every later validation loss is read against.
            baseline = validate()
            print(f"step 0 validation loss {baseline:.4f}", flush=True)
            mlflow.log_metric("validation_loss", baseline, step=0)
        if skip_params and not resumed_skips:
            # Skips at zero: the audio loss every later one is read against.
            baseline = validate_audio()
            print(f"step {step} validation audio loss {baseline:.4f}", flush=True)
            mlflow.log_metric("validation_audio_loss", baseline, step=start_epoch)

        def listen(directory: Path) -> None:
            for song in valid_set.songs[: cfg.listening_songs]:
                write_listening_window(
                    model,
                    valid_set,
                    directory / "listen" / song.name,
                    song=song,
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
            losses, audio_losses = [], []
            since, since_step = time.perf_counter(), step
            for master, mp3_audio in loader:
                # Fresh audio from the workers, (8, 2, 438_272) on medium-base, to
                # (8, 256, 107) latents. Whole frames already, so nothing to pad.
                master, mp3_audio = master.to(device), mp3_audio.to(device)
                with torch.no_grad():
                    clean = model.pretransform.encode(master)
                    mp3 = model.pretransform.encode(mp3_audio)
                loss = gt.training_step(
                    model, clean, mp3=mp3, inpaint_mask_value=cfg.inpaint_mask_value
                )
                optimizer.zero_grad()
                loss.backward()
                if skip_params:
                    # Backward after each pass, so only one DiT graph is held at a time.
                    audio_loss = gt.audio_step(
                        model,
                        clean,
                        mp3=mp3,
                        master=master,
                        mp3_audio=mp3_audio,
                        level=level,
                        inpaint_mask_value=cfg.inpaint_mask_value,
                    )
                    (cfg.audio_loss_weight * audio_loss).backward()
                    audio_losses.append(audio_loss.item())
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
                        f"audio loss {audio_losses[-1]:.4f}" if skip_params else "",
                        flush=True,
                    )
                    mlflow.log_metrics(
                        {"batch_loss": loss.item(), "seconds_per_step": seconds}, step=step
                    )
                    if skip_params:
                        mlflow.log_metric("batch_audio_loss", audio_losses[-1], step=step)

            # A Ctrl-C or kill during the steps above ends the run at once. From here to
            # the next epoch it waits, so the checkpoints on disk are always whole.
            with signals_deferred():
                valid = validate()
                print(f"epoch {epoch + 1} step {step} validation loss {valid:.4f}", flush=True)
                training = sum(losses) / len(losses)
                mlflow.log_metrics(
                    {"training_loss": training, "validation_loss": valid}, step=epoch + 1
                )
                # What picks the best checkpoint: the audio loss, once there are skips.
                score = valid
                if skip_params:
                    score = validate_audio()
                    print(f"epoch {epoch + 1} validation audio loss {score:.4f}", flush=True)
                    audio_metrics = {
                        "training_audio_loss": sum(audio_losses) / len(audio_losses),
                        "validation_audio_loss": score,
                    }
                    mlflow.log_metrics(audio_metrics, step=epoch + 1)
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
                # The run a resume from this checkpoint continues.
                (directory / "mlflow_run_id").write_text(run.info.run_id)
                print(f"wrote {directory}", flush=True)
                # Only the best and the last of this run's checkpoints are kept, and a
                # checkpoint is deleted only once a newer one has been written in full.
                stale = {best, last}
                if score < best_loss:
                    best, best_loss = directory, score
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
            name = "validation audio loss" if skip_params else "validation loss"
            print(f"best {best} ({name} {best_loss:.4f}), last {last}", flush=True)


@hydra.main(config_path="../../../configs", config_name="train_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
