"""The training data of the LoRA fine-tune in `grooveback.training`.

`LatentPairDataset` serves windows of the pairs `scripts/prepare_dataset.py`
writes, as latents. `RemixDataset` builds fresh pairs from MUSDB18-HQ stems,
as audio, the way Apollo trains. The comments follow training.py's worked
example: small-music-base, 10 s windows of 107 latent frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from grooveback import audio as ga
from grooveback import training as gt


class LatentPairDataset(torch.utils.data.Dataset):
    """Aligned windows of clean and MP3 latents, one item per window.

    Reads the layout `scripts/prepare_dataset.py` writes, `{song}/original.wav`
    and `{song}/{kbps}k.wav`, encodes every file once and keeps the latents on
    the model's device. Windows do not overlap: a 4-minute song holds 24
    windows of 107 frames.

    An item's MP3 bitrate is drawn at random, uniformly over `bitrates`, every
    time it is read, so each epoch sees every window once at a fresh bitrate.
    With `bitrates_file` (validation) each window keeps one bitrate instead,
    drawn once and stored there (see `fixed_bitrates`).
    """

    def __init__(
        self,
        data_dir: str | Path,
        model,
        *,
        bitrates: list[int],
        window_frames: int,
        bitrates_file: Path | None = None,
    ):
        self.window_frames = window_frames
        self.bitrates = bitrates
        self.songs = sorted(path for path in Path(data_dir).iterdir() if path.is_dir())
        if not self.songs:
            raise FileNotFoundError(f"No song folders in {data_dir}.")
        self.clean: dict[str, torch.Tensor] = {}
        self.mp3: dict[tuple[str, int], torch.Tensor] = {}
        self.windows: list[tuple[str, int]] = []
        for song in self.songs:
            original, sample_rate = ga.load(song / "original.wav")
            if sample_rate != model.sample_rate:
                raise ValueError(
                    f"{song.name}: {sample_rate} Hz, the model needs {model.sample_rate}."
                )
            self.clean[song.name] = gt.encode(model, original)
            for kbps in bitrates:
                mp3, _ = ga.load(song / f"{kbps}k.wav")
                if mp3.shape != original.shape:
                    raise ValueError(f"{song.name}: {kbps}k.wav is not the original's length.")
                self.mp3[(song.name, kbps)] = gt.encode(model, mp3)
            # Whole frames of real audio, not the encoder's padding past the end.
            frames = original.shape[-1] // model.pretransform.downsampling_ratio
            starts = range(0, frames - window_frames + 1, window_frames)
            self.windows += [(song.name, start) for start in starts]
        self.fixed = (
            None
            if bitrates_file is None
            else fixed_bitrates(bitrates_file, self.windows, bitrates=bitrates)
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean and MP3 latents of one window, both (256, 107)."""
        song, start = self.windows[index]
        if self.fixed is None:
            kbps = self.bitrates[int(torch.randint(len(self.bitrates), ()))]
        else:
            kbps = self.fixed[index]
        window = slice(start, start + self.window_frames)
        return self.clean[song][:, window], self.mp3[(song, kbps)][:, window]


def fixed_bitrates(path: Path, windows: list[tuple[str, int]], *, bitrates: list[int]) -> list[int]:
    """One MP3 bitrate per window, drawn once and stored in `path`.

    Validation scores the same (window, bitrate) pairs on every run, so the
    draw is written on first use and read back after. A stored draw for other
    windows (a new window length, say) or from another bitrate list is refused
    rather than redrawn: a draw from [24, 32] would otherwise quietly limit a
    six-bitrate run's validation to its two hardest bitrates.
    """
    if path.exists():
        stored = json.loads(path.read_text())
        if [tuple(window) for window in stored["windows"]] != windows:
            raise ValueError(f"{path} was drawn for other windows; delete it to draw again.")
        if stored["drawn_from"] != bitrates:
            raise ValueError(
                f"{path} was drawn from {stored['drawn_from']}, not {bitrates}; "
                "delete it to draw again."
            )
        return stored["bitrates"]
    rng = np.random.default_rng(0)
    drawn = [int(kbps) for kbps in rng.choice(bitrates, size=len(windows))]
    path.write_text(json.dumps({"drawn_from": bitrates, "windows": windows, "bitrates": drawn}))
    return drawn


def active_starts(track: np.ndarray, window_samples: int) -> np.ndarray:
    """Starts, on a 10 ms grid, of the windows of a 16-bit track that Apollo's activity check keeps.

    Over 10 ms frames (441 samples), a window is kept when more than half its
    frames are at least as loud as the 15th percentile of the track's non-silent
    frames (power above 1e-3). Source: `VAD` in Apollo-data-preprocess's
    musdb_preprocess.py, as `musdb_eval.vad_starts` ports it for 6 s clips;
    Apollo runs it on every stem as well as the mixture. One change: frames equal
    to the percentile count as loud, so a steady tone is kept; that tie does not
    occur in recorded music. A silent stem, as in an instrumental, keeps nothing.
    """
    frame = 441
    usable = track.shape[-1] // frame * frame
    audio = track[:, :usable].astype(np.float32) / 32768
    power = (audio**2).reshape(audio.shape[0], -1, frame).sum(axis=(0, 2))
    active = power[power > 1e-3]
    if not active.size:
        return np.empty(0, dtype=np.int64)
    loud = power >= np.quantile(active, 0.15)
    frames = window_samples // frame
    loud_per_window = np.convolve(loud, np.ones(frames), mode="valid")
    return np.flatnonzero(loud_per_window > frames / 2) * frame


class RemixDataset(torch.utils.data.Dataset):
    """Fresh clean and MP3 audio windows from MUSDB18-HQ songs: Apollo's training recipe.

    Reads `{song}/{mixture,bass,drums,other,vocals}.wav` into memory once, as
    16-bit integers: the 86 training songs take about 17 GB. Every item is drawn
    anew, so `windows` only sets how many items make an epoch. Half the items
    are a window of a real mixture. The other half are remixes: 1 to 4 stems,
    drawn with repeats from the four kinds, each from its own random song and
    start, each scaled by a random gain of -10 to +10 dB, then summed. Windows
    come only from stretches that pass Apollo's activity check (`active_starts`):
    measured on 67 of the songs, 30% of random 10 s vocal windows fail it, and
    15% are below -60 dBFS. The clean window is then coded as MP3 at a bitrate
    drawn uniformly from `bitrates`.

    Source: `MusdbMoisesdbDataset` in Apollo's
    look2hear/datas/musdb_moisesdb_datamodule.py, with its codec,
    `musdb_eval.codec_simu`. The changes:

    - MUSDB's four stem kinds in place of its eleven, so at most 4 stems where
      its config draws up to 8.
    - Apollo codes first and then divides both windows by their common peak, so
      a remix past full scale reaches the codec clipped, and every window ends
      up peaking at 1. Here a window past full scale is scaled down before
      coding, and quieter windows keep their level, as validation and inference
      windows do.
    - The window is coded with `codec_pad` extra samples on each side, cut off
      after. `codec_simu` cuts the MP3 to the clip's length before it removes
      sox's 1,105-sample delay (ADR-0015), so a clip coded alone ends in 25 ms
      of decoder lead-in where the clean audio has music. Validation codes whole
      songs, which pay this only in their last 25 ms.

    Items are audio, not latents: the DataLoader's worker processes build them on
    the CPU, and the training loop encodes them on the GPU. Draws use torch's
    random generator, which the DataLoader seeds differently in every worker.
    """

    STEMS = ("bass", "drums", "other", "vocals")
    codec_pad = 4_096
    """Samples coded on each side of a window and cut off after: more than sox's 1,105 of delay."""

    def __init__(
        self, musdb_dir: str | Path, *, bitrates: list[int], window_samples: int, windows: int
    ):
        import soundfile as sf

        from grooveback.cli.musdb_eval import SR

        self.bitrates = bitrates
        self.window_samples = window_samples
        self.windows = windows
        self.songs = sorted(path for path in Path(musdb_dir).iterdir() if path.is_dir())
        if not self.songs:
            raise FileNotFoundError(f"No song folders in {musdb_dir}.")
        # Per track kind, one (track, starts) pair per song that has an active window:
        # the (2, samples) int16 audio, and the window starts that leave room for the pads.
        self.tracks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            kind: [] for kind in ("mixture", *self.STEMS)
        }
        for song in self.songs:
            for kind, tracks in self.tracks.items():
                audio, sample_rate = sf.read(
                    str(song / f"{kind}.wav"), dtype="int16", always_2d=True
                )
                if sample_rate != SR:
                    raise ValueError(
                        f"{song.name}/{kind}.wav: {sample_rate} Hz, the codec needs {SR}."
                    )
                if audio.shape[0] < window_samples + 2 * self.codec_pad:
                    raise ValueError(
                        f"{song.name}/{kind}.wav is shorter than one window and its codec pads."
                    )
                track = np.ascontiguousarray(audio.T)
                starts = active_starts(track, window_samples)
                last = track.shape[-1] - window_samples - self.codec_pad
                starts = starts[(starts >= self.codec_pad) & (starts <= last)]
                if starts.size:
                    tracks.append((track, starts))
        # Remixes draw from the stem kinds some song has active: one vocal-less song
        # among others leaves vocals in, a set of instrumentals takes them out.
        self.kinds = [kind for kind in self.STEMS if self.tracks[kind]]
        if not self.tracks["mixture"] or not self.kinds:
            raise ValueError(
                f"{musdb_dir}: no mixture or no stem has a window that passes the activity check."
            )

    def __len__(self) -> int:
        return self.windows

    def _window(self, kind: str) -> np.ndarray:
        """An active window of one random song's `kind` track with its codec pads,
        float32 (2, 4_096 + 438_272 + 4_096)."""
        tracks = self.tracks[kind]
        track, starts = tracks[int(torch.randint(len(tracks), ()))]
        start = int(starts[int(torch.randint(len(starts), ()))])
        padded = track[:, start - self.codec_pad : start + self.window_samples + self.codec_pad]
        # Scaled as soundfile reads 16-bit audio as float.
        return padded.astype(np.float32) / 32768

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean and MP3 audio of one fresh window, both (2, 438_272) float32."""
        from grooveback.cli import musdb_eval

        if torch.rand(()) < 0.5:
            clean = self._window("mixture")
        else:
            count = int(torch.randint(1, len(self.STEMS) + 1, ()))
            kinds = [self.kinds[int(i)] for i in torch.randint(len(self.kinds), (count,))]
            # Whole dB, as Apollo draws them: -10, -9, ..., +10.
            gains_db = torch.randint(-10, 11, (count,)).tolist()
            clean = sum(
                self._window(kind) * 10 ** (db / 20)
                for kind, db in zip(kinds, gains_db, strict=True)
            )
        peak = float(np.abs(clean).max())
        if peak > 1.0:
            clean = clean / peak
        kbps = self.bitrates[int(torch.randint(len(self.bitrates), ()))]
        mp3 = musdb_eval.codec_simu(clean, kbps)
        window = slice(self.codec_pad, self.codec_pad + self.window_samples)
        return torch.from_numpy(clean[:, window].copy()), torch.from_numpy(mp3[:, window].copy())
