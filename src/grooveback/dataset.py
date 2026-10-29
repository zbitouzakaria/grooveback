"""The training data of the LoRA fine-tune in `grooveback.training`.

Training draws fresh pairs, as audio: `TrackDataset` cuts windows of real
tracks, `RemixDataset` sums one MUSDB18-HQ stem of each kind, and
`InterleavedDataset` picks between them by weight. `Augmentations` sets how
often each item gets ADR-0017's random EQs, left/right swap and chain of codecs.
Validation reads fixed windows of the pairs `scripts/prepare_dataset.py`
writes, as latents (`LatentPairDataset`). The comments follow training.py's
worked example: small-music-base, 10 s windows of 107 latent frames.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
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


CODEC_PAD = 4_096
"""Samples coded on each side of a window and cut off after: more than sox's 1,105 of delay."""


def load_active_tracks(
    paths: list[Path], *, window_samples: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every stereo track at `paths` that has a window passing Apollo's activity
    check, with the starts of those windows that leave room for the codec pads.

    One (track, starts) pair per kept file: the (2, samples) audio, kept as int16
    whatever the file's format, and the starts. A track with no such window is
    left out: a silent stem, as in an instrumental, keeps nothing.
    """
    import soundfile as sf

    from grooveback.cli.musdb_eval import SR

    tracks = []
    for path in paths:
        # Read as float, which soundfile scales from any format, then kept as 16 bits:
        # read as int16, a float WAV's samples would all round to 0. A 16-bit file
        # comes back exactly; a float one past full scale is clipped to it.
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if sample_rate != SR:
            raise ValueError(f"{path}: {sample_rate} Hz, the codec needs {SR}.")
        if audio.shape[1] != 2:
            raise ValueError(f"{path} is not stereo: {audio.shape[1]} channels.")
        if audio.shape[0] < window_samples + 2 * CODEC_PAD:
            raise ValueError(f"{path} is shorter than one window and its codec pads.")
        track = np.round(np.clip(audio.T, -1.0, 32_767 / 32_768) * 32_768).astype(np.int16)
        starts = active_starts(track, window_samples)
        last = track.shape[-1] - window_samples - CODEC_PAD
        starts = starts[(starts >= CODEC_PAD) & (starts <= last)]
        if starts.size:
            tracks.append((track, starts))
    return tracks


def draw_window(tracks: list[tuple[np.ndarray, np.ndarray]], *, window_samples: int) -> np.ndarray:
    """A window of a random track, at a random one of its active starts, with its
    codec pads: float32 (2, 4_096 + 438_272 + 4_096). Drawn anew on every call, so
    the same track gives a different window each time."""
    track, starts = tracks[int(torch.randint(len(tracks), ()))]
    start = int(starts[int(torch.randint(len(starts), ()))])
    padded = track[:, start - CODEC_PAD : start + window_samples + CODEC_PAD]
    # Scaled as soundfile reads 16-bit audio as float.
    return padded.astype(np.float32) / 32768


ENCODERS = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac"}
"""ffmpeg's encoder for each codec. An MP3 can also go through torchaudio's sox."""

CONTAINERS = {"libmp3lame": ".mp3", "libopus": ".ogg", "aac": ".m4a"}
"""The file each ffmpeg encoder writes, one that records the encoder's delay."""


@dataclass(frozen=True)
class Augmentations:
    """What ADR-0017 does to each training item, and how often.

    Every probability is per item, drawn from torch's generator. The defaults
    augment nothing: remixes of every stem kind, no EQ, no swap, and one sox MP3
    at one of musdb_eval's six bitrates. configs/train_lora.yaml sets the ADR's
    values. The EQs and the swap change the clean window before it is coded, so
    the coded copy carries them too and the model still only undoes the codec.
    """

    drop_kind: float = 0.0  # a remix leaves out one stem kind, picked at random
    stem_eq: float = 0.0  # every stem of a remix gets its own random EQ
    mix_eq: float = 0.0  # the whole window gets one random EQ, a track's or a remix's
    swap: float = 0.0  # left and right swapped: a track as a whole, each stem of a remix on its own
    # How many codec round trips a window goes through, each count with how often it is
    # drawn, as URGENT's simulator draws its number of degradations: {1: 0.4, 2: 0.6}.
    codec_rounds: dict[int, float] = field(default_factory=lambda: {1: 1.0})
    # For each round trip, the codecs it draws from with equal odds: mp3, opus or aac.
    codec_choices: list[list[str]] = field(default_factory=lambda: [["mp3"]])
    codec_bitrates: dict[str, list[int]] = field(
        default_factory=lambda: {"mp3": [24, 32, 48, 64, 96, 128]}
    )
    mp3_ffmpeg: float = 0.0  # MP3s coded by ffmpeg's libmp3lame instead of torchaudio's sox

    def __post_init__(self):
        if max(self.codec_rounds) > len(self.codec_choices):
            raise ValueError(
                f"codec_rounds draws up to {max(self.codec_rounds)} round trips, "
                f"but codec_choices lists {len(self.codec_choices)}."
            )
        for codec in {codec for choices in self.codec_choices for codec in choices}:
            if codec not in ENCODERS:
                raise ValueError(f"Unknown codec {codec!r}: the codecs are {', '.join(ENCODERS)}.")
            if not self.codec_bitrates.get(codec):
                raise ValueError(f"codec_bitrates has no bitrates for {codec}.")


NO_AUGMENTATIONS = Augmentations()
"""The datasets' default: every stem kind, no EQ, no swap, one sox MP3 a window."""


def chance(probability: float) -> bool:
    """True with `probability`, drawn from torch's generator; at 0, False without a draw."""
    return probability > 0 and float(torch.rand(())) < probability


def draw_codecs(augmentations: Augmentations) -> list[tuple[str, int]]:
    """The codec round trips of one window, in order: (encoder, kbps) for each,
    where the encoder is "sox" or one of ffmpeg's (`ENCODERS`). With the ADR's
    values, [("sox", 24)] or [("libmp3lame", 96), ("libopus", 128)], say."""
    counts = list(augmentations.codec_rounds)
    weights = torch.tensor([augmentations.codec_rounds[count] for count in counts])
    rounds = counts[int(torch.multinomial(weights, 1))]
    chain = []
    for choices in augmentations.codec_choices[:rounds]:
        codec = choices[int(torch.randint(len(choices), ()))]
        bitrates = augmentations.codec_bitrates[codec]
        kbps = bitrates[int(torch.randint(len(bitrates), ()))]
        sox = codec == "mp3" and not chance(augmentations.mp3_ffmpeg)
        chain.append(("sox" if sox else ENCODERS[codec], kbps))
    return chain


def ffmpeg_round_trip(audio: np.ndarray, *, codec: str, kbps: int) -> np.ndarray:
    """`audio`, float32 (2, samples) at 44.1 kHz, coded by ffmpeg's encoder `codec`
    at `kbps` and decoded back to float32 (2, samples).

    The coded file records the encoder's delay (the LAME tag, Opus's pre-skip, the
    MP4 edit list), and ffmpeg's decoder trims it, so the output starts with the
    input: on two windows of a house track, every codec at 64, 128 and 192 kbps came
    back at lag 0. It can end later, and is cut to the input's length: Opus comes
    back one sample longer from its round trip through 48 kHz, and AAC keeps the
    padding of its last 1,024-sample frame. Each ffmpeg runs on one thread, since
    every DataLoader worker runs its own.
    """
    from grooveback.cli.musdb_eval import SR

    with tempfile.TemporaryDirectory() as tmp:
        coded = Path(tmp) / f"coded{CONTAINERS[codec]}"
        raw = ["-f", "f32le", "-ar", str(SR), "-ac", "2"]
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", *raw, "-i", "pipe:0"]
            + ["-threads", "1", "-c:a", codec, "-b:a", f"{kbps}k", str(coded)],
            input=np.ascontiguousarray(audio.T, dtype=np.float32).tobytes(),
            check=True,
        )
        decoded = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-threads", "1", "-i", str(coded), *raw, "pipe:1"],
            capture_output=True,
            check=True,
        ).stdout
    out = np.frombuffer(decoded, dtype=np.float32).reshape(-1, 2).T
    if out.shape[-1] < audio.shape[-1]:
        raise RuntimeError(
            f"ffmpeg's {codec} came back {audio.shape[-1] - out.shape[-1]} samples short."
        )
    return out[:, : audio.shape[-1]].copy()


def round_trip(audio: np.ndarray, *, codec: str, kbps: int) -> np.ndarray:
    """`audio`, float32 (2, samples), coded by `codec` at `kbps` and decoded back to
    its length: "sox" is torchaudio's sox MP3, which comes back 1,105 samples late,
    and any other is an ffmpeg encoder, which comes back aligned."""
    from grooveback.cli import musdb_eval

    if codec == "sox":
        return musdb_eval.sox_round_trip(audio, kbps)
    return ffmpeg_round_trip(audio, codec=codec, kbps=kbps)


def coded_pair(
    padded: np.ndarray, *, augmentations: Augmentations
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean and coded audio of one padded window, both (2, 438_272) float32.

    A window past full scale is scaled down to it first. It goes through the
    round trips `draw_codecs` draws, with its pads, and then through one
    alignment to the clean window (`musdb_eval.align`), so the coded copy lines
    up whichever codecs it went through. A sox MP3 comes back 1,105 samples
    late, two of them 2,210, and the alignment moves those first samples to the
    end, into the 4,096-sample pad that is cut off after. Coded alone, without
    pads, a clip would end in that decoder lead-in where the clean audio has
    music (ADR-0015).

    The alignment can miss by a sample at 24 kbps: in one run, on 5 of 48
    windows of 10 s, the pair matched better one sample apart. So the clean
    window is cut where it best matches the coded window, up to 4 samples
    either side of the pad.
    """
    from grooveback.cli import musdb_eval

    peak = float(np.abs(padded).max())
    if peak > 1.0:
        padded = padded / peak
    coded = padded
    for codec, kbps in draw_codecs(augmentations):
        coded = round_trip(coded, codec=codec, kbps=kbps)
    samples = padded.shape[-1] - 2 * CODEC_PAD
    coded = musdb_eval.align(coded, clean=padded)[:, CODEC_PAD : CODEC_PAD + samples]
    # The clean window cut 0, ±1, ..., ±4 samples from the pad, nearest first, so a tie
    # keeps the smaller shift, scored by its correlation with the coded window over both
    # channels, divided by its norm so a cut with more energy does not score higher. A
    # wider search goes wrong on a held note: a louder period further away matches too.
    cuts = [
        padded[:, CODEC_PAD + offset : CODEC_PAD + offset + samples]
        for offset in sorted(range(-4, 5), key=abs)
    ]
    target = coded.astype(np.float64).ravel()

    def match(cut: np.ndarray) -> float:
        cut = cut.astype(np.float64).ravel()
        return cut @ target / np.sqrt(cut @ cut)

    clean = max(cuts, key=match)
    return torch.from_numpy(clean.copy()), torch.from_numpy(coded.copy())


def random_eq(audio: np.ndarray, *, q_max: float, max_gain_db: float) -> np.ndarray:
    """`audio`, float32 (2, samples) at 44.1 kHz, through one random EQ drawn as
    Jeon et al. 2026 draw theirs (Algorithm 1 and Table 3; `ind_eq_aug` and
    `mixbus_eq_aug` in merlresearch/embracing-cacophony's dataloader.py, whose
    pedalboard filters these are): a low shelf at 50 to 150 Hz, 0 to 4 peaks,
    and a high shelf at 6 to 12 kHz. Every Q is log-uniform from 0.7 to `q_max`,
    and every gain uniform from -`max_gain_db` to +`max_gain_db`. The peaks'
    centres are their code's: half the running sums of edges spaced evenly on a
    log scale between the shelves, each moved by a Gaussian of deviation
    centre**0.75 and held to 50 Hz..10 kHz. Stems take Q up to 5 and ±9 dB, the
    mix Q up to 3 and ±6 dB (Table 4).

    The filters start at rest inside the codec pad, which is cut off after
    coding.
    """
    from pedalboard import HighShelfFilter, LowShelfFilter, PeakFilter, Pedalboard

    from grooveback.cli.musdb_eval import SR

    def uniform(low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand(()))

    def q() -> float:
        return 0.7 * (q_max / 0.7) ** float(torch.rand(()))

    def gain_db() -> float:
        return uniform(-max_gain_db, max_gain_db)

    log_low = uniform(np.log10(50), np.log10(150))
    log_high = uniform(np.log10(6_000), np.log10(12_000))
    peaks = int(torch.randint(0, 5, ()))
    filters = [LowShelfFilter(cutoff_frequency_hz=10**log_low, gain_db=gain_db(), q=q())]
    # With 0 peaks there is one edge and no centre.
    edges = np.logspace(log_low, log_high, num=peaks + 1)
    for centre in np.cumsum(edges)[1:] / 2:
        freq = min(max(50.0, centre + centre**0.75 * float(torch.randn(()))), 10_000.0)
        filters.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain_db(), q=q()))
    filters.append(HighShelfFilter(cutoff_frequency_hz=10**log_high, gain_db=gain_db(), q=q()))
    return Pedalboard(filters)(audio, SR)


class TrackDataset(torch.utils.data.Dataset):
    """Fresh windows of real tracks, each coded at random (`coded_pair`).

    Reads every file `glob` matches under `tracks_dir` into memory once, as
    16-bit integers: MUSDB18-HQ's `*/mixture.wav` by default, or `*.wav` for a
    flat folder of songs. Every item is a window of a random track, at a random
    start on a 10 ms grid among those that pass Apollo's activity check
    (`active_starts`), drawn anew on every read: the dataset has no fixed
    windows, so it has no length of its own (`InterleavedDataset` sets one).
    A track is a finished mix with no stems, so of `augmentations` it only takes
    the swap of the whole window and the EQ on the mix.

    Items are audio, not latents: the DataLoader's worker processes build them
    on the CPU, and the training loop encodes them on the GPU. Draws use torch's
    random generator, which the DataLoader seeds differently in every worker.
    """

    def __init__(
        self,
        tracks_dir: str | Path,
        *,
        glob: str,
        window_samples: int,
        augmentations: Augmentations = NO_AUGMENTATIONS,
    ):
        self.window_samples = window_samples
        self.augmentations = augmentations
        paths = sorted(Path(tracks_dir).glob(glob))
        self.tracks = load_active_tracks(paths, window_samples=window_samples)
        if not self.tracks:
            raise ValueError(
                f"{tracks_dir}: none of the {len(paths)} files {glob!r} matches has a window "
                "that passes the activity check."
            )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean and coded audio of one fresh window, both (2, 438_272) float32."""
        padded = draw_window(self.tracks, window_samples=self.window_samples)
        if chance(self.augmentations.swap):
            padded = padded[[1, 0]]
        if chance(self.augmentations.mix_eq):
            padded = random_eq(padded, q_max=3.0, max_gain_db=6.0)
        return coded_pair(padded, augmentations=self.augmentations)


class RemixDataset(torch.utils.data.Dataset):
    """Fresh remixes of MUSDB18-HQ stems, each coded at random (`coded_pair`).

    Reads `{song}/{bass,drums,other,vocals}.wav` into memory once, as 16-bit
    integers: the 86 training songs' stems take about 14 GB. Every item sums one
    stem of each kind, each from its own random song and start, and each scaled
    by a gain uniform from 0.25 to 1.25, as Demucs's `remix` and `scale` do
    (facebookresearch/demucs, conf/config.yaml). With `augmentations`, a remix
    can leave out one kind, swap left and right of each stem on its own, as
    Demucs's `flip` does, and give each stem its own EQ.

    Windows come only from stretches that pass Apollo's activity check
    (`active_starts`): measured on 67 of the songs, 30% of random 10 s vocal
    windows fail it, and 15% are below -60 dBFS. Like `TrackDataset`, it draws
    anew on every read. A remix past full scale is scaled down before coding,
    and quieter ones keep their level, as validation and inference windows do.
    """

    STEMS = ("bass", "drums", "other", "vocals")

    def __init__(
        self,
        musdb_dir: str | Path,
        *,
        window_samples: int,
        augmentations: Augmentations = NO_AUGMENTATIONS,
    ):
        self.window_samples = window_samples
        self.augmentations = augmentations
        songs = sorted(path for path in Path(musdb_dir).iterdir() if path.is_dir())
        if not songs:
            raise FileNotFoundError(f"No song folders in {musdb_dir}.")
        self.stems = {
            kind: load_active_tracks(
                [song / f"{kind}.wav" for song in songs], window_samples=window_samples
            )
            for kind in self.STEMS
        }
        # Remixes draw from the stem kinds some song has active: one vocal-less song
        # among others leaves vocals in, a set of instrumentals takes them out.
        self.kinds = [kind for kind in self.STEMS if self.stems[kind]]
        if not self.kinds:
            raise ValueError(f"{musdb_dir}: no stem has a window that passes the activity check.")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean and coded audio of one fresh remix, both (2, 438_272) float32."""
        kinds = list(self.kinds)
        # The kind left out is picked at random; a set of one kind keeps it.
        if len(kinds) > 1 and chance(self.augmentations.drop_kind):
            del kinds[int(torch.randint(len(kinds), ()))]
        stem_eq = chance(self.augmentations.stem_eq)
        padded = sum(self.draw_stem(kind, eq=stem_eq) for kind in kinds)
        if chance(self.augmentations.mix_eq):
            padded = random_eq(padded, q_max=3.0, max_gain_db=6.0)
        return coded_pair(padded, augmentations=self.augmentations)

    def draw_stem(self, kind: str, *, eq: bool) -> np.ndarray:
        """A padded window of a random stem of `kind`, swapped with probability
        `swap`, scaled by a gain from 0.25 to 1.25, and through its own random EQ
        when `eq`."""
        stem = draw_window(self.stems[kind], window_samples=self.window_samples)
        if chance(self.augmentations.swap):
            stem = stem[[1, 0]]
        stem = stem * (0.25 + float(torch.rand(())))
        if eq:
            stem = random_eq(stem, q_max=5.0, max_gain_db=9.0)
        return stem


class InterleavedDataset(torch.utils.data.Dataset):
    """Items from several datasets, each read from one of them picked at random
    with `probabilities`, as Hugging Face's `interleave_datasets` does with its
    `probabilities`. `windows` items make an epoch.

    The sources draw a fresh window on every read, so the index only counts.
    """

    def __init__(self, sources: list, *, probabilities: list[float], windows: int):
        self.sources = sources
        self.probabilities = torch.tensor(probabilities, dtype=torch.float64)
        self.windows = windows

    def __len__(self) -> int:
        return self.windows

    def __getitem__(self, index: int):
        source = int(torch.multinomial(self.probabilities, 1))
        return self.sources[source][index]
