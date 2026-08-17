"""Audio io, loudness, and spectrograms.

No torch, no GPU, no network — this module stays testable on its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

TARGET_LUFS = -14.0
"""Everything is level-matched here before any comparison. See ADR-0005."""


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 `(channels, samples)`.

    Always 2-D, even for mono, so downstream code never branches on shape.
    """
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(audio.T), sample_rate


def save(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write `(channels, samples)` audio, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sample_rate, subtype="FLOAT")


def loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770)."""
    meter = pyln.Meter(sample_rate)
    return float(meter.integrated_loudness(audio.T))


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target_lufs: float = TARGET_LUFS
) -> np.ndarray:
    """Scale to `target_lufs`.

    Gain only — no limiting — so this can push peaks above 1.0 on already-loud
    masters. `peak_dbfs` will show it; that is information, not something to
    silently fix, since clipping the output would change what is being judged.
    """
    gain = 10.0 ** ((target_lufs - loudness(audio, sample_rate)) / 20.0)
    return audio * gain


def peak_dbfs(audio: np.ndarray) -> float:
    """Peak level in dBFS, or -inf for digital silence."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20.0 * np.log10(peak) if peak > 0 else float("-inf")


def spectrogram_db(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    floor_db: float = -100.0,
    max_frames: int = 4000,
) -> np.ndarray:
    """Magnitude spectrogram in dBFS, averaged across channels.

    Calibrated so a full-scale sine reads 0 dB (`2/sum(window)` scaling) —
    without it the plot disagrees with any real analyser. Returns
    `(freq_bins, frames)` for `imshow(origin="lower")`. `max_frames` widens
    the hop on long input so a full track stays plottable.
    """
    mono = np.ascontiguousarray(audio.mean(axis=0))
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.size))

    hop = max(hop, -(-(mono.size - n_fft) // max(max_frames, 1)))
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (mono.size - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        mono,
        shape=(n_frames, n_fft),
        strides=(mono.strides[0] * hop, mono.strides[0]),
    )
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1)).T * (2.0 / window.sum())
    return 20.0 * np.log10(np.maximum(magnitude, 10.0 ** (floor_db / 20.0)))


def band_energy_db(
    audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> float:
    """RMS level in dBFS of the content between `low_hz` and `high_hz`.

    The float64 cast is load-bearing: float32 FFT roundoff at full-track
    lengths buries quiet bands by tens of dB.
    """
    mono = audio.mean(axis=0).astype(np.float64)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    band = spectrum[(freqs >= low_hz) & (freqs < high_hz)]
    if band.size == 0:
        return float("-inf")
    # Parseval: band power back to a time-domain RMS over the whole signal.
    rms = np.sqrt(2.0 * np.sum(np.abs(band) ** 2)) / mono.size
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")
