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
    """Magnitude spectrogram in **dBFS**, averaged across channels.

    Calibrated so a full-scale sine reads 0 dB: the single-sided `2/sum(window)`
    scaling. Without it a raw `rfft` magnitude sits about +53 dB high at
    `n_fft=2048`, which silently shifts everything up the colour scale and makes
    faint content look loud — the plot disagrees with any real analyser.

    Plain numpy STFT so this stays dependency-light and importable anywhere.
    Returns `(freq_bins, frames)`, ready for `imshow(origin="lower")`.

    `max_frames` widens the hop on long input rather than returning a plot
    nobody can render: a seven-minute track at hop 512 is 39k frames, which is
    gigabytes of intermediate and far more columns than a screen has pixels.
    Pass a bigger value if you are analysing rather than looking.
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


def bandwidth_hz(
    audio: np.ndarray,
    sample_rate: int,
    min_hz: float = 3000.0,
    drop_db: float = 20.0,
    band_hz: float = 250.0,
    span_hz: float = 1000.0,
) -> float:
    """Where a band-limited file falls off a cliff, in Hz.

    Codec and resampling cutoffs are brick walls, so this looks for the steepest
    drop in the long-term average spectrum above `min_hz` and returns the last
    frequency still carrying signal.

    Deliberately not a spectral-rolloff percentile. In music almost all the
    energy sits below ~2 kHz, so a 99% rolloff reports ~2 kHz whatever the real
    bandwidth is — which is what A2SB's own helper does, and why its shipped
    config claims a 2 kHz cutoff for full-band material. A bandwidth-extension
    model handed that figure regenerates most of the spectrum instead of only
    the missing top.
    """
    mono = audio.mean(axis=0)
    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)

    edges = np.arange(0, sample_rate / 2, band_hz)
    levels = np.array(
        [
            10 * np.log10(max(spectrum[(freqs >= lo) & (freqs < lo + band_hz)].sum(), 1e-30))
            for lo in edges
        ]
    )

    # Compare each band against the one a span below it, not its neighbour. A
    # real codec rolloff is smeared over roughly a kilohertz, so no adjacent
    # 250 Hz pair ever shows the full drop and a per-step test finds nothing.
    span = max(1, int(round(span_hz / band_hz)))
    start = max(int(min_hz // band_hz), span)
    if start >= levels.size:
        return sample_rate / 2
    drops = levels[start:] - levels[start - span : levels.size - span]
    if drops.size == 0 or drops.min() > -drop_db:
        return sample_rate / 2
    return float(edges[start + int(np.argmin(drops))])


def band_correlation(
    audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> float:
    """Inter-channel correlation in a band: 1.0 is mono, 0.0 is uncorrelated.

    A model that restores each channel separately synthesises the two
    independently, so the recovered band can come back decorrelated. That is
    inaudible as a level change and obvious as a smeared stereo image, so it
    needs its own number.
    """
    if audio.shape[0] < 2:
        return 1.0
    spectra = np.fft.rfft(audio[:2], axis=1)
    freqs = np.fft.rfftfreq(audio.shape[1], 1.0 / sample_rate)
    band = (freqs >= low_hz) & (freqs < high_hz)
    left, right = spectra[0, band], spectra[1, band]
    denom = np.sqrt(np.sum(np.abs(left) ** 2) * np.sum(np.abs(right) ** 2))
    if denom == 0:
        return 1.0
    return float(np.abs(np.sum(left * np.conj(right))) / denom)


def band_energy_db(
    audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> float:
    """RMS level in dBFS of the content between `low_hz` and `high_hz`.

    A number to put next to a spectrogram, since a colour map is easy to read
    optimistically and this is not.
    """
    mono = audio.mean(axis=0)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    band = spectrum[(freqs >= low_hz) & (freqs < high_hz)]
    if band.size == 0:
        return float("-inf")
    # Parseval: band power back to a time-domain RMS over the whole signal.
    rms = np.sqrt(2.0 * np.sum(np.abs(band) ** 2)) / mono.size
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")
