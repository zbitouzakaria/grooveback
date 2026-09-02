"""Scoring restorations against a reference, and building comparison sets
that are fair to listen to.

No torch, no GPU, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grooveback import audio as ga


def _as_matched_vectors(
    reference: np.ndarray, estimate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != estimate.shape:
        raise ValueError(
            f"reference {reference.shape} and estimate {estimate.shape} must have "
            "the same shape — trim to the shorter length first."
        )
    return (
        reference.astype(np.float64).ravel(),
        estimate.astype(np.float64).ravel(),
    )


def sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-distortion ratio: how far the error sits below the reference.

    Plain waveform SDR over all channels, no distortion filter. Not
    scale-invariant — a pure gain error costs SDR — and phase changes cost it
    too, so a decoder that re-realises phase scores badly however it sounds.
    """
    ref, est = _as_matched_vectors(reference, estimate)
    error_energy = float(((ref - est) ** 2).sum())
    if error_energy == 0.0:
        return float("inf")
    return 10.0 * np.log10(float((ref**2).sum()) / error_energy)


def si_snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SNR (Le Roux et al. 2019).

    Both signals are made zero-mean, then the estimate is split into its
    projection onto the reference and a residual; the ratio of the two is the
    score. Pure gain errors cost nothing; everything else is residual.
    """
    ref, est = _as_matched_vectors(reference, estimate)
    ref = ref - ref.mean()
    est = est - est.mean()
    target = (est @ ref) / (ref @ ref) * ref
    residual_energy = float(((est - target) ** 2).sum())
    if residual_energy == 0.0:
        return float("inf")
    return 10.0 * np.log10(float((target**2).sum()) / residual_energy)


def spectral_snr_db(
    reference: np.ndarray,
    estimate: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """SNR of STFT magnitudes — phase-blind.

    Credits content with the right spectral envelope even when the waveforms
    do not align, which is exactly what `sdr_db` cannot do: a generated band
    that matches the true band's texture at unrelated phase scores near or
    below 0 dB in waveform SDR and far above it here. Silence scores 0 dB
    against any reference, so positive means the estimate carries information
    about the reference's spectrum.
    """
    ref, est = _as_matched_vectors(reference, estimate)

    def magnitudes(x: np.ndarray) -> np.ndarray:
        if x.size < n_fft:
            x = np.pad(x, (0, n_fft - x.size))
        n_frames = 1 + (x.size - n_fft) // hop
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0])
        )
        return np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))

    ref_mag, est_mag = magnitudes(ref), magnitudes(est)
    error_energy = float(((ref_mag - est_mag) ** 2).sum())
    if error_energy == 0.0:
        return float("inf")
    return 10.0 * np.log10(float((ref_mag**2).sum()) / error_energy)


def codec_edge_hz(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
    band_hz: float = 250.0,
    track_db: float = 8.0,
    floor_db: float = -70.0,
) -> float:
    """Highest frequency the codec kept, measured against the clean reference.

    The last `band_hz` band where the degraded copy still tracks the
    reference within `track_db`. Bands where the reference itself has fallen
    `floor_db` below its loudest band are ignored — there both signals are
    noise floor and would "track" spuriously. Synthetic MP3 edges are sharp,
    so this is exact to one band; the smeared-knee detection in the A2SB fork
    exists for real rips without a reference, not for twins.
    """
    ref = reference.mean(axis=0).astype(np.float64)
    deg = degraded.mean(axis=0).astype(np.float64)
    n = min(ref.size, deg.size)

    def band_levels(x: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(x[:n])) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        edges = np.arange(0, sample_rate / 2, band_hz)
        return edges, np.array([
            10 * np.log10(spectrum[(freqs >= lo) & (freqs < lo + band_hz)].sum() + 1e-30)
            for lo in edges
        ])

    edges, ref_levels = band_levels(ref)
    _, deg_levels = band_levels(deg)
    audible = ref_levels > ref_levels.max() + floor_db
    tracking = audible & (ref_levels - deg_levels < track_db)
    if not tracking.any():
        return 0.0
    return float(edges[np.where(tracking)[0][-1]] + band_hz)


def best_lag(
    reference: np.ndarray,
    estimate: np.ndarray,
    max_lag: int = 4096,
    probe_samples: int = 44_100,
) -> int:
    """Offset at which `estimate` best matches `reference`, within ±`max_lag`.

    Positive means the estimate is late: drop its first `lag` samples to
    align. Codec round-trips can shift audio by the encoder delay, and one
    sample of shift is enough to wreck a waveform metric, so callers check
    this is zero before scoring.

    Correlates one probe from the middle of the estimate against the
    surrounding stretch of the reference.
    """
    ref = reference.mean(axis=0).astype(np.float64)
    est = estimate.mean(axis=0).astype(np.float64)
    n = min(ref.size, est.size)
    if n < 2 * max_lag + probe_samples:
        raise ValueError(
            f"Signals of {n} samples are too short for max_lag={max_lag} "
            f"with a {probe_samples}-sample probe."
        )
    start = (n - probe_samples) // 2
    probe = est[start : start + probe_samples]
    if not probe.any():
        raise ValueError("The middle of the estimate is silent; cannot align.")
    haystack = ref[start - max_lag : start + probe_samples + max_lag]
    correlation = np.correlate(haystack, probe, "valid")
    return max_lag - int(np.argmax(correlation))


def level_matched_set(
    items: dict[str, np.ndarray],
    sample_rate: int,
    target_lufs: float = ga.TARGET_LUFS,
    ceiling_dbfs: float = -1.0,
) -> dict[str, np.ndarray]:
    """Level-match every item, then pull the whole set down clear of full scale.

    Restoration output can peak past 0 dBFS after loudness matching, which
    clips on playback. The headroom gain is one value applied to every item —
    per-item scaling would undo the level matching.
    """
    matched = {
        name: ga.normalize_loudness(audio, sample_rate, target_lufs)
        for name, audio in items.items()
    }
    peak = max(ga.peak_dbfs(audio) for audio in matched.values())
    if peak <= ceiling_dbfs:
        return matched
    gain = 10.0 ** ((ceiling_dbfs - peak) / 20.0)
    return {name: audio * gain for name, audio in matched.items()}


def write_listening_pack(
    items: dict[str, np.ndarray],
    sample_rate: int,
    out_dir: str | Path,
    **kwargs,
) -> dict[str, Path]:
    """Write a level-matched, headroom-safe set for listening."""
    out_dir = Path(out_dir)
    prepared = level_matched_set(items, sample_rate, **kwargs)
    written = {}
    for name, audio in prepared.items():
        path = out_dir / f"{'mono_' if audio.shape[0] == 1 else ''}{name}.wav"
        ga.save(path, audio, sample_rate)
        written[name] = path
    return written
