"""Aligning two transfers of the same performance.

A vinyl rip and a digital rip never line up: the turntable does not run at
exactly the right speed, so the offset between them grows as the track plays.
Correcting only a constant offset leaves tens of milliseconds of slip by the
end, which is enough to make any residual meaningless.

`measure_drift` fits offset and speed together; `align_to` applies the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class Drift:
    """How one recording sits against another.

    `speed_ratio` is the reference timebase divided by this one — 1.000125
    means the source runs 0.0125% fast. `offset_samples` is what remains after
    the speed is corrected. `residual_samples` is the spread of the probes
    around the fit: small means a constant speed error explains everything,
    large means wow and flutter that no linear correction will fix.
    """

    speed_ratio: float
    offset_samples: int
    confidence: float
    residual_samples: float
    probes: int


def measure_drift(
    source: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    probe_seconds: float = 4.0,
    search_seconds: float = 3.0,
    every_seconds: float = 30.0,
    min_correlation: float = 0.3,
) -> Drift:
    """Fit how `source` drifts against `reference` by probing down the track.

    Each probe cross-correlates a short window of `source` against a wider
    window of `reference`. A straight line through the probe offsets gives the
    speed error from its slope and the constant offset from its intercept.
    """
    src, ref = source.mean(0), reference.mean(0)
    probe = int(probe_seconds * sample_rate)
    search = int(search_seconds * sample_rate)
    span = min(len(src), len(ref)) / sample_rate

    times, lags, scores = [], [], []
    for start_s in np.arange(every_seconds, span - probe_seconds - every_seconds,
                             every_seconds):
        i = int(start_s * sample_rate)
        window = src[i : i + probe]
        lo, hi = max(i - search, 0), min(i + probe + search, len(ref))
        haystack = ref[lo:hi]
        if len(haystack) < len(window) + 10 or not window.any():
            continue
        window = window - window.mean()
        haystack = haystack - haystack.mean()
        correlation = np.correlate(haystack, window, "valid")
        peak = int(np.argmax(correlation))
        norm = (np.linalg.norm(window)
                * np.linalg.norm(haystack[peak : peak + len(window)]) + 1e-12)
        times.append(start_s)
        lags.append((lo + peak) - i)
        scores.append(correlation[peak] / norm)

    keep = np.array(scores) > min_correlation
    if keep.sum() < 2:
        raise ValueError(
            f"Only {keep.sum()} of {len(scores)} probes matched above "
            f"{min_correlation}. These may not be the same recording."
        )
    times, lags = np.array(times)[keep], np.array(lags)[keep]
    slope, intercept = np.polyfit(times, lags, 1)
    return Drift(
        speed_ratio=1.0 + slope / sample_rate,
        offset_samples=int(round(intercept)),
        confidence=float(np.array(scores)[keep].mean()),
        residual_samples=float(np.std(lags - (slope * times + intercept))),
        probes=int(keep.sum()),
    )


def align_to(
    source: np.ndarray, reference: np.ndarray, sample_rate: int, drift: Drift
) -> np.ndarray:
    """Resample and shift `source` onto `reference`'s timebase.

    Returns audio the same length as `reference`, so the two can be subtracted
    directly. Uses a rational resample; at the ratios a turntable produces the
    approximation error is far below a sample.
    """
    import torchaudio  # noqa: PLC0415 — keeps the rest of this module torch-free
    import torch

    ratio = Fraction(drift.speed_ratio).limit_denominator(100_000)
    stretched = torchaudio.functional.resample(
        torch.from_numpy(np.ascontiguousarray(source)),
        orig_freq=ratio.denominator,
        new_freq=ratio.numerator,
    ).numpy()

    out = np.zeros((source.shape[0], reference.shape[1]), dtype=np.float32)
    offset = drift.offset_samples
    # `offset` is where source content starts inside the reference timeline.
    src_start, dst_start = (0, offset) if offset >= 0 else (-offset, 0)
    length = min(stretched.shape[1] - src_start, out.shape[1] - dst_start)
    if length > 0:
        out[:, dst_start : dst_start + length] = stretched[
            :, src_start : src_start + length
        ]
    return out
