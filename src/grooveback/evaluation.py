"""Building comparison sets that are fair to listen to.

No torch, no GPU, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grooveback import audio as ga


def level_matched_set(
    items: dict[str, np.ndarray],
    sample_rate: int,
    target_lufs: float = ga.TARGET_LUFS,
    ceiling_dbfs: float = -1.0,
) -> dict[str, np.ndarray]:
    """Level-match every item, then pull the whole set down clear of full scale.

    Each item is normalised to `target_lufs` so no comparison is decided by
    loudness. Restoration models routinely push peaks past 0 dBFS afterwards —
    Apollo reaches +2.9 dBFS on this material — which clips in any integer
    playback path and adds distortion that has nothing to do with the model.

    The headroom gain is **one value applied to every item**, not per-item
    normalisation. Scaling individually would undo the level matching and put
    the loudest-sounding file back in front regardless of quality.
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
