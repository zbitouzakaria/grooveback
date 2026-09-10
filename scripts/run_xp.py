"""The whole benchmark, one entry point.

  uv run python scripts/run_xp.py [device=cuda] [sources=[codec]] [anchors=[]]

Cut one chunk per source, make MP3 twins at three bitrates, restore each twin
with every method, and score everything against the original chunk. Methods
are the autoencoder round-trips, a mean-damage subtraction, and the chained
{anchor}-{ae} renders (the autoencoder run on an anchor's output), anchored
by apollo and a2sb (ADR-0011):

  artifacts/xp/{source}/original.wav             the clean chunk
  artifacts/xp/{source}/{bitrate}/input.wav      the MP3 round-trip
  artifacts/xp/{source}/{bitrate}/{method}.wav   one render per method
  artifacts/xp/{source}/{bitrate}/listen/        level-matched FLAC to A/B
  artifacts/xp/_donor/{bitrate}/damage-{ae}.npy  mean damage directions
  artifacts/xp/results.json                      the metrics per render

Renders and twins are float WAV: decoder overshoot and loudness matching
both push peaks past full scale, which integer formats would clip silently
(ga.save refuses). Only the listening packs, pulled under -1 dBFS by one
common gain, are FLAC.

Everything in `configs/benchmark.yaml` is overridable from the CLI, which is
how a GPU pod renders one slice (`anchors=[]`) while another takes a2sb only
(`autoencoders=[] anchors=[a2sb]`). A render is skipped when its file exists,
so re-running is safe and renders produced on a pod are picked up as-is.
Scoring always re-runs, over whatever renders exist.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from grooveback import audio as ga
from grooveback import latents as gl
from grooveback.baselines import load_apollo, run_a2sb, run_apollo
from grooveback.evaluation import (
    best_lag,
    bss_sdr_db,
    codec_edge_hz,
    log_spectral_distance_db,
    sdr_db,
    si_snr_db,
    spectral_snr_db,
    write_listening_pack,
)
from grooveback.solvers import latent_sub, roundtrip

SR = 44_100
SOURCES = {
    "codec": ("data/original_codec_wav.wav", 0.0, 6.0),
    "aerofunk": ("data/Aerofunk - Nice One (Cpu Cant Hack It Mix) 258.wav", 0.0, 180.0),
    # One source/donor pair by a single artist, reached through neutral
    # symlinks (the files are gitignored and unnamed here): the pair measures
    # whether a damage direction learned on one track transfers to another by
    # the same hand (ADR-0011). The donor is never scored itself — the
    # default `sources` list in configs/benchmark.yaml leaves it out.
    "same-artist": ("data/same-artist/source.wav", 0.0, 180.0),
    "_donor": ("data/same-artist/donor.wav", 0.0, 180.0),
}
"""name -> (path, start_s, duration_s). One chunk per source, one fixed rule."""

XP = Path("artifacts/xp")


def render_path(name: str, bitrate: str, method: str) -> Path:
    return XP / name / bitrate / f"{method}.wav"


def band_above(x: np.ndarray, lo_hz: float) -> np.ndarray:
    """The content of `x` at and above `lo_hz` — the codec's fill zone."""
    spectrum = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / SR)
    spectrum[..., freqs < lo_hz] = 0
    return np.fft.irfft(spectrum, n=x.shape[-1], axis=-1).astype(np.float32)


def cached(path: Path) -> np.ndarray | None:
    """An existing render, or None if it has to be made."""
    return ga.load(path)[0] if path.exists() else None


def cut_chunk(name: str) -> np.ndarray:
    path, start_s, duration_s = SOURCES[name]
    expected = int(duration_s * SR)
    wav = XP / name / "original.wav"
    if wav.exists():
        chunk = ga.load(wav)[0]
        if chunk.shape[1] != expected:
            raise RuntimeError(
                f"{wav} is {chunk.shape[1]} samples, the cut now says {expected} "
                f"— delete artifacts/xp/{name} to re-cut."
            )
        return chunk
    source, rate = ga.load(path)
    if rate != SR:
        # 48 kHz masters land on the benchmark grid once, identically for
        # every method downstream (ADR-0011).
        print(f"resample {name}: {rate} -> {SR} Hz", flush=True)
        source = ga.resample(source, sr_in=rate, sr_out=SR)
    start = int(start_s * SR)
    chunk = source[:, start : start + expected]
    if chunk.shape[1] < expected:
        raise ValueError(f"{path} is too short for {duration_s:.0f} s from {start_s:.0f} s.")
    ga.save(wav, chunk, SR)
    return chunk


def build_twin(name: str, bitrate: str, original: np.ndarray) -> np.ndarray:
    """MP3-compress the chunk and read it back, verified sample-aligned."""
    wav = XP / name / bitrate / "input.wav"
    if not wav.exists():
        mp3 = wav.with_suffix(".mp3")
        mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(XP / name / "original.wav"),
                        "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        "-c:a", "pcm_f32le", str(wav)], check=True)
    degraded = ga.load(wav)[0][:, : original.shape[1]]
    # One sample of shift would wreck the waveform metrics, so refuse to score
    # a twin that is not exactly aligned. ffmpeg's decoder honours the LAME
    # header, so in practice the round-trip lands at lag 0.
    lag = best_lag(original[:, : degraded.shape[1]], degraded)
    if lag != 0:
        raise RuntimeError(f"{wav} is {lag} samples off its original.")
    return degraded


def damage_vector(ae: str, bitrate: str, model) -> np.ndarray:
    """The mean damage direction for one autoencoder at one bitrate, cached.

    mean over frames of encode(mp3(donor)) - encode(donor): points clean ->
    damaged, so the solver subtracts it (ADR-0011).
    """
    npy = XP / "_donor" / bitrate / f"damage-{ae}.npy"
    if npy.exists():
        return np.load(npy)
    donor = cut_chunk("_donor")
    twin = build_twin("_donor", bitrate, donor)
    clean = gl.ae_encode(ae, donor, SR, model)
    degraded = gl.ae_encode(ae, twin, SR, model)
    damage = gl.mean_damage(clean, degraded).astype(np.float32)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, damage)
    print(f"damage {ae} {bitrate}: |d| = {float(np.linalg.norm(damage)):.3f}", flush=True)
    return damage


def ae_tags(ae: str, chains: list[str]) -> list[str]:
    """The render names one autoencoder produces: the round-trip, the
    round-trip of each chained anchor's render, and the damage subtraction."""
    return [ae] + [f"{chain}-{ae}" for chain in chains] + [f"{ae}-sub"]


@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig) -> None:
    chunks = {name: cut_chunk(name) for name in cfg.sources}
    twins = {(name, bitrate): build_twin(name, bitrate, chunks[name])
             for name in cfg.sources for bitrate in cfg.bitrates}
    # The frequency the codec actually kept, exact on synthetic twins. A2SB is
    # walled here rather than at its own detected knee: the knee detector
    # exists for real rips with smeared rolloffs, and on a sharp LAME edge it
    # lands below the edge and deletes real content.
    edges = {key: codec_edge_hz(chunks[key[0]], twins[key], sample_rate=SR)
             for key in twins}

    # Anchors render first — the chained {anchor}-{ae} renders read them.
    if "apollo" in cfg.anchors:
        todo = [key for key in twins if not render_path(*key, "apollo").exists()]
        if todo:
            model = load_apollo(device=cfg.device)
            for name, bitrate in todo:
                print(f"render apollo: {name} {bitrate}", flush=True)
                out = run_apollo(twins[(name, bitrate)], SR, model=model)
                ga.save(render_path(name, bitrate, "apollo"), out, SR)
            del model

    if "a2sb" in cfg.anchors:
        for name, bitrate in twins:
            if not render_path(name, bitrate, "a2sb").exists():
                print(f"render a2sb: {name} {bitrate} "
                      f"(wall at {edges[(name, bitrate)] - 250:.0f} Hz)", flush=True)
                # 50 steps is the paper's default; ADR-0006 uses it as
                # canonical. The wall sits one band under the measured edge so
                # the model sees full-level content right up to a sharp,
                # training-matched edge.
                out = run_a2sb(twins[(name, bitrate)], SR, n_steps=50,
                               cutoff_hz=edges[(name, bitrate)] - 250,
                               device="mps" if cfg.device == "auto" else cfg.device)
                ga.save(render_path(name, bitrate, "a2sb"), out, SR)

    # Autoencoder arms, loading each model at most once.
    for ae in cfg.autoencoders:
        todo = [(name, bitrate, tag)
                for name, bitrate in twins
                for tag in ae_tags(ae, cfg.chains)
                if not render_path(name, bitrate, tag).exists()]
        if not todo:
            continue
        model = gl.load_ae(ae, device=cfg.device)
        for name, bitrate, tag in todo:
            print(f"render {tag}: {name} {bitrate}", flush=True)
            twin = twins[(name, bitrate)]
            if tag.endswith("-sub"):
                out = latent_sub(model, twin, SR, ae=ae,
                                 damage=damage_vector(ae, bitrate, model))
            elif tag == ae:
                out = roundtrip(model, twin, SR, ae=ae)
            else:
                chain = tag[: -len(ae) - 1]
                source = cached(render_path(name, bitrate, chain))
                if source is None:
                    raise RuntimeError(
                        f"{chain} render missing for {name} {bitrate}; "
                        "render the anchors first."
                    )
                out = roundtrip(model, source, SR, ae=ae)
            ga.save(render_path(name, bitrate, tag), out, SR)
        del model
        # The allocator keeps the freed model's blocks cached; the next
        # autoencoder then OOMs on a card the two would separately fit.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Score whatever exists against the original, and write listening sets.
    methods = [tag for ae in cfg.autoencoders for tag in ae_tags(ae, cfg.chains)]
    methods += list(cfg.anchors)
    results: dict = {}
    for name in cfg.sources:
        results[name] = {}
        for bitrate in cfg.bitrates:
            pack = {"original": chunks[name], "input": twins[(name, bitrate)]}
            for method in methods:
                render = cached(render_path(name, bitrate, method))
                if render is not None:
                    pack[method] = render
            shortest = min(item.shape[1] for item in pack.values())
            pack = {label: item[:, :shortest] for label, item in pack.items()}
            edge = edges[(name, bitrate)]
            fill_master = band_above(pack["original"], edge)
            results[name][bitrate] = {"edge_hz": edge}
            for label, item in pack.items():
                if label == "original":
                    continue
                fill = band_above(item, edge)
                results[name][bitrate][label] = {
                    "bss_sdr_db": round(bss_sdr_db(pack["original"], item), 2),
                    "sdr_db": round(sdr_db(pack["original"], item), 2),
                    "si_snr_db": round(si_snr_db(pack["original"], item), 2),
                    "spectral_snr_db": round(
                        spectral_snr_db(pack["original"], item), 2),
                    "lsd_db": round(
                        log_spectral_distance_db(pack["original"], item), 2),
                    # The band the codec removed, on its own. Silence scores
                    # 0/0 on the first two; a fill with the right texture but
                    # unaligned phase is negative on waveform SDR, positive on
                    # spectral SNR, and far below silence on the distance.
                    "fill_sdr_db": round(sdr_db(fill_master, fill), 2),
                    "fill_spectral_snr_db": round(
                        spectral_snr_db(fill_master, fill), 2),
                    "fill_lsd_db": round(
                        log_spectral_distance_db(fill_master, fill), 2),
                }
            write_listening_pack(pack, SR, XP / name / bitrate / "listen",
                                 suffix=".flac")
            print(name, bitrate, results[name][bitrate], flush=True)

    (XP / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {XP / 'results.json'}")


if __name__ == "__main__":
    main()
