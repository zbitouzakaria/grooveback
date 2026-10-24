"""Restore every WAV under a directory with A2SB, loading the model once.

    third_party/a2sb/.venv/bin/python scripts/a2sb_restore.py mp3/ a2sb/ --cutoff-hz 10750

Runs in the A2SB fork's environment (third_party/a2sb) and uses its
restore.py: each channel is cut off above `--cutoff-hz` and restored on its own,
as restore.py does for one file. Here every channel of every file goes into a
single run of the fork's inference, so the checkpoints load once. A file whose
render already exists is skipped.

One run has one cut-off. On the MP3s of one bitrate it is the codec's, the same
for every clip; scripts/run_xp.py passes it 250 Hz under the measured codec
edge, so the model sees full-level content up to a sharp edge.

The fork's inverse STFT returns a whole number of 512-sample hops, which cut the
last 408 samples off a 6 s clip. Each channel is therefore padded with zeros to
a whole number of hops, and its render cut back to the channel's length.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

A2SB_DIR = Path(__file__).resolve().parents[1] / "third_party" / "a2sb"
HOP = 512  # hop_length of the fork's STFT, configs/inference_files_upsampling.yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cutoff-hz", type=float, required=True)
    parser.add_argument("--steps", type=int, default=50, help="the paper's default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--predict-batch-size", type=int, default=16, help="memory knob only; upstream's default"
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(A2SB_DIR))
    from input_prep import brickwall_lowpass
    from restore import SAMPLE_RATE, build_config, fetch_checkpoints

    todo = [
        path
        for path in sorted(args.input_dir.rglob("*.wav"))
        if not (args.output_dir / path.relative_to(args.input_dir)).exists()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        filelist = []
        for i, path in enumerate(todo):
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != SAMPLE_RATE:
                raise ValueError(f"A2SB expects {SAMPLE_RATE} Hz, got {rate} Hz in {path}.")
            for c, channel in enumerate(audio.T):
                wav = tmp / f"in_{i}_c{c}.wav"
                walled = brickwall_lowpass(channel, rate, args.cutoff_hz)
                sf.write(wav, np.pad(walled, (0, -walled.size % HOP)), rate, subtype="FLOAT")
                filelist.append({"filepath": str(wav), "output_subdir": f"{i}/c{c}"})

        checkpoints = fetch_checkpoints(single_split=False)
        config = tmp / "override.yaml"
        config.write_text(
            yaml.safe_dump(build_config(filelist, checkpoints, args.device, args.cutoff_hz, tmp))
        )
        subprocess.run(
            [
                sys.executable, "ensembled_inference_api.py", "predict",
                "-c", "configs/ensemble_2split_sampling.yaml",
                "-c", "configs/inference_files_upsampling.yaml",
                "-c", str(config),
                f"--model.predict_n_steps={args.steps}",
                f"--model.predict_batch_size={args.predict_batch_size}",
                f"--model.output_audio_filename={tmp / 'out' / 'recon.wav'}",
                "--model.output_per_input=true",
            ],
            cwd=A2SB_DIR,
            check=True,
        )  # fmt: skip

        for i, path in enumerate(todo):
            info = sf.info(path)
            out = np.zeros((info.channels, info.frames), dtype=np.float32)
            for c in range(info.channels):
                recon_path = tmp / "out" / str(i) / f"c{c}" / "recon.wav"
                recon, _ = sf.read(recon_path, dtype="float32", always_2d=True)
                if recon.shape[0] < info.frames:
                    raise RuntimeError(
                        f"{recon_path} has {recon.shape[0]} samples, {path} has {info.frames}."
                    )
                out[c] = recon[: info.frames, 0]
            target = args.output_dir / path.relative_to(args.input_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, out.T, info.samplerate, subtype="FLOAT")
        print(f"restore: {len(todo)} files to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
