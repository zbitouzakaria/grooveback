"""Encode every twin window into SAME latents.

Batched deliberately: one model load for the whole set, rather than paying it
per file through the subprocess boundary. Runs inside the stable-audio-3 venv.

  third_party/stable-audio-3/.venv/bin/python scripts/encode_twins.py [device]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "stable-audio-3"))

from huggingface_hub import hf_hub_download  # noqa: E402
from stable_audio_3.loading_utils import load_autoencoder  # noqa: E402

def main(device: str, variant: str = "same-s") -> None:
    repo_id = f"stabilityai/{variant.upper()}"
    model = load_autoencoder(
        hf_hub_download(repo_id, "model_config.json"),
        hf_hub_download(repo_id, "model.safetensors"),
        device=device,
    ).eval()

    cut = REPO / "data" / "twins" / "cut"
    start, done = time.time(), 0
    with torch.inference_mode():
        for wav in sorted(cut.rglob("*.wav")):
            npy = wav.with_suffix(f".{variant}.npy")
            if npy.exists():
                continue
            audio, _ = sf.read(wav, dtype="float32", always_2d=True)
            z = model.encode_audio(torch.from_numpy(audio.T).unsqueeze(0).to(device))
            np.save(npy, z.squeeze(0).cpu().numpy())
            done += 1
    print(f"encoded {done} windows in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mps",
         sys.argv[2] if len(sys.argv) > 2 else "same-s")
