"""SAME encode/decode driver, run inside the stable-audio-3 venv.

Kept dependency-light on purpose: grooveback shells out to it the same way it
does to the A2SB fork, so version pins never collide.

  .../third_party/stable-audio-3/.venv/bin/python scripts/same_codec.py \
      --model same-s --input in.wav --decoded out.wav --latents z.npy

Latents can also be fed back in place of audio (--from-latents z.npy) to
decode modified latents, which is what the transport-vector experiment needs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "stable-audio-3"))

MODELS = {"same-s": "stabilityai/SAME-S", "same-l": "stabilityai/SAME-L"}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--input", help="wav to encode")
    ap.add_argument("--from-latents", help="npy latents to decode instead of encoding")
    ap.add_argument("--latents", help="npy path to store latents")
    ap.add_argument("--decoded", help="wav path for the decoded audio")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from stable_audio_3.loading_utils import load_autoencoder

    device = pick_device(args.device)
    repo = MODELS[args.model]
    config_path = hf_hub_download(repo, "model_config.json")
    ckpt_path = hf_hub_download(repo, "model.safetensors")
    model = load_autoencoder(config_path, ckpt_path, device=device)
    model.eval()
    sample_rate = json.load(open(config_path))["sample_rate"]
    print(f"same_codec: {args.model} on {device}", flush=True)

    with torch.inference_mode():
        if args.from_latents:
            z = torch.from_numpy(np.load(args.from_latents)).to(device)
            if z.ndim == 2:
                z = z.unsqueeze(0)
        else:
            audio, in_sr = sf.read(args.input, dtype="float32", always_2d=True)
            if in_sr != sample_rate:
                raise SystemExit(f"input is {in_sr} Hz, model wants {sample_rate}")
            x = torch.from_numpy(audio.T).unsqueeze(0).to(device)
            t0 = time.time()
            z = model.encode_audio(x)
            print(f"same_codec: encoded {x.shape[-1]} samples -> {tuple(z.shape)} "
                  f"in {time.time() - t0:.1f}s", flush=True)
            if args.latents:
                Path(args.latents).parent.mkdir(parents=True, exist_ok=True)
                np.save(args.latents, z.squeeze(0).cpu().numpy())

        if args.decoded:
            t0 = time.time()
            y = model.decode_audio(z)
            print(f"same_codec: decoded -> {tuple(y.shape)} in {time.time() - t0:.1f}s",
                  flush=True)
            out = y.squeeze(0).cpu().numpy().T
            Path(args.decoded).parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.decoded, out, sample_rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
