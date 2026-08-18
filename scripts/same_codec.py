"""SAME encode/decode, run inside the stable-audio-3 venv.

Kept dependency-light on purpose: grooveback shells out to it the same way it
does to the A2SB fork, so version pins never collide.

  .../third_party/stable-audio-3/.venv/bin/python scripts/same_codec.py \
      --model same-s --input in.wav --decoded out.wav --latents z.npy

`--encode-tree` encodes every wav under a directory with one model load, which
matters because loading SAME-L takes longer than encoding a clip.
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
    ap.add_argument("--from-latents", help="npy to decode instead of encoding")
    ap.add_argument("--encode-tree", help="encode every wav under this directory")
    ap.add_argument("--latents", help="npy path for the latents")
    ap.add_argument("--decoded", help="wav path for the decoded audio")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from stable_audio_3.loading_utils import load_autoencoder

    device = pick_device(args.device)
    config_path = hf_hub_download(MODELS[args.model], "model_config.json")
    model = load_autoencoder(
        config_path, hf_hub_download(MODELS[args.model], "model.safetensors"), device
    ).eval()
    sample_rate = json.load(open(config_path))["sample_rate"]
    print(f"same_codec: {args.model} on {device}", flush=True)

    def encode(path):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != sample_rate:
            raise SystemExit(f"{path} is {rate} Hz, model wants {sample_rate}")
        z = model.encode_audio(torch.from_numpy(audio.T).unsqueeze(0).to(device))
        return z.squeeze(0).float().cpu().numpy()

    with torch.inference_mode():
        if args.encode_tree:
            start, done = time.time(), 0
            for wav in sorted(Path(args.encode_tree).rglob("*.wav")):
                npy = wav.with_suffix(f".{args.model}.npy")
                if not npy.exists():
                    np.save(npy, encode(wav))
                    done += 1
            print(f"same_codec: encoded {done} files in {time.time()-start:.0f}s")
            return

        if args.from_latents:
            latents = np.load(args.from_latents)
        else:
            latents = encode(args.input)
            if args.latents:
                Path(args.latents).parent.mkdir(parents=True, exist_ok=True)
                np.save(args.latents, latents)

        if args.decoded:
            z = torch.from_numpy(latents).unsqueeze(0).to(device)
            decoded = model.decode_audio(z).squeeze(0).float().cpu().numpy().T
            Path(args.decoded).parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.decoded, decoded, sample_rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
