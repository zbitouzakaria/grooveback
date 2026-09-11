"""Convert HP-codecX's torch.package checkpoints to weights format, in place.

Run by scripts/hpcodecx_setup.sh inside the hpcodecx venv, from the clone's
root. The Zenodo packages froze the model classes as they were at training
time — before the repo's inference helpers existed (the packaged
ResidualVectorQuantize has no `from_codes`, which predict.py calls).
audiotools' `BaseModel.load` falls back from a package to a weights dict
and then instantiates the *current* repo classes, so rewriting each file in
the weights format lets the release's predict.py run unmodified (ADR-0013).

Idempotent: an already-converted file is only checked to reload.
"""

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

import torch  # noqa: E402

from train_codecx import DAC, TransformerModel  # noqa: E402

CHECKPOINTS = [
    (DAC, "runs/hp-codec/best_finetuning/dac/package.pth"),
    (TransformerModel, "runs/hp-codecx/best/transformermodel/package.pth"),
]

for cls, path in CHECKPOINTS:
    try:
        packaged = cls._load_package(path)
    except Exception:
        packaged = None  # not a torch.package: assume already converted
    if packaged is not None:
        metadata = getattr(packaged, "metadata", None)
        if not isinstance(metadata, dict) or "kwargs" not in metadata:
            raise SystemExit(
                f"{path}: the package carries no init kwargs; cannot convert."
            )
        # Write-then-rename so a failed save cannot destroy the download.
        torch.save(
            {"metadata": metadata, "state_dict": packaged.state_dict()},
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)
    # The reload proves the current code accepts the saved kwargs and state.
    cls.load(path)
    print(f"{path}: weights format, current-code reload ok", flush=True)
