"""Convert HP-codecX's torch.package checkpoints to weights format, in place.

Run by scripts/hpcodecx_setup.sh inside the hpcodecx venv, from the clone's
root. Two release defects make this necessary (ADR-0013):

- The Zenodo packages froze the model classes as they were at training time,
  before the repo's inference helpers existed — the packaged
  ResidualVectorQuantize has no `from_codes`, which predict.py calls.
  audiotools' `BaseModel.load` falls back from a package to a weights dict
  and then instantiates the *current* repo classes, so the files are
  rewritten in that format.
- The packaged metadata's kwargs also predate the current `__init__`
  signatures (its `encoder_dims` holds what is now `latent_dims`;
  constructing with it allocates a network hundreds of times larger until
  the OOM killer fires). The kwargs therefore come from the release's own
  `conf/codecx/hpcodecx.yml`, and every conversion is gated by a state-dict
  load into the current class that must match key for key.

Idempotent: re-running re-derives the kwargs and re-checks the load.
"""

import inspect
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402

from train_codecx import DAC, TransformerModel  # noqa: E402

with open("conf/codecx/hpcodecx.yml") as f:
    CONF = yaml.safe_load(f)

CHECKPOINTS = [
    (DAC, "runs/hp-codec/best_finetuning/dac/package.pth"),
    (TransformerModel, "runs/hp-codecx/best/transformermodel/package.pth"),
]

for cls, path in CHECKPOINTS:
    prefix = cls.__name__ + "."
    conf = {k[len(prefix):]: v for k, v in CONF.items() if k.startswith(prefix)}
    kwargs = {k: v for k, v in conf.items() if k in inspect.signature(cls).parameters}

    try:
        state = cls._load_package(path).state_dict()
    except Exception:
        state = torch.load(path, "cpu")["state_dict"]  # already weights format

    model = cls(**kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{path}: {len(missing)} missing / {len(unexpected)} unexpected keys "
            f"— the conf kwargs do not describe this checkpoint. "
            f"missing: {missing[:3]} unexpected: {unexpected[:3]}"
        )

    # Write-then-rename so a failed save cannot destroy the download.
    torch.save({"metadata": {"kwargs": kwargs}, "state_dict": state}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"{path}: weights format with conf kwargs, current-code load ok", flush=True)
