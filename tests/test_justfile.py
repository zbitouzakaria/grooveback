"""The just recipes, run by real `just` with a stand-in `uv` that records what it receives."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

JUSTFILE = Path(__file__).resolve().parents[1] / "justfile"


def run_recipe(tmp_path, *arguments):
    """`just {arguments}` with a `uv` on PATH that writes its arguments, one per line, to
    tmp_path/arguments, and the thread limits it was given to tmp_path/threads."""
    if shutil.which("just") is None:
        pytest.skip("just is not installed")
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{tmp_path}/arguments"\n'
        f'printf "%s\\n" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" > "{tmp_path}/threads"\n'
    )
    uv.chmod(0o755)
    subprocess.run(
        ["just", "--justfile", JUSTFILE, *arguments],
        env=os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("recipe", "command"),
    [
        ("train", ["run", "python", "-m", "grooveback.cli.train_lora"]),
        ("evaluate", ["run", "python", "-m", "grooveback.cli.evaluate"]),
        ("test", ["run", "pytest"]),
    ],
)
def test_a_recipe_hands_its_arguments_over_exactly_as_typed(tmp_path, recipe, command):
    """Brackets are a shell pattern and the space splits a word, so a recipe that pastes
    its arguments into the command line changes them."""
    (tmp_path / "seeds=0").touch()

    run_recipe(tmp_path, recipe, "seeds=[0]", "save_dir=two words")

    assert (tmp_path / "arguments").read_text().splitlines() == [
        *command,
        "seeds=[0]",
        "save_dir=two words",
    ]


def test_a_flag_reaches_pytest_rather_than_just(tmp_path):
    run_recipe(tmp_path, "test", "-q", "-k", "train")

    assert (tmp_path / "arguments").read_text().splitlines() == [
        "run",
        "pytest",
        "-q",
        "-k",
        "train",
    ]


@pytest.mark.parametrize(("recipe", "threads"), [("train", "2"), ("evaluate", "8")])
def test_a_gpu_recipe_limits_torch_to_a_few_threads(tmp_path, recipe, threads):
    """Torch on a pod sees 128 to 256 cores while the container may use 13 to 27."""
    run_recipe(tmp_path, recipe)

    assert (tmp_path / "threads").read_text().splitlines() == [threads, threads]
