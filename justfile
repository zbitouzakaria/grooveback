# Arguments reach Hydra and pytest exactly as typed: `just evaluate 'seeds=[0]'`.
set positional-arguments
# HF_TOKEN, for the gated text encoder Stable Audio 3 loads.
set dotenv-load

# List the commands
default:
    @just --list

# Train the LoRA (configs/train_lora.yaml)
train *overrides:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python -m grooveback.cli.train_lora "$@"

# Score a checkpoint on the test clips (configs/evaluate.yaml)
evaluate *overrides:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python -m grooveback.cli.evaluate "$@"

# Run the unit tests
test *args:
    uv run pytest "$@"
