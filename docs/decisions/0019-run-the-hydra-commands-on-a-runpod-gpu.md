# 19. Run the Hydra commands on a RunPod GPU from the Mac

Date: 2026-11-06

## Status

Accepted.

## Every GPU run needs its own scripts

A run on a rented GPU takes a script that creates the pod, one that sets it up, an upload, a watcher and a fetch. We
wrote a new set for almost every job, in a session's scratch folder, and one of them has already gone missing. They
also fail in ways that cost money. A training run stayed at NaN for 1.4 hours after the Mac slept and stopped its
watcher. A `pod delete` chained with `;` once removed a pod before its scores came home.

Zakaria wants to run and change all of this by hand. The commands should look like this:

```bash
just train                          # on this machine
just train remote=runpod lr=5e-5    # the same run, on an L40S
just train remote=runpod remote.background=true    # a long run that outlives a lost connection
just evaluate remote=runpod remote.wait_minutes=120 model.checkpoint=artifacts/lora/run1/checkpoint-46765
```

The command waits for the GPU, for as long as we allow. It attaches our volume unless told otherwise. The Hydra
endpoint then runs on the pod, and its logs and tracebacks print in the local terminal, as if it ran here. Ctrl-C stops
the job and deletes the pod. The code must stay short enough to read in one sitting.

## What we looked at

[Dora](https://github.com/facebookresearch/dora) starts experiments through Submitit on a Slurm cluster. It expects the
cluster to exist. We have no cluster, and starting the machine is the part we need.

[Hydra's launcher plugins](https://hydra.cc/docs/plugins/submitit_launcher/) look like the natural place, with
`hydra/launcher=runpod`. But Hydra only uses a launcher under `--multirun`, so a plain `just train` never reaches it.
The Ray launcher, the closest example, pickles the job and sends it to a Ray cluster. We want to run a command and
read its output.

[SkyPilot](https://docs.skypilot.ai/en/latest/reference/cli.html) comes closest. It knows RunPod,
[RunPod's network volumes](https://docs.skypilot.ai/en/latest/reference/volumes.html), log streaming and teardown. It
misses three of our rules. `--retry-until-up` retries forever, with no waiting time. Ctrl-C detaches from the logs and
leaves the job running. `--down` keeps the machine when provisioning or setup fails, so a broken setup keeps billing.
We would write a wrapper for these, on top of a framework with its own API server and task format.

[dstack](https://dstack.ai/docs/guides/protips/) also supports RunPod and its volumes, and `dstack apply` streams the
logs. It needs a dstack server, and we did not find what Ctrl-C does to a run.

[Flash](https://docs.runpod.io/flash/execution-model), in RunPod's own package, runs decorated functions on serverless
workers. Our run is a command that lives for 28 hours and prints as it goes, so we did not try it.

[RunPod's Python SDK](https://github.com/runpod/runpod-python) gives the three calls we need: `create_pod`, which takes
the GPU, data center and network volume, `get_pod`, which returns the pod's SSH address, and `terminate_pod`. On a copy
of this project, it adds 5 packages to the 246 in `uv.lock` and leaves torch at 2.7.1.

## We write it ourselves, on the SDK

No tool above does what we asked without a wrapper, and the job itself is small: three SDK calls, `rsync` and `ssh`. We
expect about 130 lines in `src/grooveback/remote.py`. Two choices remain.

### How the command asks for a pod

**A Hydra group, `remote=runpod`. This is what we chose.** `configs/remote/runpod.yaml` holds the machine, as
`configs/xp/` already holds optional settings. One syntax covers the run and the machine. Hydra also checks the config
on the Mac, so a missing `model.checkpoint` fails before we pay for a pod. Each entry point gets one `if`.

**A separate recipe, `just remote train lr=5e-5`.** A script forwards everything after the recipe's name, and the
training code does not change. It would work with any command. But the machine's settings need a second syntax, and a
config error only shows on the pod, after the GPU wait.

### How the job lives on the pod

We build both, because they suit different runs. Attached is the default, and `remote.background=true` asks for the
other.

**Attached to the terminal, with `ssh -tt`.** The local terminal is connected to the pod's, so logs print here with
nothing to poll. Ctrl-C reaches the job as a real SIGINT, and `train_lora` already handles that: it ends at once during
the steps, and waits while a checkpoint is written. The cost is that a lost connection ends the job. We lose at most
one epoch, 7.3 minutes in the last run, because the checkpoints are on the volume. But the run then stops until someone
resumes it. This suits `just evaluate` and short trainings.

**In the background on the pod, with a log file the Mac follows.** A lost connection only stops the printing. The job
goes on, and the Mac connects again and continues from the last line it printed. This suits a 28-hour training, which
would otherwise need the Mac connected for 28 hours. It takes about 30 more lines: start the job, follow the log,
reconnect, and send the Ctrl-C ourselves. That last part has a catch. We measured that a job started with `&` from a
script ignores SIGINT, so the Mac sends SIGTERM instead, which `train_lora` handles the same way.

## What we want to build

This builds on `just train` and `just evaluate`, and follows the PR that adds them. The machine is one small config. A
volume is pinned to its data center, so another volume is `remote.volume=... remote.data_center=...`.

```yaml
# configs/remote/runpod.yaml
gpu: NVIDIA L40S          # $1.09 per hour
volume: 18v73b8ggl        # grooveback-US-MO-1, 80 GB
data_center: US-MO-1
wait_minutes: 60          # how long to keep asking for the GPU
background: false         # true: the job goes on after a lost connection
image: runpod/pytorch:1.3.3-cu1281-torch271-ubuntu2404
```

Each entry point chooses where to run. The pod gets the same command without the `remote` overrides, so it takes the
`else` branch and runs exactly what runs on any machine:

```python
def main(cfg: DictConfig) -> None:
    if cfg.remote:    # just train remote=runpod lr=5e-5
        run_on_pod(cfg.remote, module="grooveback.cli.train_lora", results=cfg.save_dir)
    else:             # on the pod: python -m grooveback.cli.train_lora lr=5e-5
        train(cfg)
```

`run_on_pod` is the whole mechanism:

```python
def run_on_pod(remote, *, module, results):
    pod = wait_for_pod(remote)           # create_pod once a minute until wait_minutes run out
    try:
        host = wait_for_ssh(pod)         # get_pod until ssh.direct appears
        ssh(host, INSTALL)               # rsync, sox, ffmpeg, uv; data/ and artifacts/ link to the volume
        rsync(".", host)                 # this tree as it stands, uncommitted edits included
        job = command(module)            # uv run python -m ... with the overrides
        code = run_in_background(pod, job) if remote.background else ssh(host, job, terminal=True)
        rsync(host, results)             # the run's checkpoints or scores come home
    finally:
        runpod.terminate_pod(pod["id"])  # success, crash, Ctrl-C: the pod always goes
    sys.exit(code)
```

The background mode adds one function. The job writes its exit code to a file when it ends, so that file tells the Mac
the job is over, however many times the connection dropped:

```python
def run_in_background(pod, job):
    host = wait_for_ssh(pod)
    ssh(host, f"setsid nohup bash -c '{job}; echo $? > job.code' > job.log 2>&1 &")
    shown = 0                                    # log lines already printed here
    try:
        while not (code := ssh(host, "cat job.code")):
            shown += print_log(host, after=shown)    # returns when the job ends or the connection drops
            host = wait_for_ssh(pod)                 # raises if the pod is gone
    except KeyboardInterrupt:
        code = stop_job(host)                    # SIGTERM, then wait for job.code
    return int(code)
```

A Ctrl-C ends the job, and the results still come home. A second Ctrl-C skips that copy, and the results stay on the
volume.

The paths in the configs are relative, and on the pod `data/` and `artifacts/` are links to the volume. So
`artifacts/lora/run1` means the same run on both machines, and what a run writes survives the pod. Hydra resolves a
default `save_dir`, which holds the time, on the Mac, and the pod is given that value. The environment stays on the
pod's own disk. `uv sync --frozen` took 10 s on the last fresh L40S pod and over half an hour on an earlier one, so a
slow sync costs about $0.55, and we do not need a venv on the volume.

The right RunPod configuration means `RUNPOD_API_KEY` and `HF_TOKEN` in `.env`, and the Mac's SSH key registered with
RunPod. The Hugging Face token reaches the pod through `create_pod`'s `env`, never on a command line. A missing key is
refused before any pod exists.

### When the Mac cannot delete the pod

The `finally` above needs the Mac awake. If the Mac sleeps or dies, nothing runs it, and `--terminate-after` already
failed to stop a pod once. So when the job ends, for any reason, the pod starts a timer that deletes it 15 minutes
later. That gives the Mac time to bring the results home first, and `setsid` lets `ssh` return without waiting for it:

```bash
trap 'setsid nohup sh -c "sleep 900; runpodctl pod delete $RUNPOD_POD_ID" > /dev/null 2>&1 &' EXIT
```

[RunPod documents](https://docs.runpod.io/pods/manage-pods) `runpodctl` and a pod-scoped API key inside every pod, and
gives `runpodctl pod stop` as an example. We have not tried a delete with that key.

The two modes differ in what a lost Mac costs. Attached, the pod's SSH server gets a 5-minute timeout, so a vanished
Mac ends the job. With the timer, that limits a lost pod to about 20 minutes of billing, $0.36 on an L40S. On the Mac,
`ssh` runs under `caffeinate -i`, so idle sleep does not end an attached run. A closed lid on battery still does.

In the background, the job goes on without the Mac, which is the point. It runs to its end, the pod deletes itself 15
minutes later, and the results stay on the volume. So a Mac that never comes back costs the rest of the run: up to 28
hours, about $31, for a training. A Mac that comes back after the pod is gone says so, and the command ends with an
error.

## How we test it

Thirteen tests, on a fake `runpod` module and fake `ssh` and `rsync` commands on the `PATH` that record their arguments:

- `test_remote_run_trains_nothing_on_this_machine`
- `test_overrides_reach_the_pod_unchanged_without_the_remote_ones`
- `test_a_refused_gpu_is_asked_for_again_until_wait_minutes_run_out`
- `test_no_gpu_within_wait_minutes_ends_with_an_error_and_no_pod`
- `test_pod_is_deleted_after_a_job_that_succeeds_fails_or_is_interrupted`
- `test_pod_is_deleted_when_setup_fails`
- `test_results_come_home_before_the_pod_is_deleted`
- `test_config_errors_fail_before_a_pod_is_created`
- `test_the_jobs_exit_code_is_the_commands_exit_code`
- `test_background_log_has_no_gap_or_repeat_after_a_lost_connection`
- `test_ctrl_c_stops_a_background_job_before_the_pod_is_deleted`
- `test_a_background_jobs_exit_code_is_the_commands_exit_code`
- `test_background_run_ends_with_an_error_when_the_pod_is_gone`

Fakes cannot show what a real pod does, so we also run short checks on a real L40S, for about $2:

```bash
just train remote=runpod num_train_epochs=1 epoch_windows=16
```

This must print its logs here and end with `pod list` empty. A second run gets a Ctrl-C after its first steps: the job
must stop and the pod must go. A third has its `ssh` killed from outside: the pod must delete itself.

Two more runs add `remote.background=true`. In one we kill the `ssh` from outside: the job must go on and its log must
continue here. In the other we kill the local command itself: the job must finish, its results must be on the volume,
and the pod must delete itself. A Ctrl-C in the first of these must also stop the job, since SIGTERM takes a different
path from the attached run's SIGINT.

## What we do not know yet

We do not know if the pod-scoped key can delete its own pod. If it cannot, `runpodctl pod stop` still frees the GPU,
and only the disk bills until the Mac deletes the pod. A background run depends on this most, since no Mac may be there
to delete it.

SQLite on the network volume is untested, since every run so far kept `mlflow.db` on the pod's own disk. We have also
not measured how much of the volume's 80 GB is free, and one `just evaluate` writes about 6 GB of audio.

Everything on the Mac that a pod needs must already be on the volume. A checkpoint trained before this ADR, such as
`checkpoint-46765`, has to be copied there once.

This is for one job on one GPU. Sweeps and several pods at once are out of scope.
