# 9. SDEdit as the first solver

Date: 2026-09-07

## Status

Accepted

## Context

The baseline phase is closed (ADR-0005 → 0006/0007/0008). The next phase is
using the prior at inference time: given a degraded file, steer the
generative model toward a clean track that could have produced it (ADR-0004).
The full posterior-sampling methods (the DPS/DAPS/LOUDAR line) need an
operator model and solver machinery; before building any of that, the
restoration chain itself — degraded audio → SAME latents → noise → denoise
with the prior → decode → listen — should exist and be exercised end to end.

## Decision

SDEdit (Meng et al. 2022) is the first solver: encode the input, mix in
noise at a chosen level, and denoise with the base Stable Audio 3 checkpoint
from there. Nothing is trained — no surrogate, no fine-tuning; the noise
level is the whole method. `stable-audio-3` implements it natively
(`init_audio` + `init_noise_level`, the schedule's starting sigma), so
`grooveback.solvers.sdedit` is a thin wrapper and the work is the harness.

- **One Hydra entry point**, `grooveback.cli.run`: a `mode` argument names
  the run type (only `inference` exists), a `model` config group selects the
  method and carries its parameters — a future solver is `model=dps` plus a
  yaml. Input is a file or a folder; each file is restored whole and written
  to the output dir under its own name plus `_{method}_{noise_level}`. One
  noise level per run; sweeps are multiruns sharing an output dir.
- **Whole files, up to 6 minutes.** The `seconds_total` conditioner is
  calibrated to 384 s (ADR-0008); the solver refuses longer input rather
  than silently truncating. For the small variants this runs past their
  120 s trained window — a deliberate experiment; `medium` covers 6 minutes
  inside its trained length.
- **Empty prompt by default.** What the init audio alone contributes comes
  first; text conditioning is an override to explore later.
- **No scoring in the chain.** Degraded material (64/128/192 kbps MP3s of
  the codec asset and a 6-minute aerofunk cut) is prepared once with ffmpeg;
  renders are judged by level-matched listening (audio.md). The one
  mechanical gate: over a noise-level sweep, MSE(render, input) must rise
  monotonically with the noise level — more starting noise regenerates more
  — which catches a backwards noise mapping or an ignored init.

## Consequences

- The whole inference path of ADR-0004 exists and is parametrized; every
  later solver reuses the harness and replaces only the model group.
- Expected failure mode, accepted going in: one global noise level erases
  in-band detail it should keep while inventing what is missing — SDEdit
  has no mechanism to stay consistent with the observation. Where the sweep
  lands between "still degraded" and "no longer the same track" is exactly
  the finding to record.
- At noise level → 0 the chain reduces to the bare autoencoder round-trip,
  the benchmark's `same-s` row (ADR-0007) — a known answer that verifies
  the wiring.

## Revisit triggers

- Listening results land → record findings, and pick the posterior-sampling
  solver work that follows.
- A setting worth keeping emerges → promote it into the ADR-0007 benchmark
  as a named method.
