# 5. Establish baselines and prior viability on real library material

Date: 2026-08-15

## Status

Accepted

Supersedes [3. Run Apollo on target distribution and analyse results](0003-run-apollo-on-target-distribution-and-analyse-results.md)

## Context

ADR-0003 decided to run Apollo on grooveback-distribution tracks before writing any model, so that architectural
decisions would rest on evidence rather than assumption. That instinct was right and is kept. What has changed is that
ADR-0004 replaced the supervised design with a prior-first one, which widens the set of things that must be measured
before anything else is decided:

- Apollo is no longer a reference point on the way to training a similar model. It is a **standing baseline** that
  every future method must beat.
- A second free baseline exists that ADR-0003 did not know about: **A2SB**.
- The autoencoder and the pretrained prior are now load-bearing components, and both can be smoke-tested cheaply before
  any GPU budget is committed.

There is also a discipline problem worth naming. Every remaining decision in this project — solver choice, degradation
modelling, evaluation protocol, whether to fine-tune at all — is easier and better-informed after hearing what the
existing free tools do to actual tracks from the library. Writing those ADRs first would be speculating in public.

## Decision

Before any further ADR is written, run the following on real material from the library, in this order. This is a
deliberate stopping point: **ADR-0006 onward is not drafted until these results exist.**

### 1. Apollo and A2SB, as two first-class baselines

Not one plus a follow-on. They address complementary damage and the target material has both:

- **Apollo** (band-split / band-sequence modelling, pretrained on MUSDB18-HQ and MoisesDB at various MP3 bitrates)
  targets **codec artifacts**. Its training distribution is professional multitrack material, not vinyl rips
  re-encoded by YouTube on a narrow genre, so generalisation across that gap is the open question — as ADR-0003
  correctly identified.
- **A2SB** (NVIDIA, Schrödinger bridge over a magnitude-phase factorized representation, 44.1 kHz, trained on 2.3k
  hours of permissively-licensed music) targets **missing bandwidth** and also does inpainting, end-to-end with no
  vocoder.

A practical asymmetry to record while running them: Apollo needs chunking to fit the 16 GB local machine, while A2SB
claims hour-long inputs natively. That makes A2SB the cheaper one to run over whole tracks, and it means the chunking
and overlap-add correctness gate is Apollo-specific.

### 2. The two chained, in both orders

Apollo → A2SB and A2SB → Apollo. Cheap once both wrappers exist, and the ordering plausibly matters: removing codec
artifacts before extending bandwidth is a different proposition from extending bandwidth over artifacts that are still
present.

### 3. The DSP control condition

Gain match, high shelf, mono below roughly 120 Hz. Sequenced after the learned baselines rather than before, but it
serves as the **floor** — the first question asked of any output, now and for the rest of the project, is whether it
beats an EQ.

The point is to separate how much of the perceived damage is level, tonal balance and low-end phase from how much is
genuine codec loss. If a simple linear treatment closes most of the perceptual gap, that is decisive information about
what the generative work is actually for, and it is far better learned now than in three months.

### 4. SAME round-trip sanity check

Encode and decode library tracks through SAME and listen to the result against the input.

This is **not** an autoencoder bake-off. SAME is fixed by the Stable Audio 3 pairing, so there is no selection to make;
the question is only whether its encodings of *this* material are sensible before any LoRA work is built on top of
them. The discriminating cases are the ones this material is full of and that compression handles worst: **hi-hats,
rides, and reverb tails**.

The round-trip bounds everything downstream. Restoration happens entirely inside the latent space, so whatever the
autoencoder loses is lost regardless of how good the prior or the solver is.

### 5. Stable Audio 3 viability probe

Two checks on the **base** checkpoint, before any fine-tuning:

- **Unconditional sampling** — null text conditioning at CFG=1. Does it produce structurally coherent music at all
  under the conditions the prior will actually be used in?
- **Inpainting** — mask a two-second region of a library track and have the model fill it. This is the cheapest
  possible test of prior quality: the operator is exactly known, there is no solver subtlety, and the answer is
  audible immediately. If the prior cannot do this in-style, no amount of solver work rescues it.

This probe also resolves two facts that published sources disagree on and which should not be asserted in
documentation until checked: the model's **parameter count** (variously reported as 433M, 459M, and 0.6B) and its
**sample rate** (the paper and repository say 44.1 kHz; one Hugging Face model-card field reads 16 kHz).

### How output is judged at this stage

- **Level-matched to −14 LUFS before any comparison**, without exception. Loudness differences dominate every informal
  audio comparison ever made and will produce confident wrong conclusions otherwise.
- **By ear on monitoring hardware, and by spectrogram.** That is the judge at this stage.
- Objective metrics are not the deciding instrument here and the full evaluation protocol is deliberately deferred.

### What is deliberately not done yet

Recorded so it is clear these are scheduled, not overlooked:

- **No alignment of the real clean/rip pairs.** The ~20 held pairs carry arbitrary time offsets, sample-rate drift, and
  are often different masters, so paired metrics are not available until alignment is built. That is second-wave work;
  ear and spectrogram come first.
- **No paired-metric evaluation**, following from the above.
- **No degradation chain**, no solvers, no fine-tuning, no latent precompute of the full library.

### Recorded negative result: SonicMaster

SonicMaster (all-in-one text-controllable music restoration and mastering) was tested via its Hugging Face demo on
electronic material. It **duplicates and stacks kick drums** and introduces assorted structural artifacts, including
under an explicit negative prompt instructing it not to and to enhance only what was already present.

It is therefore not adopted as a baseline. This is recorded rather than discarded so the same evaluation is not
repeated in three months on the assumption that an untested all-in-one model might help.

## Consequences

- Every subsequent method has a number and a listening reference to beat, established before it is designed.
- The two components the prior-first design depends on — the autoencoder and the pretrained prior — are validated
  before any GPU budget is spent on them.
- Work stops after this. The next ADR round is written from results, which delays the method work by roughly a week and
  is the point.
- The infrastructure built here (manifests, level matching, evaluation, listening packs, baseline wrappers) is the
  infrastructure the rest of the project uses. Building it around frozen pretrained models rather than around a model
  that must also be trained is the reason ADR-0004 cut the supervised baseline.
- Running someone else's checkpoints on a 16 GB Apple Silicon machine is its own source of friction — chunking,
  dependency conflicts, and MPS gaps. That cost is accepted; it is smaller than training anything.

## Revisit triggers

These are the conditions that open the next ADR round, and what each implies:

- **A baseline is already good enough on this material.** Then the project's value proposition narrows sharply and the
  scope should be reconsidered before more is built.
- **The DSP control closes most of the perceptual gap.** Same conclusion, more bluntly: the problem was tonal balance
  and level, not codec loss.
- **SAME's round-trip is audibly damaging on hats, rides or reverb tails.** This bounds the whole approach and forces
  reconsideration of the latent space — which, because Stable Audio 3 is tied to SAME, means reconsidering the prior
  source too. This is the most consequential possible outcome of this ADR.
- **The Stable Audio 3 probe fails** — no coherent unconditional output, or inpainting that is structurally or
  stylistically wrong. This is the from-scratch trigger recorded in ADR-0004.
- **The probe succeeds well** — in which case fine-tuning may prove unnecessary for a first restoration attempt, and
  solver work can begin against the base prior directly.
- **Apollo or A2SB proves impractical to run locally**, in which case the baseline moves to rented GPU and is costed
  accordingly.
