# grooveback

🚧 **Work in progress — not ready for use.**

**A generative prior over minimal and electronic music, and restoration as its first application.**

As a DJ and listener, I keep running into the same problem: records I love that only exist, now, as 128 kbps MP3s
ripped from YouTube — themselves sourced from someone's imperfect vinyl transfer. On laptop speakers they're fine. On a
decent system they fall apart: smeared transients, no air in the top end, a generally flat and tired sound that makes
everything around them in the library sound better by contrast.

## What it does

The artifact is an unconditional generative prior over the music I actually listen to: old-school house, minimal, and
adjacent electronic genres. It lives in the latent space of a pretrained audio autoencoder, and it is adapted from
[Stable Audio 3](#ref-sa3) by fine-tuning on my own library rather than trained from scratch.

Restoration is the prior's first application. A degraded rip is treated as an observation of a clean track through some
unknown operator, and restoration means sampling from the posterior at inference time. The prior is learned once, from
clean audio only, and never sees a degraded example; the degradation enters only when the model runs.

That split is the point. Changing what damage gets restored means changing the operator, not retraining. Unconditional
generation falls out of the prior for free. Conditioning on inspiration samples — a vocal, a loop, a reference texture
— is a later and more open problem.

[ADR-0004](docs/decisions/0004-restoration-as-a-latent-inverse-problem.md) has the full reasoning.

## What it does not do

grooveback is not strict signal recovery. MP3 psychoacoustic masking, bandwidth cuts, and analog transfer noise discard
information irreversibly. No system can recover what is genuinely gone.

What comes out is a *plausible* reconstruction — structurally coherent with the input, perceptually convincing on good
playback, but not faithful to the original master. Where the observation says nothing, the prior invents. For the
intended use — listening on a home system to records you can't re-source — this is the right trade.

## Approach

Baselines first. Before any method work, the existing free tools run on real tracks from the library, and the two
components the design depends on get checked:

1. [Apollo](#ref-apollo) and [A2SB](#ref-a2sb) — codec artifacts and missing bandwidth respectively, and these rips
   have both.
2. The two chained, in both orders.
3. A DSP control — gain match, high shelf, mono below ~120 Hz. The floor everything else has to beat.
4. A [SAME](#ref-same) round-trip on this material, listening to hats, rides and reverb tails. Restoration never leaves
   the latent space, so whatever the autoencoder loses is lost.
5. A [Stable Audio 3](#ref-sa3) probe — unconditional sampling, and inpainting a masked region of a track.

Then work stops and the next decisions get made from results. See
[ADR-0005](docs/decisions/0005-baselines-and-prior-viability-on-real-library-material.md).

There is no supervised spectrogram baseline. Apollo already provides that ceiling as a pretrained checkpoint, and the
code would be thrown away in the move to a latent prior.

## Evaluation

Everything is level-matched to −14 LUFS before any comparison. Then it's the ear, on monitoring hardware, plus
spectrograms. Objective metrics come in later to catch regressions; the listening is the judge, because this is
ultimately a playback problem.

## A note on genre focus

Training and evaluation are deliberately biased toward old-school house, minimal, and adjacent electronic music. A
narrow distribution is easier to model than a broad one, and that is much of what makes this tractable for one person.
It is also what fine-tuning does well — the base model supplies the general structure a personal library can't teach.

## Research context

grooveback sits between diffusion-based inverse problems, music restoration, and learned priors over audio.

**Inverse problems with diffusion priors** — the line this descends from. [DPS](#ref-dps) established posterior
sampling with a diffusion prior and a known operator. [SILO](#ref-silo) moved the problem into an autoencoder's latent
space. [DAPS](#ref-daps) fixes DPS's tendency to oversmooth, which matters here since smearing is the thing being
fixed. [LOUDAR](#ref-loudar) is the closest published match to this project's method.

**Music restoration** — [CQT-Diff](#ref-cqtdiff) brought diffusion-based audio inverse problems to music.
[BABE](#ref-babe) and [BABE-2](#ref-babe2) do blind bandwidth extension and generative equalisation on historical
recordings. On the supervised side, [Apollo](#ref-apollo) targets codec artifacts and [A2SB](#ref-a2sb) targets
bandwidth and inpainting at 44.1 kHz. [SonicMaster](#ref-sonicmaster) was tested and rejected — it duplicates and
stacks kicks on electronic material.

**Autoencoders and priors** — the prior trains inside a specific latent space and can't be transplanted, so this choice
is upstream of everything. [SAME](#ref-same) is stereo 44.1 kHz with open weights, and [Stable Audio 3](#ref-sa3) ships
an open-weights music prior in that same space — which is why the two are chosen together rather than separately.
Alternatives held as fallbacks: [Music2Latent](#ref-m2l) (mono, which is disqualifying for this material),
[ε-VAE](#ref-epsvae), [CoDiCodec](#ref-codicodec).

## References

<a id="ref-dps"></a>
[DPS] Chung et al. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

<a id="ref-silo"></a>
[SILO] Raphaeli, Man, Elad. *SILO: Solving Inverse Problems with Latent Operators.* ICCV 2025. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)

<a id="ref-daps"></a>
[DAPS] Zhang et al. *Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing.* CVPR 2025. [arXiv:2407.01521](https://arxiv.org/abs/2407.01521)

<a id="ref-loudar"></a>
[LOUDAR] Švento et al. *Music Restoration via Latent Operator Optimization and Diffusion Model Priors.* ISMIR 2026. [arXiv:2608.01972](https://arxiv.org/abs/2608.01972) · [demo](https://michalsvento.github.io/loudar/)

<a id="ref-cqtdiff"></a>
[CQT-Diff] Moliner, Lehtinen, Välimäki. *Solving Audio Inverse Problems with a Diffusion Model.* [arXiv:2210.15228](https://arxiv.org/abs/2210.15228)

<a id="ref-babe"></a>
[BABE] *Blind Audio Bandwidth Extension: A Diffusion-Based Zero-Shot Approach.* [arXiv:2306.01433](https://arxiv.org/abs/2306.01433)

<a id="ref-babe2"></a>
[BABE-2] Moliner et al. *A Diffusion-Based Generative Equalizer for Music Restoration.* DAFx-24. [arXiv:2403.18636](https://arxiv.org/abs/2403.18636)

<a id="ref-apollo"></a>
[Apollo] Li & Luo. *Apollo: Band-sequence Modeling for High-Quality Audio Restoration.* [arXiv:2409.08514](https://arxiv.org/abs/2409.08514) · [code](https://github.com/JusperLee/Apollo)

<a id="ref-a2sb"></a>
[A2SB] NVIDIA. *Audio-to-Audio Schrödinger Bridges.* [arXiv:2501.11311](https://arxiv.org/abs/2501.11311) · [code](https://github.com/NVIDIA/diffusion-audio-restoration)

<a id="ref-sonicmaster"></a>
[SonicMaster] *Towards Controllable All-in-One Music Restoration and Mastering.* [arXiv:2508.03448](https://arxiv.org/abs/2508.03448)

<a id="ref-same"></a>
[SAME] Parker, Evans et al. *SAME: A Semantically-Aligned Music Autoencoder.* [arXiv:2605.18613](https://arxiv.org/abs/2605.18613) · [weights](https://huggingface.co/stabilityai/SAME-L)

<a id="ref-sa3"></a>
[SA3] Stability AI. *Stable Audio 3.* [arXiv:2605.17991](https://arxiv.org/abs/2605.17991) · [code](https://github.com/Stability-AI/stable-audio-3) · [weights](https://huggingface.co/stabilityai/stable-audio-3-small-music)

<a id="ref-m2l"></a>
[M2L] Pasini, Lattner et al. *Music2Latent: Consistency Autoencoders for Latent Audio Compression.* [arXiv:2408.06500](https://arxiv.org/abs/2408.06500)

<a id="ref-epsvae"></a>
[ε-VAE] *Back to Ear: Perceptually Driven High Fidelity Music Reconstruction.* [arXiv:2509.14912](https://arxiv.org/abs/2509.14912)

<a id="ref-codicodec"></a>
[CoDiCodec] *Unifying Continuous and Discrete Compressed Representations of Audio.* [arXiv:2509.09836](https://arxiv.org/abs/2509.09836)

---

Code here is MIT. Weights adapted from Stable Audio 3 inherit the
[Stability AI Community License](https://stability.ai/license) — fine for personal and research use, which is this
project's scope.

Decisions are recorded as ADRs in [`docs/decisions/`](docs/decisions/).
