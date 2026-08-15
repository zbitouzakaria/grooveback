# grooveback

🚧 **Work in progress — not ready for use.** No demo, no serving, no packaging for other users. The method comes first.

**A generative prior over minimal and electronic music, and restoration as its first application.**

## The problem

As a DJ and listener I keep running into the same thing: records I love that only exist now as low-bitrate MP3s ripped
from YouTube, themselves sourced from someone's imperfect vinyl transfer. On laptop speakers they are fine. On a decent
system they fall apart — smeared transients, no air in the top end, a flat and tired sound that makes everything around
them in the library sound better by contrast.

## What this is

The artifact is an **unconditional generative prior** over the music I actually listen to: old-school house, minimal,
and adjacent electronic genres. It lives in the latent space of a pretrained audio autoencoder rather than on waveforms
or spectrograms.

The prior is **adapted from [Stable Audio 3](#ref-sa3)** — an open-weights music model released in May 2026 — by
fine-tuning on my own library, rather than trained from scratch. A ~2,000-track library is roughly 215 hours, which is
thin for training a generative model from noise but ample for specialising one that already knows what music is.
[ADR-0004](docs/decisions/0004-restoration-as-a-latent-inverse-problem.md) argues that case in full.

Restoration is the prior's first application, not the object of the project. A degraded rip is treated as an
observation of a clean track through an unknown operator:

```
y = A(x) + n
```

and restoration means sampling from the posterior `p(x | y) ∝ p(y | x) · p(x)` at inference time. The prior `p(x)` is
learned once, from clean audio only, and never sees a degraded example. The likelihood `p(y | x)` is not learned at all
— it is computed from an assumed or estimated `A` when the model runs.

That split is the whole design. Changing what damage gets restored means changing `A`, not retraining. Unconditional
generation falls out of the prior for free. Conditioning on inspiration samples — a vocal, a loop, a reference texture
— is a later and separately open problem, likely inpainting-shaped at first.

See [ADR-0004](docs/decisions/0004-restoration-as-a-latent-inverse-problem.md) for the full reasoning.

## What this is not

**This is not signal recovery.** MP3 psychoacoustic masking discards information by design; bandwidth truncation and
analog transfer noise destroy more. None of it is recoverable, and nothing here recovers it.

What comes out is a sample from the posterior: **structurally coherent with the input** and **perceptually convincing
on good playback**, but not faithful to the original master. Where the observation carries no information, the prior
fills in — which is to say the system invents plausible detail. For listening at home to records that cannot be
re-sourced, that is the right trade. For anything needing provenance or fidelity guarantees, it is not, and no such
claim is made.

This has a methodological consequence that shapes the evaluation: waveform-level and spectral-magnitude metrics
systematically penalise this class of method, because a plausible reconstruction differs from the reference in exactly
the places those metrics measure.

## Status

Baselines first. Before any method work, the free existing tools run on real tracks from the library, and the two
components the design depends on get smoke-tested. In order:

1. **[Apollo](#ref-apollo)** and **[A2SB](#ref-a2sb)** as two first-class baselines — codec artifacts and missing
   bandwidth respectively, and these rips have both.
2. The two **chained**, in both orders.
3. A **DSP control** — gain match, high shelf, mono below ~120 Hz. The floor every method must beat, and the test of
   how much of the perceived damage is level and tonal balance rather than codec loss.
4. A **[SAME](#ref-same) round-trip** on this material, checking hi-hats, rides and reverb tails specifically.
   Restoration never leaves the latent space, so whatever the autoencoder loses is lost regardless.
5. A **[Stable Audio 3](#ref-sa3) viability probe** — unconditional sampling, and inpainting a masked region of a
   library track.

Then work stops and the next decisions are made from results. See
[ADR-0005](docs/decisions/0005-baselines-and-prior-viability-on-real-library-material.md).

**There is no supervised spectrogram U-Net baseline.** The ceiling one would establish is already in the literature and
Apollo provides it as a pretrained checkpoint; almost none of that code survives the transition to a latent generative
prior; and the infrastructure that does survive is cheaper to build around a frozen model than around one that must
also be trained.

## Evaluation

Everything is **level-matched to −14 LUFS before any comparison**, without exception. Loudness differences dominate
informal audio comparisons and produce confident wrong conclusions.

Objective metrics catch regressions. **Blinded listening on monitoring hardware is the judge.** Embedding-based metrics
are preferred over waveform-level comparison, since the autoencoder introduces phase differences that unfairly penalise
generative methods; distributional metrics like FAD are unreliable at this project's sample sizes and want the
[per-song treatment](#ref-fad) and a music-appropriate embedding rather than the default one. Per-sample metrics such
as [MuQ-Eval](#ref-muqeval) are the right shape for an evaluation set of a few dozen tracks, where distributional
distances are close to noise.

## A note on genre focus

Training and evaluation are deliberately biased toward old-school house, minimal, and adjacent electronic music. A
narrow distribution is easier to model than a broad one, and that is a large part of what makes this tractable for one
person. It is also what fine-tuning a pretrained prior does well: the base checkpoint supplies the generic structure a
personal library cannot teach, and adaptation supplies the specialisation.

## Research context

grooveback sits between diffusion-based inverse problems, music restoration, and learned priors over audio. The
guiding assumption is that **audio diffusion imports from vision diffusion with a lag of a year or two**, so the vision
literature is read as a preview of what has not crossed over yet.

### Inverse problems with diffusion priors

The line this project descends from. [DPS](#ref-dps) established posterior sampling with a diffusion prior and a known
operator; blind variants added joint operator estimation; latent-space variants (PSLD, ReSample, LatentDAPS,
[SILO](#ref-silo)) moved the problem into an autoencoder's latent space so the prior could be a large pretrained model.
[LOUDAR](#ref-loudar) is the closest published match to this project's method: SILO-style latent operator optimisation
with a blind operator, over an unconditional music prior.

[DAPS](#ref-daps) and [DAPS++](#ref-dapspp) are the part that has **not** crossed over to audio. DPS's known failure is
oversmoothing, which is the exact failure this project cannot tolerate; DAPS decouples the sampling trajectory to fix
it, and DAPS++ reports large reductions in the number of function evaluations — directly relevant given LOUDAR's
inference cost.

### Audio priors and restoration

[CQT-Diff](#ref-cqtdiff) brought diffusion-based audio inverse problems to music with a Constant-Q representation.
[BABE](#ref-babe) and [BABE-2](#ref-babe2) do blind bandwidth extension and generative equalisation for historical
recordings — the closest published work to restoring degraded historical music. UnDiff and BUDDy cover unsupervised
operator estimation in audio. On the supervised and bridge-model side, [Apollo](#ref-apollo) targets codec artifacts
and [A2SB](#ref-a2sb) targets missing bandwidth and inpainting at 44.1 kHz.

[SonicMaster](#ref-sonicmaster) was tested and rejected: it duplicates and stacks kick drums on electronic material,
including under an explicit negative prompt.

### Autoencoders and latent spaces

The prior is trained inside a specific latent space and cannot be transplanted, so this choice is upstream of
everything. [SAME](#ref-same) is stereo 44.1 kHz with open weights, and notably includes a small unconditional
diffusion transformer trained jointly on its own latents — so its suitability for diffusion is a training objective
rather than a hope. [Stable Audio 3](#ref-sa3) ships an open-weights music prior in exactly that space, which is why
the two decisions are made together rather than independently.

Alternatives, held as fallbacks: [Music2Latent](#ref-m2l) and [Music2Latent2](#ref-m2l2) (mono, which is disqualifying
for this material), [ε-VAE](#ref-epsvae), [CoDiCodec](#ref-codicodec), the Stable Audio Open VAE, DAC and EnCodec.

### Vision methods not yet crossed over

Read as a preview rather than as background:

- [EQ-VAE](#ref-eqvae) and [latent spectral analysis of video VAEs](#ref-videovae) give a *measurable* account of why
  some latent spaces train well — channel eigenspectrum and low-frequency bias — turning "diffusability" from an
  assertion into a diagnostic.
- [REPA](#ref-repa) and [representation autoencoders](#ref-rae) on training diffusion transformers cheaply.
- [StableSR](#ref-stablesr), [DiffBIR](#ref-diffbir), [SUPIR](#ref-supir) and [OSEDiff](#ref-osediff) define the
  conditional fine-tuning branch, should test-time posterior sampling prove impractical.

[Real-ESRGAN](#ref-realesrgan)'s thesis — that the degradation model is the research artifact — is **not** adopted
here. It holds when training supervised on synthetic pairs, where the simulator *is* the training distribution. In a
prior-first regime the prior trains on clean audio only and the operator is assumed or estimated at test time, so the
degradation chain becomes an evaluation instrument instead.

## References

**Inverse problems with diffusion priors**

<a id="ref-dps"></a>
[DPS] Chung et al. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

<a id="ref-silo"></a>
[SILO] Raphaeli, Man, Elad. *SILO: Solving Inverse Problems with Latent Operators.* ICCV 2025. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)

<a id="ref-daps"></a>
[DAPS] Zhang et al. *Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing.* CVPR 2025 Oral. [arXiv:2407.01521](https://arxiv.org/abs/2407.01521) · [code](https://github.com/zhangbingliang2019/DAPS)

<a id="ref-dapspp"></a>
[DAPS++] *DAPS++: Rethinking Diffusion Inverse Problems with Decoupled Posterior Annealing.* [arXiv:2511.17038](https://arxiv.org/abs/2511.17038)

<a id="ref-loudar"></a>
[LOUDAR] Švento et al. *Music Restoration via Latent Operator Optimization and Diffusion Model Priors.* ISMIR 2026. [arXiv:2608.01972](https://arxiv.org/abs/2608.01972) · [demo](https://michalsvento.github.io/loudar/) · [code](https://github.com/michalsvento/loudar)

**Audio priors and restoration**

<a id="ref-cqtdiff"></a>
[CQT-Diff] Moliner, Lehtinen, Välimäki. *Solving Audio Inverse Problems with a Diffusion Model.* [arXiv:2210.15228](https://arxiv.org/abs/2210.15228)

<a id="ref-babe"></a>
[BABE] *Blind Audio Bandwidth Extension: A Diffusion-Based Zero-Shot Approach.* [arXiv:2306.01433](https://arxiv.org/abs/2306.01433)

<a id="ref-babe2"></a>
[BABE-2] Moliner et al. *A Diffusion-Based Generative Equalizer for Music Restoration.* DAFx-24. [arXiv:2403.18636](https://arxiv.org/abs/2403.18636) · [code](https://github.com/eloimoliner/BABE2-music-restoration)

<a id="ref-apollo"></a>
[Apollo] Li & Luo. *Apollo: Band-sequence Modeling for High-Quality Audio Restoration.* ICASSP 2025. [arXiv:2409.08514](https://arxiv.org/abs/2409.08514) · [code](https://github.com/JusperLee/Apollo)

<a id="ref-a2sb"></a>
[A2SB] NVIDIA. *A2SB: Audio-to-Audio Schrödinger Bridges.* [arXiv:2501.11311](https://arxiv.org/abs/2501.11311) · [code](https://github.com/NVIDIA/diffusion-audio-restoration)

<a id="ref-sonicmaster"></a>
[SonicMaster] *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering.* [arXiv:2508.03448](https://arxiv.org/abs/2508.03448)

**Autoencoders and latent spaces**

<a id="ref-same"></a>
[SAME] Parker, Evans et al. *SAME: A Semantically-Aligned Music Autoencoder.* Stability AI. [arXiv:2605.18613](https://arxiv.org/abs/2605.18613) · [weights](https://huggingface.co/stabilityai/SAME-L)

<a id="ref-sa3"></a>
[SA3] Stability AI. *Stable Audio 3.* [arXiv:2605.17991](https://arxiv.org/abs/2605.17991) · [code](https://github.com/Stability-AI/stable-audio-3) · [small-music weights](https://huggingface.co/stabilityai/stable-audio-3-small-music)

<a id="ref-m2l"></a>
[M2L] Pasini, Lattner et al. *Music2Latent: Consistency Autoencoders for Latent Audio Compression.* [arXiv:2408.06500](https://arxiv.org/abs/2408.06500)

<a id="ref-m2l2"></a>
[M2L2] *Music2Latent2: Audio Compression with Summary Embeddings and Autoregressive Decoding.* [arXiv:2501.17578](https://arxiv.org/abs/2501.17578)

<a id="ref-epsvae"></a>
[ε-VAE] *Back to Ear: Perceptually Driven High Fidelity Music Reconstruction.* [arXiv:2509.14912](https://arxiv.org/abs/2509.14912)

<a id="ref-codicodec"></a>
[CoDiCodec] *CoDiCodec: Unifying Continuous and Discrete Compressed Representations of Audio.* [arXiv:2509.09836](https://arxiv.org/abs/2509.09836)

**Vision methods not yet crossed over**

<a id="ref-eqvae"></a>
[EQ-VAE] *EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Image Modeling.* [arXiv:2502.09509](https://arxiv.org/abs/2502.09509)

<a id="ref-videovae"></a>
[VideoVAE] *Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability.* [arXiv:2512.05394](https://arxiv.org/abs/2512.05394)

<a id="ref-repa"></a>
[REPA] Yu et al. *Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think.* ICLR 2025 Oral. [arXiv:2410.06940](https://arxiv.org/abs/2410.06940) · [code](https://github.com/sihyun-yu/REPA)

<a id="ref-rae"></a>
[RAE] *Improved Baselines with Representation Autoencoders.* [arXiv:2605.18324](https://arxiv.org/abs/2605.18324)

<a id="ref-realesrgan"></a>
[Real-ESRGAN] Wang et al. *Real-ESRGAN: Training Real-World Blind Super-Resolution With Pure Synthetic Data.* ICCVW 2021. [code](https://github.com/xinntao/Real-ESRGAN)

<a id="ref-stablesr"></a>
[StableSR] *Exploiting Diffusion Prior for Real-World Image Super-Resolution.* [arXiv:2305.07015](https://arxiv.org/abs/2305.07015) · [code](https://github.com/IceClear/StableSR)

<a id="ref-diffbir"></a>
[DiffBIR] *DiffBIR: Blind Image Restoration with Generative Diffusion Prior.* [code](https://github.com/XPixelGroup/DiffBIR)

<a id="ref-supir"></a>
[SUPIR] *Scaling Up to Excellence: Photo-Realistic Image Restoration In the Wild.* [code](https://github.com/Fanghua-Yu/SUPIR)

<a id="ref-osediff"></a>
[OSEDiff] *One-Step Effective Diffusion Network for Real-World Image Super-Resolution.* [code](https://github.com/cswry/OSEDiff)

**Evaluation**

<a id="ref-fad"></a>
[FAD] Gui et al. *Adapting Fréchet Audio Distance for Generative Music Evaluation.* ICASSP 2024. [arXiv:2311.01616](https://arxiv.org/abs/2311.01616) · [fadtk](https://github.com/microsoft/fadtk)

<a id="ref-muqeval"></a>
[MuQ-Eval] *MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation.* [arXiv:2603.22677](https://arxiv.org/abs/2603.22677)

---

## Licensing

The code here is MIT. The prior is adapted from Stable Audio 3, so any resulting weights inherit the
[Stability AI Community License](https://stability.ai/license) — free for personal and research use, which is this
project's stated and indefinite scope, but not an unencumbered artifact. Training a prior from scratch is the route to
unencumbered weights, and [ADR-0004](docs/decisions/0004-restoration-as-a-latent-inverse-problem.md) records why that
is not the starting point.

Decisions are recorded as ADRs in [`docs/decisions/`](docs/decisions/).
