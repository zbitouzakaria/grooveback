# grooveback

**Restoring records that survive only as MP3/AAC/Opus rips.**

As a DJ and listener, I keep running into the same problem: records I love that only exist, now, as rips from
YouTube. Someone played their vinyl through an imperfect rig, compressed the transfer to 128 or 192 kbps MP3, and
uploaded it; YouTube then re-encoded that to AAC or Opus. The result is double lossy compression, stacked on whatever
the turntable and capture already did. On laptop speakers the rips are fine. On a decent system they fall apart:
smeared transients, no air in the top end, a generally flat and tired sound that makes everything around them in the
library sound better by contrast.

## Introduction

This project aims to restore music recordings that survive only as lossy-codec rips. The target material is
electronic music — old-school house, minimal, and adjacent genres.

Restoration under these conditions is not exact signal recovery. Psychoacoustic masking, bandwidth truncation, and
analog transfer noise discard information irreversibly, so a generative restorer must invent what is gone. The output is therefore a *plausible* reconstruction — structurally coherent with the input and
perceptually convincing, but not guaranteed faithful to the original master.

We evaluate the existing literature on this material first and keep what survives. A genre-specific generative prior
adapted from [Stable Audio 3](#ref-sa3), with the degradation handled at inference, is the path we will explore once
the existing methods are exhausted
([restoration with a generative prior](docs/decisions/0004-restoration-as-a-latent-inverse-problem.md)).

## Related work

Lossy codecs — MP3, AAC, Opus — reach their bitrate by allocating bits according to psychoacoustic masking: content
the ear is unlikely to notice is quantized coarsely or dropped. Above a cutoff frequency nothing is kept; 128 kbps
MP3 drops everything above roughly 16 kHz. Below the cutoff the signal survives but degrades — transients smear, and
reverb tails and stereo detail thin out. A spectrogram shows the cutoff plainly and the in-band degradation barely.
Rips from YouTube typically carry this damage twice, because uploads are commonly 128–192 kbps MP3 and YouTube serves
standard streams at ~128 kbps AAC-LC or Opus, reserving higher bitrates such as 256 kbps AAC for Premium tiers
([YouTube audio formats](#ref-ytformats)).

The methods split along two lines: which damage they address, and which representation they work in. Bandwidth
extension reconstructs only the band above the cutoff, and most restoration models target acoustic or tonal faults
such as reverb, clipping, or tonal balance rather than MP3/AAC/Opus damage. As for representation, models operate on
the waveform directly, on spectrograms, or inside the compressed latent space of a neural autoencoder.

### Autoencoders — the VAE line

Generation in a latent space inherits the autoencoder's limits: no method operating on latents can sound better than
what the decoder reconstructs. The decoder also invents plausible detail where its input carries none, which makes the
bare round-trip `decode(encode(x))` a restoration baseline in itself.

- **[SAME](#ref-same)** is a stereo 44.1 kHz transformer autoencoder that reaches 4096× temporal compression through
  semantic regularisation; both variants, the large L and a CPU-scale S, are open weights.
- **[εar-VAE](#ref-earvae)** retrains the music VAE around perception, with K-weighted losses, explicit phase terms,
  and mid/side supervision — aimed exactly at the high-frequency harmonics and stereo image that MP3, AAC and Opus
  damage.
- **[εar-VAE2](#ref-earvae2)** moves the autoencoder onto the complex STFT so that every band gets targeted magnitude
  and phase correction. Code and weights are released, though the open weights are retrained on public data rather
  than the paper's corpus.
- **[Elastic Time](#ref-elastic)** retrofits a fixed-frame-rate autoencoder with a dynamic one, spending latent frames
  where the signal is dense. It is an efficiency mechanism, not a restorer.

### Bandwidth extension and super-resolution

These methods reconstruct the band a lowpass removed. All five train on lowpass damage alone; none trains on what
MP3, AAC or Opus do inside the kept band. They also treat the kept band differently by construction: AudioSR pastes
the input's band below the cutoff back into its output, A2SB trains on brick-wall cutoffs, and the latent and token
methods re-synthesize the whole signal without being trained for in-band repair.

- **[AudioSR](#ref-audiosr)** performs diffusion super-resolution for any audio type, taking any input bandwidth
  between 2 and 16 kHz up to a 48 kHz output, and serves as a plug-and-play enhancement stage for generative models.
- **[A2SB](#ref-a2sb)** is NVIDIA's Schrödinger bridge for 44.1 kHz music. It performs bandwidth extension and
  inpainting without a vocoder and processes hour-long inputs.
- **[AudioLBM](#ref-audiolbm)** moves the bridge into a latent space and cascades it, reaching the first
  any-to-192 kHz upsampling. Only a demo page exists; nothing is released.
- **[HP-codecX](#ref-hpcodec)** frames bandwidth extension as next-token prediction: a transformer LM generates only
  the missing band's tokens over a codec disentangled by harmonic/percussive decomposition, extending content above
  8 kHz from a 16 kHz-rate input to a 48 kHz-rate output. Code and both checkpoints are released.
- **[Latent upsample/upmix](#ref-latentup)** runs bandwidth extension and mono-to-stereo upmixing entirely inside a
  frozen autoencoder's latent space, at up to 100× less compute.

### MP3/AAC/Opus artifacts and all-in-one restoration

The supervised line trains on synthetic damaged/clean pairs and maps one to the other in a single pass.

- **[Apollo](#ref-apollo)** is a band-split waveform GAN trained on MP3-compressed music at multiple bitrates. It
  restores the full band in one pass, runs faster than realtime, and is the only model in this table that targets MP3
  by name.
- **[SonicMaster](#ref-sonicmaster)** maps degraded audio to mastered audio with text-instructed flow matching,
  covering nineteen degradations across EQ, dynamics, reverb, amplitude and stereo. MP3/AAC/Opus damage is not among
  them.

### Generative music priors

These are the models a restoration prior could be adapted from. All of them commit to a compressed latent or token
space with a reconstruction stage on top.

- **[Stable Audio 3](#ref-sa3)** is a family of fast latent diffusion models (small, medium, large) over the SAME
  space, with variable-length generation, inpainting, and adversarial post-training; small and medium are open
  weights.
- **[Qwen-Music](#ref-qwenmusic)** runs an LLM over 25 Hz semantic tokens, plans melodies with a chain-of-thought
  mechanism, and renders full sung songs in stereo at state-of-the-art musicality. No weights are public.
- **[Qwen-Audio-3.0-Gen](#ref-qwengen)** renders whole multi-role audio scenes in a single pass, using one diffusion
  transformer over a shared 48 kHz stereo VAE. It is available only as a hosted preview.

### Inverse problems with a diffusion prior

In this line a prior proposes clean audio while the sampler holds it consistent with the observation.
[DPS](#ref-dps) established posterior sampling with a known operator, [SILO](#ref-silo) moved it into an
autoencoder's latent space, and [DAPS](#ref-daps) fixes the oversmoothing that matters here, because smearing is the
dominant artifact. [CQT-Diff](#ref-cqtdiff) brought the approach to music audio, and [BABE](#ref-babe) /
[BABE-2](#ref-babe2) apply it blind — operator unknown — to historical recordings. **[LOUDAR](#ref-loudar)** learns
the unknown operator in latent space while a latent diffusion prior regularises the estimate; it is demonstrated on
singing-voice effects and guitar distortion, not on MP3/AAC/Opus damage.

### The table

The table is sorted by year. The *Restores* column says what a method can reconstruct: the **full band** including
in-band damage, the **missing band** above a cutoff, a **round-trip** (an autoencoder's `decode(encode(x))`), or a
**prior** that needs a solver; a dash marks methods that do not restore at all. The *Weights* column distinguishes
**open** releases, **code**-only releases, **closed** hosted models, weights **announced** but unreleased, and
**none**.

| Paper | Year | Method | Restores | Targets MP3 | Weights | Tested here |
|---|---|---|---|---|---|---|
| [AudioSR](#ref-audiosr) | 2023 | diffusion SR, mel domain + vocoder | missing band | — | open | — |
| [Apollo](#ref-apollo) | 2024 | band-split GAN regression, waveform | full band | ✓ | open | ✓ [baseline findings](docs/decisions/0006-baseline-findings.md), [MP3-twin benchmark](docs/decisions/0007-benchmark-codec-restoration.md) |
| [A2SB](#ref-a2sb) | 2025 | Schrödinger bridge, vocoder-free | missing band + gaps | — | open¹ | ✓ [baseline findings](docs/decisions/0006-baseline-findings.md), [MP3-twin benchmark](docs/decisions/0007-benchmark-codec-restoration.md) |
| [Latent upsample/upmix](#ref-latentup) | 2025 | supervised latent-to-latent BWE/upmix | missing band | — | none | — |
| [SonicMaster](#ref-sonicmaster) | 2025 | text-guided flow matching, all-in-one | in-band faults² | — | open | demo — [baselines and prior viability](docs/decisions/0005-baselines-and-prior-viability-on-real-library-material.md) |
| [εar-VAE](#ref-earvae) | 2025 | perceptual VAE, phase + M/S losses | round-trip | — | open | — |
| [AudioLBM](#ref-audiolbm) | 2025 | latent bridge SR, cascaded to 192 kHz | missing band | — | none | — |
| [HP-codecX](#ref-hpcodec) | 2025 | codec tokens + transformer LM | missing band | — | open | — |
| [Stable Audio 3](#ref-sa3) | 2026 | latent diffusion prior + inpainting | prior | — | open | ✓ [SA3 generation probe](docs/decisions/0008-sa3-generation-probe.md), [SDEdit as the first solver](docs/decisions/0009-sdedit-as-the-first-solver.md) |
| [SAME](#ref-same) | 2026 | transformer autoencoder, 4096× | round-trip | — | open | ✓ [MP3-twin benchmark](docs/decisions/0007-benchmark-codec-restoration.md) |
| [Elastic Time](#ref-elastic) | 2026 | dynamic frame-rate bottleneck | — | — | code | — |
| [Qwen-Music](#ref-qwenmusic) | 2026 | semantic-token LLM + stereo renderer | — | — | closed | — |
| [Qwen-Audio-3.0-Gen](#ref-qwengen) | 2026 | DiT + shared VAE, scene generation | — | — | closed | — |
| [LOUDAR](#ref-loudar) | 2026 | latent operator + diffusion prior | full band, blind | — | code | — |
| [εar-VAE2](#ref-earvae2) | 2026 | complex-STFT autoencoder, band refiner | round-trip | — | open³ | — |

¹ The released checkpoints are the 1- and 2-split models; the paper's 4-split ensemble was never published.
² Its nineteen degradations cover EQ, dynamics, reverb, amplitude and stereo; MP3/AAC/Opus and bandwidth damage are
not among them.
³ The open weights are retrained on public datasets, not the paper's internal corpus; reported numbers may not
transfer.

Only one model targets MP3 directly, and it is a regression model. Every generative method either stops at the
missing band or has not been applied to MP3, AAC or Opus. Repairing in-band codec damage with a generative model
remains unexplored.

## Experiments

We ran [Apollo](#ref-apollo) and [A2SB](#ref-a2sb) on real rips and on MP3s of clean masters, alongside a DSP control
and SAME round-trips
([baselines and prior viability](docs/decisions/0005-baselines-and-prior-viability-on-real-library-material.md)).
Every render was scored against its master on the same board
([the MP3-twin benchmark](docs/decisions/0007-benchmark-codec-restoration.md)). **Apollo is the most promising model
we tested, and it still fails to produce master-sounding output on the material targeted here**
([baseline findings](docs/decisions/0006-baseline-findings.md)). A2SB only fills what sits above the cutoff, by
design. We also probed whether the base Stable Audio 3 checkpoints generate master-like audio
([the SA3 generation probe](docs/decisions/0008-sa3-generation-probe.md)), and we evaluated
[SonicMaster](#ref-sonicmaster) through its hosted demo and excluded it, because it duplicates and stacks kick drums
on electronic material.

We then tried a first method beyond the supervised tools: **SDEdit ([Meng et al. 2022](#ref-sdedit)) with Stable
Audio 3** ([SDEdit as the first solver](docs/decisions/0009-sdedit-as-the-first-solver.md)). The solver encodes the
degraded track into SAME latents, mixes in noise at level *n*, and lets the model denoise from schedule time *n*; the
noise destroys detail, and the model re-synthesizes the destroyed content as clean music. A second variant adds no
noise and only starts the schedule at θ, asking the model to treat the damage itself as what it removes. Both fail
the same way: band-limited, codec-degraded audio lies inside the model's distribution of plausible music, so the
damage is preserved as signal — and by the time enough noise is added to destroy the damage, the track itself is
gone.

## Future work

- **The decoder as the restorer.** An autoencoder trained on clean music outputs clean-sounding audio, so a degraded
  input may come back cleaner from a bare round-trip — and better latents than `encode(x)` exist to be found, since
  the encoder is not the decoder's inverse. The SAME round-trip is already wired into
  [the MP3-twin benchmark](docs/decisions/0007-benchmark-codec-restoration.md). [εar-VAE](#ref-earvae) and
  [εar-VAE2](#ref-earvae2) are the next candidates to screen, since they are built for exactly the losses that MP3,
  AAC and Opus cause: phase, stereo image, and high-frequency harmonics.
- **Posterior sampling with a generative prior.** In the [DPS](#ref-dps)/[DAPS](#ref-daps)/[LOUDAR](#ref-loudar)
  line, a prior proposes full-band audio while each sampling step is held to consistency with the observation.
  [SDEdit as the first solver](docs/decisions/0009-sdedit-as-the-first-solver.md) records a first, failed attempt. We
  will explore the full version once the methods above are exhausted, with an operator that covers what MP3, AAC and
  Opus do inside the kept band rather than just the lowpass.

## References

<a id="ref-audiosr"></a>
[AudioSR] Liu et al. *AudioSR: Versatile Audio Super-resolution at Scale.* [arXiv:2309.07314](https://arxiv.org/abs/2309.07314) · [code](https://github.com/haoheliu/versatile_audio_super_resolution)

<a id="ref-apollo"></a>
[Apollo] Li & Luo. *Apollo: Band-sequence Modeling for High-Quality Audio Restoration.* ICASSP 2025. [arXiv:2409.08514](https://arxiv.org/abs/2409.08514) · [code](https://github.com/JusperLee/Apollo)

<a id="ref-a2sb"></a>
[A2SB] Kong et al. (NVIDIA). *A2SB: Audio-to-Audio Schrödinger Bridges.* [arXiv:2501.11311](https://arxiv.org/abs/2501.11311) · [code](https://github.com/NVIDIA/diffusion-audio-restoration)

<a id="ref-latentup"></a>
[LatentUp] Bralios, Smaragdis, Casebeer. *Learning to Upsample and Upmix Audio in the Latent Domain.* WASPAA 2025. [arXiv:2506.00681](https://arxiv.org/abs/2506.00681)

<a id="ref-sonicmaster"></a>
[SonicMaster] Melechovsky et al. *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering.* ICML 2026. [arXiv:2508.03448](https://arxiv.org/abs/2508.03448) · [weights](https://huggingface.co/amaai-lab/SonicMaster)

<a id="ref-earvae"></a>
[εar-VAE] Wang et al. *Back to Ear: Perceptually Driven High Fidelity Music Reconstruction.* [arXiv:2509.14912](https://arxiv.org/abs/2509.14912) · [code](https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE) · [weights](https://huggingface.co/earlab/EAR_VAE)

<a id="ref-audiolbm"></a>
[AudioLBM] Li et al. *Audio Super-Resolution with Latent Bridge Models.* NeurIPS 2025. [arXiv:2509.17609](https://arxiv.org/abs/2509.17609)

<a id="ref-hpcodec"></a>
[HP-codecX] Giniès et al. *Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth Extension.* [arXiv:2511.21580](https://arxiv.org/abs/2511.21580) · [code](https://zenodo.org/records/22144166) · [weights](https://zenodo.org/records/22144249)

<a id="ref-sa3"></a>
[SA3] Evans, Parker et al. (Stability AI). *Stable Audio 3.* [arXiv:2605.17991](https://arxiv.org/abs/2605.17991) · [code](https://github.com/Stability-AI/stable-audio-3) · [weights](https://huggingface.co/stabilityai/stable-audio-3-small-music)

<a id="ref-same"></a>
[SAME] Parker, Evans et al. *SAME: A Semantically-Aligned Music Autoencoder.* [arXiv:2605.18613](https://arxiv.org/abs/2605.18613) · [weights](https://huggingface.co/stabilityai/SAME-L)

<a id="ref-elastic"></a>
[Elastic] Bralios, Smaragdis, Kim. *Elastic Time: Dynamic Frame Rate Bottlenecks for Neural Audio Coding.* Interspeech 2026. [arXiv:2606.27320](https://arxiv.org/abs/2606.27320) · [code](https://github.com/dbralios/elastic-time)

<a id="ref-qwenmusic"></a>
[Qwen-Music] Xu et al. (Alibaba). *Qwen-Music Technical Report.* [arXiv:2607.11699](https://arxiv.org/abs/2607.11699)

<a id="ref-qwengen"></a>
[Qwen-Gen] Dai et al. (Alibaba). *Qwen-Audio-3.0-Gen-Preview Technical Report.* [arXiv:2607.27011](https://arxiv.org/abs/2607.27011)

<a id="ref-loudar"></a>
[LOUDAR] Švento et al. *Music Restoration via Latent Operator Optimization and Diffusion Model Priors.* ISMIR 2026. [arXiv:2608.01972](https://arxiv.org/abs/2608.01972) · [code](https://github.com/michalsvento/loudar) · [demo](https://michalsvento.github.io/loudar/)

<a id="ref-earvae2"></a>
[εar-VAE2] Wang, Dai, Xu. *Fourier is Frontier: Frequency-Aware Autoencoding for High-Fidelity Music Reconstruction.* [arXiv:2608.19843](https://arxiv.org/abs/2608.19843) · [code](https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE2) · [weights](https://huggingface.co/earlab/EAR_VAE2) · [demo](https://eps-acoustic-revolution-lab.github.io/EAR_VAE2/)

<a id="ref-sdedit"></a>
[SDEdit] Meng et al. *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations.* ICLR 2022. [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)

<a id="ref-dps"></a>
[DPS] Chung et al. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

<a id="ref-silo"></a>
[SILO] Raphaeli, Man, Elad. *SILO: Solving Inverse Problems with Latent Operators.* ICCV 2025. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)

<a id="ref-daps"></a>
[DAPS] Zhang et al. *Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing.* CVPR 2025. [arXiv:2407.01521](https://arxiv.org/abs/2407.01521)

<a id="ref-cqtdiff"></a>
[CQT-Diff] Moliner, Lehtinen, Välimäki. *Solving Audio Inverse Problems with a Diffusion Model.* [arXiv:2210.15228](https://arxiv.org/abs/2210.15228)

<a id="ref-babe"></a>
[BABE] Moliner et al. *Blind Audio Bandwidth Extension: A Diffusion-Based Zero-Shot Approach.* [arXiv:2306.01433](https://arxiv.org/abs/2306.01433)

<a id="ref-babe2"></a>
[BABE-2] Moliner et al. *A Diffusion-Based Generative Equalizer for Music Restoration.* DAFx-24. [arXiv:2403.18636](https://arxiv.org/abs/2403.18636)

<a id="ref-ytformats"></a>
[YT-Formats] Eesmaa, M. *YouTube Audio Formats — itags, codecs and bitrates.* [gist](https://gist.github.com/MartinEesmaa/2f4b261cb90a47e9c41ba115a011a4aa)

---

The code here is MIT-licensed. Any weights adapted from Stable Audio 3 would inherit the
[Stability AI Community License](https://stability.ai/license), which permits the personal and research use this
project is scoped to.

Decisions are recorded as ADRs in [`docs/decisions/`](docs/decisions/).
