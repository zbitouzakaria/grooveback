# grooveback

**Restoring records that survive only as MP3, AAC or Opus rips.**

**[Listen to the results](https://zbitouzakaria.github.io/grooveback/listen/)**: clean tracks, their MP3 copies, and
the restorations described below.

As a DJ and listener, I keep running into the same problem: records I love that now exist only as rips on YouTube. A rip
is a copy someone made of their own record. They played the vinyl through an imperfect turntable setup, saved the
recording as a 128 or 192 kbps MP3, and uploaded it. YouTube then compressed it again, to AAC or Opus. On laptop
speakers the rips sound fine. On a good sound system the damage is easy to hear: drum hits lose their sharp edge, the
shimmer of the cymbals is gone, and the track sounds flat and tired next to the rest of the library.

This project tries to restore them, starting with electronic music: old-school house, minimal, and nearby genres.

## What an MP3 throws away

An MP3 at 128 kilobits per second (kbps) stores music in about a tenth of the space a CD uses. The lower the bitrate,
the more it throws away. To fit, the encoder drops what the ear is least likely to miss. It removes every frequency
above a cut-off, the top band. It stores the rest less precisely, which blurs short sounds such as a drum hit. When it
runs short of data, it drops quiet parts entirely. None of this comes back when the file is played. AAC and Opus work
the same way. A YouTube rip went through this at least twice: when the uploader saved it, and again when YouTube
re-encoded it to roughly 128 kbps AAC or Opus ([YouTube audio formats](#ref-ytformats)).

The spectrograms below show 6 s of an electronic track. Time runs left to right, frequency bottom to top, and brighter
means louder. The master, the track as released, reaches 20 kHz. Its 128 kbps MP3 stops at about 16.5 kHz. Its 32 kbps
MP3 stops at 5.5 kHz, with dark holes below that where quiet parts were dropped.

![Spectrograms of 6 s of an electronic track: the master, its MP3 at 128 kbps, and its MP3 at 32 kbps](docs/assets/mp3-spectrograms.png)

## A restorer has to invent what the MP3 removed

Since the information is gone, no restorer can recover the master. It has to invent the missing sound, so its result can
be plausible but never identical to the master. That is a job for a generative model. Our plan is to take
[Stable Audio 3](#ref-sa3), an open music generator, and train it further on my library of about 215 hours. We call the
trained model the prior: it knows only what clean music sounds like. A second program, the solver, uses the prior to
produce a clean track that matches the rip.

We have not trained the prior yet. First we measured what existing tools do.

## What we found

### Apollo restores heavy damage better than A2SB, by ear

[Apollo](#ref-apollo) is trained to repair MP3 damage. [A2SB](#ref-a2sb) only fills in the top band. On a heavily
damaged 6 s clip, Apollo put back a full hi-hat where A2SB added a click. The picture shows that clip as a 32 kbps MP3:
Apollo fills the top band and the holes below the cut-off, while A2SB only adds short bursts above it. On a real
128 kbps rip, the MP3 and both restorations sounded the same on headphones.

![Spectrograms of a 6 s clip: the master, its MP3 at 32 kbps, Apollo's restoration, and A2SB's restoration](docs/assets/restorations-32k.png)

### Stable Audio 3 generates the kind of detail an MP3 removes

A prior cannot put back detail it cannot generate. We compressed clips from Stable Audio 3's music models to 64 kbps
MP3. The MP3 removed about as much from them as from a real master, so the clips contain that kind of detail. This used
one text prompt, and it does not show that the clips sound like a master.

### SDEdit keeps the damage it should remove

Stable Audio 3 generates music by removing noise step by step. Our first solver, [SDEdit](#ref-sdedit), adds noise to
the MP3 and lets the model remove it, in the hope that the damage goes with it. It does not: to the model, music without
its top band is still plausible music. We rejected SDEdit, and will test it again once our own prior is trained.

### Apollo followed by εar-VAE sounds best on heavy damage, by ear

We tried four open autoencoders, networks that compress audio and rebuild it. An autoencoder trained on clean music
tends to rebuild what it hears as clean music, so it may remove some of the damage. On heavy damage, the best by ear is
Apollo's output passed through [εar-VAE](#ref-earvae). It sounds better than Apollo alone at 32 kbps, neither clearly
wins at 64 kbps, and at 128 kbps Apollo alone sounds closer to the master. Scores that measure how close a restoration
is to the master all rate it below Apollo alone, so here we trust the ear. Our listening is not blind yet: we knew which
version we were hearing.

### On MUSDB18-HQ, no method scores above the untouched MP3 on SDR

To score methods the way published work does, we used 6 s clips of the MUSDB18-HQ test songs, a public music dataset,
compressed to MP3 at 24 to 128 kbps. The figure shows Apollo, A2SB, [AudioSR](#ref-audiosr) (another model that fills in
the top band), Apollo followed by εar-VAE, and the MP3 passed through εar-VAE or εar-VAE2 alone.

The four scores are different ways to measure how similar a restoration is to the master. SDR and SI-SNR compare the
waveforms, and spectral SNR and LSD compare the spectrograms. Higher is better, except for LSD.

![SDR, SI-SNR, spectral SNR and LSD per MP3 bitrate on 50 MUSDB18-HQ test clips, for six restorations and the untouched MP3](docs/decisions/assets/0016/methods.png)

The grey line is the untouched MP3. No method scores above it on SDR.

The full write-up of each experiment, with its evidence, is in [`docs/decisions/`](docs/decisions/).

## Related work

The methods differ in what damage they repair and where they work: on the waveform, on spectrograms, or in the latent
space of a neural autoencoder, the shorter list of numbers it compresses audio into. Most rebuild the band above a
cut-off, or fix faults like reverb and clipping. Few target what MP3, AAC and Opus do inside the band they keep.

- **Supervised restorers** learn from pairs of damaged and clean audio. [SR-GAN](#ref-srgan) and [Apollo](#ref-apollo)
  are trained on MP3 damage, and SR-GAN's weights are not available. [SonicMaster](#ref-sonicmaster) does not cover
  codec damage.
- **Bandwidth extension** rebuilds the band above a cut-off: [AudioSR](#ref-audiosr), [A2SB](#ref-a2sb),
  [AudioLBM](#ref-audiolbm), [HP-codecX](#ref-hpcodec) and [latent upsampling](#ref-latentup). None of them trains on
  codec damage.
- **Music autoencoders** give generative models their latent space. [SAME](#ref-same) is Stable Audio 3's autoencoder.
  Its large (L) and small (S) versions are both open. [εar-VAE](#ref-earvae) and [εar-VAE2](#ref-earvae2) are built to
  keep phase, stereo image and high harmonics. [CoDiCodec](#ref-codicodec) gives continuous and discrete latents from
  one model, and [Elastic Time](#ref-elastic) spends latent frames where the signal is dense. A method that works in a
  latent space cannot sound better than the decoder.
- **Generative music models** a restoration prior could start from: [Stable Audio 3](#ref-sa3), whose small and medium
  checkpoints are open, and [Qwen-Music](#ref-qwenmusic) and [Qwen-Audio-3.0-Gen](#ref-qwengen), which have no open
  weights.
- **Inverse problems with a diffusion prior.** The prior proposes clean audio, and the solver keeps it consistent with
  the observed signal. [DPS](#ref-dps) introduced this with a known damage model, [SILO](#ref-silo) moved it into a
  latent space, and [DAPS](#ref-daps) reduces the over-smoothing that blurs short, sharp sounds such as drum hits.
  [CQT-Diff](#ref-cqtdiff) brought it to music, [BABE](#ref-babe) and [BABE-2](#ref-babe2) apply it to historical
  recordings without knowing the damage, and [LOUDAR](#ref-loudar) learns the unknown damage in a latent space, shown on
  vocal effects and guitar distortion. We found none tried on MP3, AAC or Opus damage.

*Restores* says what a method reconstructs: the **full band** including in-band damage, the **missing band** above a
cut-off, a **round trip** (an autoencoder's `decode(encode(x))`), or a **prior** that needs a solver. A dash means it
does not restore. **Gaps** are short stretches of missing audio, and **blind** means the method is not told what the
damage is. *Weights* are **open**, **code** only, **closed** (hosted), or **none**.

| Paper | Year | Method | Restores | Targets MP3 | Weights | Tested here |
|---|---|---|---|---|---|---|
| [SR-GAN](#ref-srgan) | 2021 | stochastic GAN, complex STFT | full band | ✓ | none | — |
| [AudioSR](#ref-audiosr) | 2023 | diffusion bandwidth extension, mel domain + vocoder | missing band | — | open | ✓ |
| [Apollo](#ref-apollo) | 2024 | band-split GAN, waveform | full band | ✓ | open | ✓ |
| [A2SB](#ref-a2sb) | 2025 | Schrödinger bridge, vocoder-free | missing band + gaps | — | open¹ | ✓ |
| [Latent upsample/upmix](#ref-latentup) | 2025 | supervised latent-to-latent bandwidth extension and upmix | missing band | — | none | — |
| [SonicMaster](#ref-sonicmaster) | 2025 | text-guided flow matching, all-in-one | in-band faults² | — | open | demo only, dropped |
| [CoDiCodec](#ref-codicodec) | 2025 | consistency autoencoder, continuous + discrete latents | round trip | — | open | ✓ |
| [εar-VAE](#ref-earvae) | 2025 | perceptual VAE, phase + M/S losses | round trip | — | open | ✓ |
| [AudioLBM](#ref-audiolbm) | 2025 | latent bridge bandwidth extension, cascaded to 192 kHz | missing band | — | none | — |
| [HP-codecX](#ref-hpcodec) | 2025 | codec tokens + transformer LM | missing band | — | open | — |
| [Stable Audio 3](#ref-sa3) | 2026 | latent diffusion prior + inpainting | prior | — | open | ✓ (SDEdit, rejected) |
| [SAME](#ref-same) | 2026 | transformer autoencoder, 4096× | round trip | — | open | ✓ |
| [Elastic Time](#ref-elastic) | 2026 | dynamic frame-rate bottleneck | — | — | code | — |
| [Qwen-Music](#ref-qwenmusic) | 2026 | semantic-token LLM + stereo renderer | — | — | closed | — |
| [Qwen-Audio-3.0-Gen](#ref-qwengen) | 2026 | DiT + shared VAE, scene generation | — | — | closed | — |
| [LOUDAR](#ref-loudar) | 2026 | latent operator + diffusion prior | full band, blind | — | code | — |
| [εar-VAE2](#ref-earvae2) | 2026 | complex-STFT autoencoder, band refiner | round trip | — | open³ | ✓ |

¹ Only the 1- and 2-split checkpoints are released; the paper's 4-split ensemble is not.
² Its nineteen faults cover EQ, dynamics, reverb, amplitude and stereo; MP3/AAC/Opus and bandwidth damage are not
among them.
³ The open weights are retrained on public datasets, not the paper's internal corpus. By ear they sound clearly worse
than εar-VAE.

Two models target MP3 directly, SR-GAN and Apollo, and both are supervised. Outside this project, every other
generative method either stops at the missing band or has not been tried on MP3, AAC or Opus. Repairing in-band codec
damage with a prior trained only on clean music is still open.

## Running the code

The code runs with [uv](https://docs.astral.sh/uv/) and Python 3.12. Apollo, εar-VAE and εar-VAE2 come in as git
submodules:

```bash
git submodule update --init --recursive
uv sync --all-groups
uv run pytest
```

The MUSDB18-HQ benchmark scores Apollo on the 50 test songs at one MP3 bitrate and writes one row per 6 s clip. Its MP3
step needs sox with MP3 support (on Ubuntu, `apt install sox libsox-fmt-mp3`); without it, the tests that need it are
skipped.

```bash
uv run python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps 24
```

A smaller benchmark restores one excerpt per track at 32, 64 and 128 kbps with every method, and scores the results
against the clean excerpt. It reads the clean tracks from `data/`, which is not in the repository.

```bash
uv run python scripts/run_xp.py device=cuda
```

The library is in `src/grooveback`, the scripts that run the models and build the listening page are in `scripts/`, and
the page itself is in `docs/listen/`. A2SB and AudioSR need dependency versions that conflict with this project's, so
each runs in its own environment; [`.claude/rules/environment.md`](.claude/rules/environment.md) says how to set them up
and which jobs need a rented GPU.

## References

<a id="ref-srgan"></a>
[SR-GAN] Lattner & Nistal. *Stochastic Restoration of Heavily Compressed Musical Audio Using Generative Adversarial Networks.* Electronics 2021. [arXiv:2207.01667](https://arxiv.org/abs/2207.01667) · called SR-GAN in the Apollo paper

<a id="ref-audiosr"></a>
[AudioSR] Liu et al. *AudioSR: Versatile Audio Super-resolution at Scale.* [arXiv:2309.07314](https://arxiv.org/abs/2309.07314) · [code](https://github.com/haoheliu/versatile_audio_super_resolution)

<a id="ref-apollo"></a>
[Apollo] Li & Luo. *Apollo: Band-sequence Modeling for High-Quality Audio Restoration.* ICASSP 2025. [arXiv:2409.08514](https://arxiv.org/abs/2409.08514) · [code](https://github.com/JusperLee/Apollo)

<a id="ref-a2sb"></a>
[A2SB] Kong et al. (NVIDIA). *A2SB: Audio-to-Audio Schrödinger Bridges.* [arXiv:2501.11311](https://arxiv.org/abs/2501.11311) · [code](https://github.com/NVIDIA/diffusion-audio-restoration)

<a id="ref-latentup"></a>
[Latent upsample/upmix] Bralios, Smaragdis, Casebeer. *Learning to Upsample and Upmix Audio in the Latent Domain.* WASPAA 2025. [arXiv:2506.00681](https://arxiv.org/abs/2506.00681)

<a id="ref-sonicmaster"></a>
[SonicMaster] Melechovsky et al. *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering.* ICML 2026. [arXiv:2508.03448](https://arxiv.org/abs/2508.03448) · [weights](https://huggingface.co/amaai-lab/SonicMaster)

<a id="ref-codicodec"></a>
[CoDiCodec] Pasini, Lattner, Fazekas. *CoDiCodec: Unifying Continuous and Discrete Compressed Representations of Audio.* ISMIR 2025. [arXiv:2509.09836](https://arxiv.org/abs/2509.09836) · [code](https://github.com/SonyCSLParis/codicodec) · [weights](https://huggingface.co/SonyCSLParis/codicodec)

<a id="ref-earvae"></a>
[εar-VAE] Wang et al. *Back to Ear: Perceptually Driven High Fidelity Music Reconstruction.* [arXiv:2509.14912](https://arxiv.org/abs/2509.14912) · [code](https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE) · [weights](https://huggingface.co/earlab/EAR_VAE)

<a id="ref-audiolbm"></a>
[AudioLBM] Li et al. *Audio Super-Resolution with Latent Bridge Models.* NeurIPS 2025. [arXiv:2509.17609](https://arxiv.org/abs/2509.17609)

<a id="ref-hpcodec"></a>
[HP-codecX] Giniès et al. *Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth Extension.* [arXiv:2511.21580](https://arxiv.org/abs/2511.21580) · [code](https://zenodo.org/records/22144166) · [weights](https://zenodo.org/records/22144249)

<a id="ref-sa3"></a>
[Stable Audio 3] Evans, Parker et al. (Stability AI). *Stable Audio 3.* [arXiv:2605.17991](https://arxiv.org/abs/2605.17991) · [code](https://github.com/Stability-AI/stable-audio-3) · [weights](https://huggingface.co/stabilityai/stable-audio-3-small-music)

<a id="ref-same"></a>
[SAME] Parker, Evans et al. *SAME: A Semantically-Aligned Music Autoencoder.* [arXiv:2605.18613](https://arxiv.org/abs/2605.18613) · [weights](https://huggingface.co/stabilityai/SAME-L)

<a id="ref-elastic"></a>
[Elastic Time] Bralios, Smaragdis, Kim. *Elastic Time: Dynamic Frame Rate Bottlenecks for Neural Audio Coding.* Interspeech 2026. [arXiv:2606.27320](https://arxiv.org/abs/2606.27320) · [code](https://github.com/dbralios/elastic-time)

<a id="ref-qwenmusic"></a>
[Qwen-Music] Xu et al. (Alibaba). *Qwen-Music Technical Report.* [arXiv:2607.11699](https://arxiv.org/abs/2607.11699)

<a id="ref-qwengen"></a>
[Qwen-Audio-3.0-Gen] Dai et al. (Alibaba). *Qwen-Audio-3.0-Gen-Preview Technical Report.* [arXiv:2607.27011](https://arxiv.org/abs/2607.27011)

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

The code here is MIT-licensed. The third-party code it runs has its own licenses: Apollo CC BY-SA 4.0, A2SB the NVIDIA
Source Code License (non-commercial), AudioSR MIT, εar-VAE and εar-VAE2 Apache-2.0, CoDiCodec CC BY-NC 4.0, and the
vendored trackswitch.js player MIT. The Stable Audio 3 checkpoints are released under the
[Stability AI Community License](https://stability.ai/license), which any weights adapted from them would inherit; it
permits the personal and research use this project is scoped to. The audio on the listening page belongs to its owners
and is not covered by the MIT license.
