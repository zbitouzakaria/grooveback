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

## What does an MP3 throw away?

An MP3 at 128 kilobits per second (kbps) stores music in about a tenth of the space a CD uses. To fit, the encoder drops
what the ear is least likely to miss, and the lower the bitrate, the more it drops. It removes every frequency above a
cut-off; we call the range above it the top band. It stores the rest less precisely, which blurs short sounds such as a
drum hit. When it runs short of data, it drops quiet parts entirely. None of this comes back when the file is played.
AAC and Opus work the same way. A YouTube rip went through this at least twice: once when the uploader saved it, and
again when YouTube re-encoded it to roughly 128 kbps AAC or Opus ([YouTube audio formats](#ref-ytformats)).

The spectrograms below show 6 s of an electronic track. Time runs left to right, frequency bottom to top, and brighter
means louder. The master, the track as released, reaches 20 kHz. Its 128 kbps MP3 stops at about 16.5 kHz. Its 32 kbps
MP3 stops at 5.5 kHz, with dark holes below that where quiet parts were dropped.

![Spectrograms of 6 s of an electronic track: the master, its MP3 at 128 kbps, and its MP3 at 32 kbps](docs/assets/mp3-spectrograms.png)

## Can a restorer bring back what the MP3 removed?

The information is gone, so no restorer can recover the master exactly. It has to invent the missing sound, and its
result can be plausible but never identical to the master. Inventing plausible sound is what generative models do. Our
plan is to take [Stable Audio 3](#ref-sa3), an open music generator, and train it further on my own library of about
215 hours of music. We call the trained model the prior: it knows only what clean music sounds like, and never hears an
MP3. A second program, the solver, uses the prior to produce a clean track that matches the rip.

We have not trained the prior yet. First, we looked at what existing methods do, and measured the open ones.

## Related work

This section reviews the current literature on the subject. Not every method below targets MP3 damage, but each is close
enough to the problem to be worth testing on it. We group the methods by how they work, and the table at the end lists
them all.

**Supervised restorers** learn from pairs of damaged and clean audio. In training, the model receives the damaged
version and learns to output the clean one, so it repairs the kinds of damage it was trained on. [SR-GAN](#ref-srgan)
and [Apollo](#ref-apollo) are trained on MP3 damage. SR-GAN takes random noise as a second input, so it can produce
several plausible restorations of one MP3, but its weights are not available. Apollo splits the spectrum into bands and
models the bands as a sequence, and its weights are open. [SonicMaster](#ref-sonicmaster) uses one model, guided by a
text prompt, to repair many mixing and mastering faults such as reverb and equalization, but codec damage is not among
them.

**Bandwidth extension**, also called audio super-resolution, rebuilds the band above a cut-off from the band below it.
The missing band is the most visible part of MP3 damage, as the spectrograms above show, but not the only part. None of
the models below is trained on codec damage. [AudioSR](#ref-audiosr) generates a mel spectrogram with a diffusion model,
and a vocoder turns it into audio. [A2SB](#ref-a2sb) uses a Schrödinger bridge, a diffusion process that starts from the
damaged audio instead of from noise, and it also fills short gaps. [AudioLBM](#ref-audiolbm) runs such a bridge in a
latent space, and chains several models to reach a 192 kHz sample rate. [HP-codecX](#ref-hpcodec) encodes audio with a
neural codec that keeps harmonic and percussive sound apart, and predicts the missing band with a transformer language
model over the codec's tokens. [Latent upsampling](#ref-latentup) maps the latent of band-limited audio directly to the
latent of full-band audio, and learns to turn mono into stereo the same way.

**Music autoencoders** compress audio into a latent and decode it back. Most generative music models work in such a
latent space because it is small, so every sound they produce passes through the decoder. A method that works in a
latent space therefore cannot sound better than the decoder. A round trip through the encoder and the decoder can also
serve as a simple restorer, as we try below. [SAME](#ref-same) is Stable Audio 3's autoencoder, and its large (L) and
small (S) versions are both open. [εar-VAE](#ref-earvae) and [εar-VAE2](#ref-earvae2) are built to keep phase, stereo
image and high harmonics. [CoDiCodec](#ref-codicodec) gives continuous and discrete latents from one model, and
[Elastic Time](#ref-elastic) spends more latent frames where the signal is dense.

**Generative music models** learn from large collections of music and generate new tracks, usually from a text prompt. A
restoration prior can start from one of them, as ours will start from Stable Audio 3. [Stable Audio 3](#ref-sa3) is a
latent diffusion model that can also fill in missing parts of a track, and its small and medium checkpoints are open.
[Qwen-Music](#ref-qwenmusic) generates semantic tokens with a language model and renders them as stereo audio, and
[Qwen-Audio-3.0-Gen](#ref-qwengen) generates sound scenes with a diffusion transformer. Neither has open weights.

**Inverse-problem solvers with a diffusion prior** start from the observed signal and ask which clean signal produced
it. A diffusion model trained on clean signals proposes a clean signal step by step, and the solver keeps each step
consistent with the observed signal through a model of the damage. Our plan belongs to this family. [DPS](#ref-dps)
introduced the approach with a known damage model, [SILO](#ref-silo) moved it into a latent space, and [DAPS](#ref-daps)
reduces the over-smoothing that blurs short, sharp sounds such as drum hits. [CQT-Diff](#ref-cqtdiff) brought it to
music. [BABE](#ref-babe) and [BABE-2](#ref-babe2) apply it to historical recordings without knowing the damage.
[LOUDAR](#ref-loudar) learns the unknown damage in a latent space, and its authors show it on vocal effects and guitar
distortion. We found none that has been tried on MP3, AAC or Opus damage.

The table lists the methods by year. *Restores* says what a method rebuilds: the **full band**, with the damage inside
it repaired; the **missing band** above a cut-off; a **round trip**, an autoencoder's `decode(encode(x))`; or a
**prior**, which needs a solver. A dash means it does not restore. **Gaps** are short stretches of missing audio, and
**blind** means the method is not told what the damage is. *Weights* are **open**, **code** only, **closed** (hosted),
or **none**.

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

Two models target MP3 directly, SR-GAN and Apollo, and both are supervised. Every other generative method either stops
at the missing band or, outside this project, has not been tried on MP3, AAC or Opus. Whether a prior trained only on
clean music can repair the damage a codec leaves inside the band it keeps is still an open question, and it is the
question this project sets out to answer.

## Benchmark on MUSDB18-HQ

We tested the three open restorers in the table, Apollo, A2SB and AudioSR, the way published work compares methods: on
6 s clips from the test songs of the public MUSDB18-HQ dataset, compressed to MP3 at 24 to 128 kbps. The figure also
shows three runs from [our own experiments](#our-contribution): the MP3 passed through εar-VAE or εar-VAE2, and Apollo's
output passed through εar-VAE.

We use four scores, each a different way to measure how similar a restoration is to the master. SDR and SI-SNR compare
the waveforms, and spectral SNR and LSD compare the spectrograms. Higher is better, except for LSD.

![SDR, SI-SNR, spectral SNR and LSD per MP3 bitrate on 50 MUSDB18-HQ test clips, for six restorations and the untouched MP3](docs/decisions/assets/0016/methods.png)

The two kinds of score rank the restorers differently. On the waveform scores, A2SB and Apollo come first: A2SB is ahead
by about 0.5 dB at 32 to 64 kbps, and the two are within about 0.2 dB of each other at the other bitrates. AudioSR is 2
to 5 dB behind them on SDR. On the spectrogram scores, Apollo is the best method at every bitrate.

By ear, Apollo sounds far better than anything else we tried, including on a real 128 kbps rip. Apollo is trained on MP3
damage, while A2SB, as we use it, only fills in the top band. The figure below shows the difference on a heavily damaged
6 s clip, compressed to 32 kbps: Apollo fills the top band and the holes below the cut-off, while A2SB only adds short
bursts above it. On this clip, Apollo put back a full hi-hat where A2SB added a click.

![Spectrograms of a 6 s clip: the master, its MP3 at 32 kbps, Apollo's restoration, and A2SB's restoration](docs/assets/restorations-32k.png)

As a curiosity, we also scored the untouched MP3 against the master; its scores are the grey line. No method beats it on
the waveform scores. On LSD, Apollo, A2SB, AudioSR and Apollo followed by εar-VAE all beat it at every bitrate, but on
spectral SNR only Apollo clearly does, by 2.2 to 2.7 dB. Ideally, a restoration would score at least as well as the MP3
it starts from; one that scores worse may lose relevant information while it rebuilds the signal.

## Our contribution

The restorers above either learn MP3 damage from pairs of damaged and clean audio, or only fill in the top band. Our
experiments so far test the route our plan takes: restoring with models that have only heard clean music.

### Can a denoiser started from the MP3 converge to a plausible master?

Stable Audio 3 is a diffusion model: it generates music by starting from pure noise and removing it step by step.
[SDEdit](#ref-sdedit), first proposed for images, uses such a model to edit an existing input rather than generate one
from scratch, and it is the simplest way to restore with a model that has only heard clean music. It adds noise to the
MP3 and lets the model remove it, in the hope that the damage goes away with the noise. The damage stays: to the model,
music without its top band is still plausible music, so the model treats the damage as part of the signal. With more
noise the model rebuilds more, but its output moves audibly away from the track well before the noise is strong enough
to cover the damage. We rejected SDEdit, and will test it again once our own prior is trained.

### Can an autoencoder trained on clean music remove the damage?

An autoencoder is a network that compresses audio into a much shorter list of numbers, its latent, and rebuilds the
audio from it. One trained only on clean music may rebuild a damaged input as clean music, and so remove part of the
damage. We tried four open music autoencoders, on the MP3 itself and on Apollo's restoration of it.

Applied to the MP3 alone, εar-VAE and εar-VAE2 score about 10.5 dB and 8.2 dB SDR in the benchmark, whatever the
bitrate, so the autoencoder, not the damage, sets their score. Applied to Apollo's output, εar-VAE helped only at
32 kbps, where it smooths the sound. At 64 and 128 kbps, it made Apollo's output sound worse. All four scores rate this
chain below Apollo alone at every bitrate. Our listening is not blind yet: we knew which version we were hearing.

The full write-up of each experiment, with its evidence, is in [`docs/decisions/`](docs/decisions/).

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
