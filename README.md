# grooveback

**Restoring records that only survive as MP3, AAC or Opus rips.**

[**Listen to the results**](https://zbitouzakaria.github.io/grooveback/lora/) — Stable Audio 3 fine-tuned with a LoRA to restore MP3s, next to Apollo, on the MUSDB18-HQ test songs. It's an initial run, and I'm now iterating on data augmentation and training strategies. The [earlier listening sets](https://zbitouzakaria.github.io/grooveback/listen/) have every open method and the LoRA on two tracks.

As a DJ and record collector, I keep hitting the same wall: a lot of the records I love only exist online as YouTube rips. Somebody played their copy on whatever turntable setup they had, saved it as a 128 or 192 kbps MP3, and uploaded it. Then YouTube re-encoded it to AAC or Opus. On laptop speakers you'd never notice. On a proper system you do: the drums lose their snap, the cymbals stop shimmering, and the track sounds flat and tired next to everything else in the set.

grooveback is my attempt to fix that. I'm starting with what I play most: old-school house, minimal, and the stuff around them.

## The problem: what does MP3 do to a music recording

A 128 kbps MP3 takes up roughly a tenth of the space of the same audio on CD. To get there, the encoder throws out whatever the ear is least likely to miss, and the lower the bitrate, the more goes. Three things happen:

- everything above a cut-off frequency is removed (I'll call that region the *top band*);
- what's left is stored less precisely, which smears short sounds like drum hits;
- when the encoder runs out of bits, quiet parts get dropped outright.

None of it comes back on playback. AAC and Opus work along the same lines, and a YouTube rip has likely been through the process at least twice: frequently on the uploader's machine, and again when YouTube re-encoded it to around 128 kbps AAC or Opus ([YouTube audio formats](#ref-ytformats)).

Here's 6 s of an electronic track as spectrograms (time left to right, frequency bottom to top, brighter = louder). The master, i.e. the track as released, goes all the way up to 20 kHz. The 128 kbps MP3 is cut off around 16.5 kHz. At 32 kbps the cut-off drops to 5.5 kHz, and you can see dark holes underneath where quiet material was thrown away.

[![Spectrograms of 6 s of an electronic track: the master, its MP3 at 128 kbps, and its MP3 at 32 kbps](docs/assets/mp3-spectrograms.png)](docs/assets/mp3-spectrograms.png)

## Can we undo the damage?

Not exactly. The information is gone, so any restorer has to make up the missing sound. The best it can do is plausible, never identical to the master. But making up plausible sound is exactly what generative models are good at.

So the plan is to take [Stable Audio 3](#ref-sa3), an open music generator, and fine-tune it on my own library (about 215 hours). That fine-tuned model we call the *prior*. It only ever hears clean music, never an MP3. A separate *solver* then uses the prior to find a clean track that's consistent with the rip.

Before training anything, I wanted to know what's already out there, and how well the open methods actually do. Then I made a first attempt at the prior: a LoRA fine-tune of Stable Audio 3 ([below](#a-lora-fine-tune-of-stable-audio-3)). I'm now working on other training strategies and on augmenting the dataset.

## Related work

Not everything here was built for MP3 damage, but all of it is close enough to be worth a try. I've grouped things by approach, with a summary table at the end.

**Supervised restorers.** Train on pairs of damaged and clean audio, and the model learns to repair that kind of damage. [SR-GAN](#ref-srgan) and [Apollo](#ref-apollo) are both trained on MP3 damage specifically. SR-GAN also takes a noise input, so it can give you several different plausible restorations of the same MP3; unfortunately the weights were never released. Apollo splits the spectrum into bands and models them as a sequence, and its weights are open. [SonicMaster](#ref-sonicmaster) is a single text-prompted model that fixes a long list of mixing and mastering problems (reverb, EQ and so on), but codec damage isn't one of them.

**Bandwidth extension** (a.k.a. audio super-resolution) rebuilds the top band from what's below it. That's the most obvious part of MP3 damage, as the spectrograms show, but it's not all of it, and none of these models were trained on codec artifacts. [AudioSR](#ref-audiosr) generates a mel spectrogram with diffusion and runs it through a vocoder. [A2SB](#ref-a2sb) uses a Schrödinger bridge, a diffusion process that starts from the damaged audio rather than from noise, and it can also fill short gaps. [AudioLBM](#ref-audiolbm) runs the bridge in latent space and cascades several models up to 192 kHz. [HP-codecX](#ref-hpcodec) uses a neural codec that keeps harmonic and percussive content separate, then predicts the missing band with a transformer LM over the codec tokens. [Latent upsampling](#ref-latentup) maps band-limited latents straight to full-band ones, and handles mono-to-stereo the same way.

**Music autoencoders** squeeze audio into a latent and decode it back. Most generative music models live in a latent space because it's small, which means everything they output goes through the decoder, so a latent method can never sound better than its decoder. An encode/decode round trip can also work as a crude restorer, which I try below. [SAME](#ref-same) is Stable Audio 3's autoencoder (both the L and S versions are open). [εar-VAE](#ref-earvae) and [εar-VAE2](#ref-earvae2) are designed to preserve phase, stereo image and high harmonics. [CoDiCodec](#ref-codicodec) gives you continuous and discrete latents from one model, and [Elastic Time](#ref-elastic) spends more latent frames where the signal is busy.

**Generative music models** learn from huge amounts of music and usually generate from a text prompt. A restoration prior can be built on top of one, which is what I'm doing with Stable Audio 3. [Stable Audio 3](#ref-sa3) is a latent diffusion model, can inpaint missing sections, and has open small and medium checkpoints. [Qwen-Music](#ref-qwenmusic) generates semantic tokens with an LLM and renders them to stereo audio; [Qwen-Audio-3.0-Gen](#ref-qwengen) generates sound scenes with a diffusion transformer. Neither has open weights.

**Inverse-problem solvers with a diffusion prior** are the family my plan belongs to. You start from what you observed and ask which clean signal could have produced it. A diffusion model trained on clean data proposes a clean signal step by step, and the solver keeps each step consistent with the observation through a model of the damage. [DPS](#ref-dps) introduced the idea with a known damage model, [SILO](#ref-silo) moved it to latent space, and [DAPS](#ref-daps) cuts down the over-smoothing that blurs transients like drum hits. [CQT-Diff](#ref-cqtdiff) brought it to music. [BABE](#ref-babe) and [BABE-2](#ref-babe2) use it on historical recordings without knowing the degradation. [LOUDAR](#ref-loudar) learns the unknown degradation in latent space; the authors demo it on vocal effects and guitar distortion. As far as I can tell, nobody has tried any of these on MP3, AAC or Opus.

How to read the table: *Restores* is what the method rebuilds, either the **full band** (repairing the damage inside it), just the **missing band** above the cut-off, a **round trip** (`decode(encode(x))`), or a **prior** that needs a solver. A dash means it doesn't restore anything. **Gaps** are short stretches of missing audio, and **blind** means the method isn't told what the damage is. *Weights* are **open**, **code** only, **closed** (hosted), or **none**.

| Paper                                  | Year | Method                                                    | Restores            | Targets MP3 | Weights | Tested here          |
|----------------------------------------|------|-----------------------------------------------------------|---------------------|-------------|---------|----------------------|
| [SR-GAN](#ref-srgan)                   | 2021 | stochastic GAN, complex STFT                              | full band           | ✓           | none    | —                    |
| [AudioSR](#ref-audiosr)                | 2023 | diffusion bandwidth extension, mel domain + vocoder       | missing band        | —           | open    | ✓                    |
| [Apollo](#ref-apollo)                  | 2024 | band-split GAN, waveform                                  | full band           | ✓           | open    | ✓                    |
| [A2SB](#ref-a2sb)                      | 2025 | Schrödinger bridge, vocoder-free                          | missing band + gaps | —           | open¹   | ✓                    |
| [Latent upsample/upmix](#ref-latentup) | 2025 | supervised latent-to-latent bandwidth extension and upmix | missing band        | —           | none    | —                    |
| [SonicMaster](#ref-sonicmaster)        | 2025 | text-guided flow matching, all-in-one                     | in-band faults²     | —           | open    | demo only, dropped   |
| [CoDiCodec](#ref-codicodec)            | 2025 | consistency autoencoder, continuous + discrete latents    | round trip          | —           | open    | ✓                    |
| [εar-VAE](#ref-earvae)                 | 2025 | perceptual VAE, phase + M/S losses                        | round trip          | —           | open    | ✓                    |
| [AudioLBM](#ref-audiolbm)              | 2025 | latent bridge bandwidth extension, cascaded to 192 kHz    | missing band        | —           | none    | —                    |
| [HP-codecX](#ref-hpcodec)              | 2025 | codec tokens + transformer LM                             | missing band        | —           | open    | —                    |
| [Stable Audio 3](#ref-sa3)             | 2026 | latent diffusion prior + inpainting                       | prior               | —           | open    | ✓ (SDEdit, rejected) |
| [SAME](#ref-same)                      | 2026 | transformer autoencoder, 4096×                            | round trip          | —           | open    | ✓                    |
| [Elastic Time](#ref-elastic)           | 2026 | dynamic frame-rate bottleneck                             | —                   | —           | code    | —                    |
| [Qwen-Music](#ref-qwenmusic)           | 2026 | semantic-token LLM + stereo renderer                      | —                   | —           | closed  | —                    |
| [Qwen-Audio-3.0-Gen](#ref-qwengen)     | 2026 | DiT + shared VAE, scene generation                        | —                   | —           | closed  | —                    |
| [LOUDAR](#ref-loudar)                  | 2026 | latent operator + diffusion prior                         | full band, blind    | —           | code    | —                    |
| [εar-VAE2](#ref-earvae2)               | 2026 | complex-STFT autoencoder, band refiner                    | round trip          | —           | open³   | ✓                    |

¹ Only the 1- and 2-split checkpoints are out; the 4-split ensemble from the paper isn't.
² Its nineteen faults cover EQ, dynamics, reverb, amplitude and stereo. MP3/AAC/Opus and bandwidth loss aren't on the list.
³ The released weights were retrained on public data, not the paper's internal corpus, and to my ears they sound clearly worse than εar-VAE.

So only two models go after MP3 directly, SR-GAN and Apollo, and both are supervised. Every other generative method either stops at the missing band or hasn't been tried on codec damage (outside this repo, anyway). Whether a prior that has only heard clean music can fix what a codec does *inside* the band it keeps is still an open question. That's the question I want to answer.

## Benchmark on MUSDB18-HQ

To compare with published work, I ran the three open restorers from the table (Apollo, A2SB and AudioSR) the standard way: 6 s clips from the MUSDB18-HQ test songs, encoded to MP3 at 24 to 128 kbps. The plot also includes three runs from [my own experiments](#my-experiments-so-far): the MP3 through εar-VAE, the MP3 through εar-VAE2, and Apollo's output through εar-VAE.

There are four metrics, each measuring how close a restoration is to the master in a different way. SDR and SI-SNR look at the waveform, spectral SNR and LSD at the spectrogram. Higher is better, except for LSD.

[![SDR, SI-SNR, spectral SNR and LSD per MP3 bitrate on 50 MUSDB18-HQ test clips, for six restorations and the untouched MP3](docs/decisions/assets/0016/methods.png)](docs/decisions/assets/0016/methods.png)

The waveform and spectrogram metrics don't agree. On the waveform side, A2SB and Apollo are on top: A2SB leads by about 0.5 dB at 32–64 kbps, and the two are within ~0.2 dB of each other elsewhere. AudioSR trails them by 2 to 5 dB SDR. On the spectrogram side, Apollo wins at every bitrate.

By ear it's not close. Apollo sounds far better than anything else I tried, including on a real 128 kbps rip. That makes sense, since Apollo was trained on MP3 damage while A2SB (the way I use it) only fills in the top band. You can see it on this heavily damaged 32 kbps clip: Apollo fills the top band *and* the holes below the cut-off, while A2SB just adds short bursts above it. Where Apollo put back an actual hi-hat, A2SB added a click.

[![Spectrograms of a 6 s clip: the master, its MP3 at 32 kbps, Apollo's restoration, and A2SB's restoration](docs/assets/restorations-32k.png)](docs/assets/restorations-32k.png)

Out of curiosity I also scored the untouched MP3 against the master (the grey line). Nothing beats it on the waveform metrics. On LSD, Apollo, A2SB, AudioSR and Apollo → εar-VAE all beat it at every bitrate, but on spectral SNR only Apollo clearly does, by 2.2 to 2.7 dB. Ideally a restoration should score at least as well as the MP3 it started from. If it scores worse, it may be throwing away real information while it rebuilds the signal.

## My experiments so far

Everything above either learns MP3 damage from paired data or only fills in the top band. My first experiments test the route the plan depends on: restoring with models that have only ever heard clean music. The last one teaches Stable Audio 3 the damage directly.

### SDEdit with Stable Audio 3

Stable Audio 3 is a diffusion model: it generates music by starting from pure noise and denoising it step by step. [SDEdit](#ref-sdedit), originally an image technique, uses that process to edit an existing input instead of generating from scratch. It's the simplest way to restore with a clean-only model: add noise to the MP3, let the model remove it, and hope the damage goes away with the noise.

It doesn't. To the model, music without a top band is still perfectly plausible music, so it treats the damage as part of the signal. Add more noise and it rebuilds more, but the output drifts audibly away from the original track long before the noise is strong enough to hide the damage. I've shelved SDEdit for now and will give it another go once my own prior is trained.

### Autoencoder round trips

An autoencoder compresses audio into a much smaller latent and rebuilds it from there. If it has only seen clean music, maybe it rebuilds a damaged input as clean music and loses some of the damage on the way. I tried four open music autoencoders, both on the raw MP3 and on Apollo's output.

On the MP3 alone, εar-VAE and εar-VAE2 land at about 10.5 dB and 8.2 dB SDR regardless of bitrate. On top of Apollo, εar-VAE only helped at 32 kbps, where it smooths things out. At 64 and 128 kbps it made Apollo sound worse, and all four metrics put the chain below Apollo alone at every bitrate. One caveat: my listening isn't blind yet, so I knew which version I was hearing.

### A LoRA fine-tune of Stable Audio 3

Neither SDEdit nor the round trips remove the damage, so the next step was to show Stable Audio 3 what the damage is. I fine-tuned its medium checkpoint with a LoRA on MP3/clean pairs. The MP3's latents enter through a new input in every transformer block, and the model learns to generate the clean latents from them. Like Apollo, it trains on fresh windows of the 86 MUSDB18-HQ training songs and on remixes of their stems, each coded as MP3 at a random bitrate. This first run trained for 5 hours on one GPU.

Apollo still wins. On all 3,677 test clips it scores above the LoRA on all four metrics, at every bitrate. The LoRA does fill the band the MP3 lost above the codec's cutoff, so it beats the untouched MP3 on LSD, but on the waveform metrics it stays below the MP3.

The [listening page](https://zbitouzakaria.github.io/grooveback/lora/) compares it with Apollo on all 3,677 test clips: the scores per bitrate, and the clips where each model does best and worst. It's an initial run, and I'm now iterating on data augmentation and training strategies.

Each experiment is written up in full, with the evidence, in [`docs/decisions/`](docs/decisions).

## Running the code

You'll need [uv](https://docs.astral.sh/uv/) and Python 3.12. Apollo, εar-VAE and εar-VAE2 are pulled in as git submodules:

```
git submodule update --init --recursive
uv sync --all-groups
uv run pytest
```

The MUSDB18-HQ benchmark scores Apollo on the 50 test songs at a given MP3 bitrate and writes one row per 6 s clip. The MP3 step needs sox with MP3 support (`apt install sox libsox-fmt-mp3` on Ubuntu). Without it, the tests that depend on it are skipped.

```
uv run python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps 24
```

There's also a smaller benchmark that restores one excerpt per track at 32, 64 and 128 kbps with every method and scores each result against the clean excerpt. It reads the clean tracks from `data/`, which isn't in the repo.

```
uv run python scripts/run_xp.py device=cuda
```

The library lives in `src/grooveback`, the scripts that run the models and build the listening pages are in `scripts/`, and the pages themselves are in `docs/lora/` and `docs/listen/`. A2SB and AudioSR pin dependency versions that clash with this project's, so each one gets its own environment. [`.claude/rules/environment.md`](.claude/rules/environment.md) explains how to set them up and which jobs need a rented GPU.

## References

<a id="ref-srgan"></a>[SR-GAN] Lattner & Nistal. *Stochastic Restoration of Heavily Compressed Musical Audio Using Generative Adversarial Networks.* Electronics 2021. [arXiv:2207.01667](https://arxiv.org/abs/2207.01667) · called SR-GAN in the Apollo paper

<a id="ref-audiosr"></a>[AudioSR] Liu et al. *AudioSR: Versatile Audio Super-resolution at Scale.* [arXiv:2309.07314](https://arxiv.org/abs/2309.07314) · [code](https://github.com/haoheliu/versatile_audio_super_resolution)

<a id="ref-apollo"></a>[Apollo] Li & Luo. *Apollo: Band-sequence Modeling for High-Quality Audio Restoration.* ICASSP 2025. [arXiv:2409.08514](https://arxiv.org/abs/2409.08514) · [code](https://github.com/JusperLee/Apollo)

<a id="ref-a2sb"></a>[A2SB] Kong et al. (NVIDIA). *A2SB: Audio-to-Audio Schrödinger Bridges.* [arXiv:2501.11311](https://arxiv.org/abs/2501.11311) · [code](https://github.com/NVIDIA/diffusion-audio-restoration)

<a id="ref-latentup"></a>[Latent upsample/upmix] Bralios, Smaragdis, Casebeer. *Learning to Upsample and Upmix Audio in the Latent Domain.* WASPAA 2025. [arXiv:2506.00681](https://arxiv.org/abs/2506.00681)

<a id="ref-sonicmaster"></a>[SonicMaster] Melechovsky et al. *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering.* ICML 2026. [arXiv:2508.03448](https://arxiv.org/abs/2508.03448) · [weights](https://huggingface.co/amaai-lab/SonicMaster)

<a id="ref-codicodec"></a>[CoDiCodec] Pasini, Lattner, Fazekas. *CoDiCodec: Unifying Continuous and Discrete Compressed Representations of Audio.* ISMIR 2025. [arXiv:2509.09836](https://arxiv.org/abs/2509.09836) · [code](https://github.com/SonyCSLParis/codicodec) · [weights](https://huggingface.co/SonyCSLParis/codicodec)

<a id="ref-earvae"></a>[εar-VAE] Wang et al. *Back to Ear: Perceptually Driven High Fidelity Music Reconstruction.* [arXiv:2509.14912](https://arxiv.org/abs/2509.14912) · [code](https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE) · [weights](https://huggingface.co/earlab/EAR_VAE)

<a id="ref-audiolbm"></a>[AudioLBM] Li et al. *Audio Super-Resolution with Latent Bridge Models.* NeurIPS 2025. [arXiv:2509.17609](https://arxiv.org/abs/2509.17609)

<a id="ref-hpcodec"></a>[HP-codecX] Giniès et al. *Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth Extension.* [arXiv:2511.21580](https://arxiv.org/abs/2511.21580) · [code](https://zenodo.org/records/22144166) · [weights](https://zenodo.org/records/22144249)

<a id="ref-sa3"></a>[Stable Audio 3] Evans, Parker et al. (Stability AI). *Stable Audio 3.* [arXiv:2605.17991](https://arxiv.org/abs/2605.17991) · [code](https://github.com/Stability-AI/stable-audio-3) · [weights](https://huggingface.co/stabilityai/stable-audio-3-small-music)

<a id="ref-same"></a>[SAME] Parker, Evans et al. *SAME: A Semantically-Aligned Music Autoencoder.* [arXiv:2605.18613](https://arxiv.org/abs/2605.18613) · [weights](https://huggingface.co/stabilityai/SAME-L)

<a id="ref-elastic"></a>[Elastic Time] Bralios, Smaragdis, Kim. *Elastic Time: Dynamic Frame Rate Bottlenecks for Neural Audio Coding.* Interspeech 2026. [arXiv:2606.27320](https://arxiv.org/abs/2606.27320) · [code](https://github.com/dbralios/elastic-time)

<a id="ref-qwenmusic"></a>[Qwen-Music] Xu et al. (Alibaba). *Qwen-Music Technical Report.* [arXiv:2607.11699](https://arxiv.org/abs/2607.11699)

<a id="ref-qwengen"></a>[Qwen-Audio-3.0-Gen] Dai et al. (Alibaba). *Qwen-Audio-3.0-Gen-Preview Technical Report.* [arXiv:2607.27011](https://arxiv.org/abs/2607.27011)

<a id="ref-loudar"></a>[LOUDAR] Švento et al. *Music Restoration via Latent Operator Optimization and Diffusion Model Priors.* ISMIR 2026. [arXiv:2608.01972](https://arxiv.org/abs/2608.01972) · [code](https://github.com/michalsvento/loudar) · [demo](https://michalsvento.github.io/loudar/)

<a id="ref-earvae2"></a>[εar-VAE2] Wang, Dai, Xu. *Fourier is Frontier: Frequency-Aware Autoencoding for High-Fidelity Music Reconstruction.* [arXiv:2608.19843](https://arxiv.org/abs/2608.19843) · [code](https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE2) · [weights](https://huggingface.co/earlab/EAR_VAE2) · [demo](https://eps-acoustic-revolution-lab.github.io/EAR_VAE2/)

<a id="ref-sdedit"></a>[SDEdit] Meng et al. *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations.* ICLR 2022. [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)

<a id="ref-dps"></a>[DPS] Chung et al. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

<a id="ref-silo"></a>[SILO] Raphaeli, Man, Elad. *SILO: Solving Inverse Problems with Latent Operators.* ICCV 2025. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)

<a id="ref-daps"></a>[DAPS] Zhang et al. *Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing.* CVPR 2025. [arXiv:2407.01521](https://arxiv.org/abs/2407.01521)

<a id="ref-cqtdiff"></a>[CQT-Diff] Moliner, Lehtinen, Välimäki. *Solving Audio Inverse Problems with a Diffusion Model.* [arXiv:2210.15228](https://arxiv.org/abs/2210.15228)

<a id="ref-babe"></a>[BABE] Moliner et al. *Blind Audio Bandwidth Extension: A Diffusion-Based Zero-Shot Approach.* [arXiv:2306.01433](https://arxiv.org/abs/2306.01433)

<a id="ref-babe2"></a>[BABE-2] Moliner et al. *A Diffusion-Based Generative Equalizer for Music Restoration.* DAFx-24. [arXiv:2403.18636](https://arxiv.org/abs/2403.18636)

<a id="ref-ytformats"></a>[YT-Formats] Eesmaa, M. *YouTube Audio Formats — itags, codecs and bitrates.* [gist](https://gist.github.com/MartinEesmaa/2f4b261cb90a47e9c41ba115a011a4aa)

---

The code is MIT-licensed.
