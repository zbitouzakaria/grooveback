# grooveback

**Music super-resolution for rare records that only survived as degraded rips.**

As a DJ and listener, I keep running into the same problem: records I love that only exist, now, as 128 kbps MP3s ripped from YouTube — themselves sourced from someone's imperfect vinyl transfer. On laptop speakers they're fine. On a decent system they fall apart: smeared transients, no air in the top end, a generally flat and tired sound that makes everything around them in the library sound better by contrast.

grooveback is a learning-based system that takes these degraded tracks and produces a higher-fidelity version that holds up on real playback. It is focused, for now, on the music I actually listen to: old-school house, minimal, and adjacent electronic genres.

## What it does

Given a degraded input — typically a low-bitrate MP3 that has also been through YouTube's re-encoding pipeline, often originally sourced from vinyl — grooveback produces a restored output intended for high-quality playback. The model is trained on clean electronic music passed through a synthetic degradation pipeline that mimics the vinyl → poor recording setup → YouTube → yt-dlp chain.

## What it does not do

grooveback is not strict signal recovery. The degradations involved discard information irreversibly: MP3 psychoacoustic masking, bandwidth cuts from resampling, noise and coloration from analog transfers. No system can recover what is genuinely gone.

Instead, grooveback produces a *plausible* high-fidelity reconstruction — structurally coherent with the input, perceptually convincing on good playback systems, but not a guarantee of faithfulness to the original master. This is the same trade-off made by modern real-world image restoration systems like Real-ESRGAN and DiffBIR: recover what's recoverable, hallucinate what isn't, prioritize perceptual quality over measured fidelity.

For the intended use case — listening on a home system to records you can't re-source — this trade-off is the right one.

## Approach

The project is built incrementally.

The first real work is the degradation pipeline, not the model. grooveback learns from clean electronic music passed through a synthetic degradation chain designed to match the actual YouTube-vinyl-rip distribution. This pipeline — MP3 encoding at realistic bitrates, resampling artifacts, vinyl-style surface noise and wow/flutter, mild coloration, YouTube-style re-encoding — is validated by ear against real rips of tracks I also have in clean form. The quality of the degradation pipeline bounds the quality of everything downstream.

The model itself starts in the time-frequency domain with a spectrogram-based U-Net baseline, using multi-resolution STFT loss and conservative regression training. Later iterations move toward generative priors (adversarial training, or latent-space diffusion/flow matching in the style of LatentFlowSR and Latent Bridge Models) to address the characteristic muffled output of pure regression models on heavily-degraded sources.

Evaluation combines standard signal metrics (LSD, ViSQOL, FAD) with informal listening tests on a curated evaluation set of tracks from my own library, played back on monitoring hardware. The listening tests matter more than the metrics — this is ultimately a playback problem, and the final judge is what it sounds like on a real system.

## Research context

grooveback sits between audio super-resolution, music restoration, and learned-prior approaches to inverse problems.

### Audio and music restoration

- AudioSR — general audio super-resolution; explicitly not trained for MP3 losses. [arXiv:2309.07314](https://arxiv.org/abs/2309.07314), [GitHub](https://github.com/haoheliu/versatile_audio_super_resolution)
- FlashSR — single-step diffusion-distilled audio super-resolution. [arXiv:2501.10807](https://arxiv.org/abs/2501.10807)
- CQT-Diff — diffusion-based bandwidth extension using Constant-Q representations, music-focused. [arXiv:2210.15228](https://arxiv.org/abs/2210.15228)
- LatentFlowSR — conditional flow matching in a latent space for audio super-resolution. [arXiv:2604.09188](https://arxiv.org/abs/2604.09188)
- Latent Bridge Models for audio SR — follow-up work in the latent-domain generative direction. [arXiv:2509.17609](https://arxiv.org/abs/2509.17609)
- BABE / BABE-2 — blind bandwidth extension and generative equalization for music restoration, including historical recordings. [arXiv:2306.01433](https://arxiv.org/abs/2306.01433)
- Apollo — band-sequence modeling for restoring MP3-compressed music. [GitHub](https://github.com/JusperLee/Apollo)

### Real-world image restoration (conceptual precedents)

The strongest conceptual influence comes from real-world image super-resolution and blind restoration, where the field moved from strict reconstruction toward perceptually convincing restoration guided by learned priors:

- Real-ESRGAN — practical real-world image restoration trained with synthetic degradation pipelines. [ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html), [GitHub](https://github.com/xinntao/Real-ESRGAN)
- StableSR — diffusion-prior-based super-resolution. [arXiv:2305.07015](https://arxiv.org/abs/2305.07015), [GitHub](https://github.com/IceClear/StableSR)
- DiffBIR — blind image restoration combining degradation removal and generative reconstruction. [GitHub](https://github.com/XPixelGroup/DiffBIR)
- SUPIR — photo-realistic restoration in the wild. [GitHub](https://github.com/Fanghua-Yu/SUPIR)
- OSEDiff — one-step effective diffusion for real-world image super-resolution. [GitHub](https://github.com/cswry/OSEDiff)

Real-ESRGAN in particular is a mental model for grooveback: the degradation pipeline is the central research artifact, and the model is trained to invert a realistic chain of degradations rather than a single clean operator.

## Scope and status

Early development. The project is currently focused on:

- Building a degradation pipeline that realistically models the vinyl → YouTube → rip chain
- Curating a personal evaluation set of electronic music tracks at both clean and degraded quality
- Training a spectrogram-domain baseline as a foundation to iterate from

Planned but not committed:

- Generative priors (adversarial and/or diffusion/flow-matching) to move past the muffled-regression ceiling
- Chunked inference for arbitrarily long audio
- A desktop tool for batch-processing personal libraries

## A note on genre focus

grooveback is trained and evaluated with a bias toward old-school house, minimal, and adjacent electronic music. This is deliberate. A model trained on a narrower distribution outperforms a general model on that distribution, and the use case — restoring tracks I and other listeners in this genre space actually play — rewards specialization. Broadening the training distribution to other genres is possible but not currently a priority.
