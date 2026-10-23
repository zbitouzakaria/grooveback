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

Sound is a mix of frequencies: low ones for the bass and the kick drum, high ones for hi-hats and cymbals. People hear
from about 20 Hz to 20 kHz, that is, from 20 to 20,000 vibrations per second. A CD stores the sound wave as a long list
of numbers, 44,100 per second for each of the two stereo channels; each number is a sample. That makes 1,411 kilobits of
data per second (1,411 kbps). An MP3 at 128 kbps stores about a tenth of that, and one at 32 kbps less than a fortieth.

To fit, the encoder, the program that compresses the audio, drops what the ear is least likely to miss, much as JPEG
does for photos. It removes every frequency above a cut-off, the top band. It stores the rest with less precision, which
blurs short, sharp sounds such as a drum hit, and when it runs short of data it drops quiet parts entirely. This is
lossy compression: decoding the file cannot bring back what was dropped. MP3, AAC and Opus all drop what the ear is
least likely to miss; each is a codec, short for coder-decoder.

The picture below shows this on 6 s of an electronic track, Aerofunk's "Nice One". It is a spectrogram: time runs left
to right, frequency runs bottom to top, and brighter means louder. Level is in decibels (dB), a scale for comparing two
quantities, where every 10 dB means ten times more energy. In this picture, 0 dB is the loudest a file can hold, so real
levels are negative. Each vertical stripe is a drum hit. The master, the clean version of the track as released, reaches
20 kHz. Its MP3 at 128 kbps keeps nothing above about 16.5 kHz. At 32 kbps it keeps nothing above 5.5 kHz, and dark
holes, the dropped quiet parts, appear below the cut-off too.

![Spectrograms of 6 s of an electronic track: the master, its MP3 at 128 kbps, and its MP3 at 32 kbps](docs/assets/mp3-spectrograms.png)

At 128 kbps, the lost precision below the cut-off is too fine to see in this picture. As we show below, it is still most
of the damage.

A rip from YouTube was compressed at least twice: once by the person who uploaded it, and again by YouTube, which serves
its standard streams as roughly 128 kbps AAC or Opus ([YouTube audio formats](#ref-ytformats)).

## A restorer has to invent what the codec removed

Because the information is gone, restoring a rip cannot mean recovering the master. A restorer has to invent the missing
sound. Its output can be plausible, consistent with the rip and convincing on good speakers, but not faithful: it will
not match the master exactly.

That makes restoration a job for a generative model, a neural network trained on many examples until it can produce new
ones of the same kind. Our plan is a generative model of clean house and minimal music, made by training [Stable Audio
3](#ref-sa3), a public music generator, further on my library of about 215 hours. We call this model the prior: it knows
only what clean music sounds like. A second program, the solver, runs the prior and steers what it generates toward a
clean track that matches the rip. Only the solver knows about the damage, so a new kind of damage, such as vinyl
crackle, means changing the solver, not retraining the prior
([ADR-0004](docs/decisions/0004-restoration-as-a-latent-inverse-problem.md)).

We have not trained the prior yet. First we measured what the existing tools do, and that is what the rest of this page
reports. Each finding links to its decision record (ADR), a short write-up of the experiment and its evidence, in
[`docs/decisions/`](docs/decisions/).

## What we found

### Apollo beats A2SB on heavy damage, but neither changed how our 128 kbps rip sounds

We started with two models that anyone can download and run ([ADR-0006](docs/decisions/0006-baseline-findings.md)):
[Apollo](#ref-apollo), a neural network trained to repair MP3-compressed music, and [A2SB](#ref-a2sb), which fills in
the top band above a cut-off. We tried them on two inputs: Apollo's own 6 s test clip, a heavily damaged copy whose
sound stops at about 5 kHz, like a 32 kbps MP3; and one real 128 kbps rip (AN-2, "Moonshine", 7.6 min). We listened on
studio headphones, with every track at the same loudness, because a louder track tends to sound better.

On the 6 s clip, Apollo won clearly: it put back a full hi-hat where A2SB added a click. On the 128 kbps rip, both
models filled the band above its cut-off (15.75 kHz here; the exact cut-off depends on the encoder), yet the MP3, Apollo
and A2SB all sounded the same on headphones. We have not yet compared them on speakers, where the damage is easiest to
hear. Apollo also runs faster than real time on a laptop and keeps left and right separate. A2SB works in mono only, and
took about 35 minutes for the 7.6-minute track on an A100 GPU.

We also have a clean version of Apollo's test clip. The picture below shows the same contrast on our own 32 kbps MP3 of
it, from our listening page, rather than on the damaged copy we listened to. Apollo fills the top band and the holes
below the cut-off. A2SB adds short bursts above the cut-off and leaves the holes.

![Spectrograms of a 6 s clip: the master, its MP3 at 32 kbps, Apollo's restoration, and A2SB's restoration](docs/assets/restorations-32k.png)

We also tried [SonicMaster](#ref-sonicmaster), which takes a typed instruction, through its online demo, and dropped it:
it adds extra kick drums on electronic music, even when the instruction says not to
([ADR-0005](docs/decisions/0005-baselines-and-prior-viability-on-real-library-material.md)).

### At 64 kbps and above, most of the damage is inside the band the MP3 keeps

For real rips we rarely have the master, and when we do, the rip starts at a different moment and often comes from a
different master. So we made our own test material: MP3 copies of clean tracks at known bitrates, scored against the
master ([ADR-0007](docs/decisions/0007-benchmark-codec-restoration.md), extended in
[ADR-0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md)). Each copy is encoded
once. A YouTube rip is encoded at least twice, so these results describe the easier case.

The score here is SDR (signal-to-distortion ratio): how much stronger the master is than the track's difference from it,
sample by sample, in dB. Higher is better, and every 10 dB means ten times less error energy. On a 12 s excerpt of "Nice
One", the 64 kbps MP3 keeps nothing above 11 kHz. The master with only that band removed still scores 27 dB against the
full master, so if the missing band were the only damage, the MP3 would score about 27 dB too. It scores 17.6 dB: the
missing band holds only about a tenth of the MP3's error energy, and the other nine tenths are inside the band it keeps.
At 128 and 192 kbps the share inside the kept band is larger still. So a restorer that only fills the top band, like
A2SB, cannot gain much at these bitrates. At 32 kbps the cut-off falls to 5.5 kHz and the missing band matters more; we
did not measure that case.

### No method gets closer to the master than the MP3 on waveform scores

There are two families of scores. Waveform scores, such as SDR, compare the restored audio with the master sample by
sample. Spectrum scores, such as the log-spectral distance (LSD, lower is better), compare only how much energy each
frequency has over time, as a spectrogram shows it. A restorer invents detail, and invented detail never lines up
exactly with the master, so waveform scores punish it. As expected, no method got closer to the master than the
untouched MP3 on the waveform scores, on any of our benchmark tracks at any bitrate, while the spectrum scores, LSD
above all, did reward restoration. A score can tell us that something broke, but not that something sounds better. Only
listening can judge that.

### MP3 removes about as much from Stable Audio 3's clips as from a master

Our plan depends on Stable Audio 3, so we checked whether it generates the fine detail a real master has
([ADR-0008](docs/decisions/0008-sa3-generation-probe.md)). Stable Audio 3 comes in a small and a medium model, each in a
fast and a slow version. A generated clip has no master to compare with, so we compressed each clip to MP3 and scored
the MP3 against the clip. A clip without fine detail would lose little and score high, so here a lower score means more
detail. At 64 kbps, clips from the four music versions score 10.3 to 21.3 dB, on both sides of the 17.6 dB that the
master excerpt's MP3 scores. So the clips hold detail for an MP3 to remove. This is one prompt and three random seeds:
it does not show that they sound like a master.

### SDEdit keeps the damage it should remove

Stable Audio 3 generates music by removing noise in many small steps, starting from pure noise. Our first method,
[SDEdit](#ref-sdedit) on the medium model, uses that: it adds some noise to the MP3 and lets the model remove it, in the
hope that the codec damage goes with it ([ADR-0009](docs/decisions/0009-sdedit-as-the-first-solver.md)). It did not. To
the model, music with its top band missing is still plausible music, so it keeps it. No setting filled the band above
the cut-off. When we filled that band with noise first, the model kept the noise instead of turning it into music. A
second variant added no noise, but told the model the MP3 was already partly noise, so that it would remove the damage
as if it were noise. It kept the damage too.

The numbers show why. Stable Audio 3 works on a compressed summary of the audio, its latent space, rather than on the
samples. In that space, the MP3 damage is as large as noise at level 0.54 to 0.80, depending on the track and bitrate,
on a scale from 0 (no noise) to 1 (pure noise). To remove the damage, SDEdit must add at least that much noise, but
above 0.25 the model starts to change the music itself, not just the damage. We rejected SDEdit.

### Running εar-VAE after Apollo helps on heavy damage, by ear

An autoencoder is the pair of networks that makes such a latent space: an encoder compresses audio into a much shorter
list of numbers, the latents, and a decoder turns them back into audio. One trained on clean music might return a
cleaner track from a round trip through both. We tested four open music autoencoders: [SAME-L](#ref-same) (used by
Stable Audio 3's medium model), [εar-VAE](#ref-earvae), [εar-VAE2](#ref-earvae2) and [CoDiCodec](#ref-codicodec)
([ADR-0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md)).

We tried three uses at 32, 64 and 128 kbps, on the clean version of Apollo's test clip and on 3-minute excerpts of two
other tracks, one of them "Nice One". The first is a round trip of the MP3. The second is a round trip of Apollo's
output, written Apollo → εar-VAE when the autoencoder is εar-VAE. The third, damage subtraction, works like a correction
EQ: on a track whose master we have, we measure how MP3 shifts the latents on average over time, then shift the latents
of the MP3 we want to restore back by the same amount. We measure the shift on a different track because a real rip has
no master.

By ear, two of these were worth keeping: Apollo → εar-VAE and the εar-VAE damage subtraction. The rest, every plain
round trip included, sounded worse than these two. We have not ranked the damage subtraction against Apollo alone.
Apollo → εar-VAE beats Apollo alone at 32 kbps, ties with it at 64 kbps, and loses at 128 kbps, where Apollo alone
sounds closer to the master. To our ears, εar-VAE's decoder smooths Apollo's output: at 32 kbps this removes flaws
Apollo adds, and at 128 kbps, where Apollo adds few, it moves the sound away from the master. We have not measured this.

At 32 kbps the scores disagree with the ear: they rate Apollo → εar-VAE below Apollo alone, for example an LSD of 11.2
against 10.4 dB (lower is better) on "Nice One". We trust the ear here, because a score can show that something broke,
not that something sounds better. The damage subtraction improves the spectrum score almost everywhere, but not the
waveform score. We measured its shift on one track only, and on another artist's track at 128 kbps it overcorrects. We
knew which track was which while listening; a blind test, which hides the names, has not been done.

### On public test songs too, the scores rate Apollo → εar-VAE below Apollo

To compare with other work, we scored methods the way Apollo's authors did, on 6 s clips with MP3 at 24 to 128 kbps
([ADR-0015](docs/decisions/0015-reproduce-apollo-test-protocol.md)). Their scores are two waveform scores, SDR and
SI-SNR (a version of SDR that ignores overall volume); we added two spectrum scores, LSD and spectral SNR (higher is
better). We used MUSDB18-HQ, a public research dataset of 150 songs, 50 of which are set aside for testing; according to
[Apollo's maintainer](https://github.com/JusperLee/Apollo/issues/46#issuecomment-6281774888), the released model was not
trained on those 50. On these songs, Apollo → εar-VAE scores worse than Apollo on all four scores at every bitrate, and
its scores barely change with bitrate. These are not house and minimal tracks, and we did not listen to them.

![SDR, SI-SNR, spectral SNR and LSD per MP3 bitrate on the 50 MUSDB18-HQ test songs](docs/decisions/assets/0015/bitrates.png)

The dotted line is Apollo's result as published, on a different test set, and is not comparable with the solid lines.
With the authors' code as released, their evaluation script and ours give the same average scores on the same 3,677
clips at 24 kbps. We also found and fixed a bug in their MP3 simulation: it left 2.5% of the 24 kbps clips 25 ms late,
which let a method gain score just by shifting the MP3 back into place. The figure uses the fixed version.

## Where this leaves us

The case that started this project is still open. Our gains are at 32 kbps, far below a typical rip: there, by ear,
Apollo → εar-VAE is our best restorer. At 128 kbps, Apollo alone sounds closer to the master than Apollo → εar-VAE on
our MP3 copies, but whether any method makes a 128 kbps MP3 sound better than it already does is still open: on the one
real rip we tried, neither Apollo nor A2SB made an audible difference on headphones. The scores rate Apollo alone above
Apollo → εar-VAE at every bitrate.

What comes next:

- **The prior.** Train Stable Audio 3 further on the 215-hour library
  ([ADR-0004](docs/decisions/0004-restoration-as-a-latent-inverse-problem.md)). It generates in the latent space of
  SAME, so everything it makes passes through SAME's decoder, and a plain round trip through SAME-L was not among the
  methods we kept by ear. That does not rule the plan out: in a round trip, SAME's encoder chooses the latents, while
  the prior will choose its own, and they can sound better through the same decoder.
- **A solver that stays close to the MP3.** SDEdit let the model drift from the input. The next solver checks, at each
  step, that the output still matches the MP3, in the line of [DPS](#ref-dps), [DAPS](#ref-daps) and
  [LOUDAR](#ref-loudar). Stable Audio 3's slow version, which removes noise in 50 steps, is easier to steer this way
  than the fast one, which takes 8, but we prefer the fast one by ear, so the first choice is between them.
- **The same evaluation.** Every new method is scored on MUSDB18-HQ and heard on the listening pages, against Apollo and
  Apollo → εar-VAE.

## How we listen and measure

Every listening set is first brought to the same loudness, −14 LUFS on a standard loudness scale, then, if needed,
lowered by one shared amount so that no peak goes above 1 dB below the loudest level a file can hold. We listen in
browser pages built on [trackswitch.js](https://github.com/audiolabs/trackswitch.js), which switch between tracks
without restarting playback ([ADR-0010](docs/decisions/0010-listen-in-the-browser-with-trackswitch.md)). Scores are
measured against the master. Tests catch bugs that produce worse audio without an error message: a stand-in model that
returns its input unchanged must come out of the full pipeline unchanged, and every evaluation also scores the untouched
MP3.

## Running it

```bash
git clone --recurse-submodules https://github.com/zbitouzakaria/grooveback.git
cd grooveback
uv sync --all-groups   # Python 3.12 or later; stable-audio-3 pins torch to 2.7.1
uv run pytest
```

- `uv run python -m grooveback.cli.baseline INPUT OUTPUT --method apollo` restores one 44.1 kHz file with Apollo, or
  with A2SB (`--method a2sb --device cuda`).
- `uv run python -m grooveback.cli.musdb_eval musdb18hq/test scores.csv --kbps 24 --earvae` scores one bitrate of the
  MUSDB18-HQ protocol; download the dataset to `musdb18hq/` first. All six bitrates take about an hour on one H100.
- `uv run python scripts/run_xp.py device=cuda` runs our MP3 benchmark. Its source tracks live in `data/`, which is not
  in the repository.
- `uv run python -m grooveback.cli.run model=roundtrip input=INPUT output_dir=DIR` runs a solver through
  [Hydra](https://hydra.cc): `roundtrip`, `sdedit`, or `latent-sub`, which needs a damage vector from `run_xp.py`.
- After `run_xp.py`, `uv run --group notebooks python scripts/publish_listen.py` rebuilds the listening page in
  `docs/listen/` from its renders, replacing the old page.

A2SB runs from its own clone and environment, because its dependencies conflict with ours:
`git clone -b runnable-anywhere https://github.com/zbitouzakaria/diffusion-audio-restoration.git third_party/a2sb`,
then `./setup.sh` inside it. The Stable Audio 3 checkpoints are gated on Hugging Face: accept their license and export
`HF_TOKEN`. The MP3 benchmark needs ffmpeg with libmp3lame, and the MUSDB18-HQ protocol needs sox with MP3 support.

## Related work

The methods differ in what damage they repair and where they work: on the waveform, on spectrograms, or in the latent
space of a neural autoencoder. Most rebuild the band above a cut-off, or fix faults like reverb and clipping. Few
target what MP3, AAC and Opus do inside the band they keep.

- **Supervised restorers** learn from pairs of damaged and clean audio. [Apollo](#ref-apollo) is the only one trained
  on MP3 damage. [SonicMaster](#ref-sonicmaster) does not cover codec damage.
- **Bandwidth extension** rebuilds the band above a cut-off: [AudioSR](#ref-audiosr), [A2SB](#ref-a2sb),
  [AudioLBM](#ref-audiolbm), [HP-codecX](#ref-hpcodec) and [latent upsampling](#ref-latentup). None of them trains on
  codec damage, and AudioSR copies the input's band below the cut-off into its output unchanged.
- **Music autoencoders** give generative models their latent space. [SAME](#ref-same), in a large L and a small S, both
  open, is the space of Stable Audio 3. [εar-VAE](#ref-earvae) and [εar-VAE2](#ref-earvae2) are built to keep phase,
  stereo image and high harmonics. [CoDiCodec](#ref-codicodec) gives continuous and discrete latents from one model,
  and [Elastic Time](#ref-elastic) spends latent frames where the signal is dense. A method that works in a latent space
  cannot sound better than the decoder.
- **Generative music models** a restoration prior could start from: [Stable Audio 3](#ref-sa3), whose small and medium
  checkpoints are open, and [Qwen-Music](#ref-qwenmusic) and [Qwen-Audio-3.0-Gen](#ref-qwengen), which have no open
  weights.
- **Inverse problems with a diffusion prior.** The prior proposes clean audio, and the sampler keeps it consistent with
  the observed signal. [DPS](#ref-dps) introduced this with a known damage model, [SILO](#ref-silo) moved it into a
  latent space, and [DAPS](#ref-daps) reduces the over-smoothing that hides transients. [CQT-Diff](#ref-cqtdiff)
  brought it to music, [BABE](#ref-babe) and [BABE-2](#ref-babe2) apply it to historical recordings without knowing the
  damage, and [LOUDAR](#ref-loudar) learns the unknown damage in a latent space, shown on vocal effects and guitar
  distortion. We found none tried on MP3, AAC or Opus damage.

*Restores* says what a method reconstructs: the **full band** including in-band damage, the **missing band** above a
cut-off, a **round trip** (an autoencoder's `decode(encode(x))`), or a **prior** that needs a solver. A dash means it
does not restore. *Weights* are **open**, **code** only, **closed** (hosted), or **none**.

| Paper | Year | Method | Restores | Targets MP3 | Weights | Tested here |
|---|---|---|---|---|---|---|
| [AudioSR](#ref-audiosr) | 2023 | diffusion SR, mel domain + vocoder | missing band | — | open | — |
| [Apollo](#ref-apollo) | 2024 | band-split GAN, waveform | full band | ✓ | open | ✓ [0006](docs/decisions/0006-baseline-findings.md), [0007](docs/decisions/0007-benchmark-codec-restoration.md), [0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md), [0015](docs/decisions/0015-reproduce-apollo-test-protocol.md) |
| [A2SB](#ref-a2sb) | 2025 | Schrödinger bridge, vocoder-free | missing band + gaps | — | open¹ | ✓ [0006](docs/decisions/0006-baseline-findings.md), [0007](docs/decisions/0007-benchmark-codec-restoration.md), [0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md) |
| [Latent upsample/upmix](#ref-latentup) | 2025 | supervised latent-to-latent BWE/upmix | missing band | — | none | — |
| [SonicMaster](#ref-sonicmaster) | 2025 | text-guided flow matching, all-in-one | in-band faults² | — | open | demo only, dropped ([0005](docs/decisions/0005-baselines-and-prior-viability-on-real-library-material.md)) |
| [CoDiCodec](#ref-codicodec) | 2025 | consistency autoencoder, continuous + discrete latents | round trip | — | open | ✓ [0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md) |
| [εar-VAE](#ref-earvae) | 2025 | perceptual VAE, phase + M/S losses | round trip | — | open | ✓ [0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md), [0015](docs/decisions/0015-reproduce-apollo-test-protocol.md) |
| [AudioLBM](#ref-audiolbm) | 2025 | latent bridge SR, cascaded to 192 kHz | missing band | — | none | — |
| [HP-codecX](#ref-hpcodec) | 2025 | codec tokens + transformer LM | missing band | — | open | — |
| [Stable Audio 3](#ref-sa3) | 2026 | latent diffusion prior + inpainting | prior | — | open | ✓ [0008](docs/decisions/0008-sa3-generation-probe.md), [0009](docs/decisions/0009-sdedit-as-the-first-solver.md) (SDEdit, rejected) |
| [SAME](#ref-same) | 2026 | transformer autoencoder, 4096× | round trip | — | open | ✓ [0007](docs/decisions/0007-benchmark-codec-restoration.md), [0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md) |
| [Elastic Time](#ref-elastic) | 2026 | dynamic frame-rate bottleneck | — | — | code | — |
| [Qwen-Music](#ref-qwenmusic) | 2026 | semantic-token LLM + stereo renderer | — | — | closed | — |
| [Qwen-Audio-3.0-Gen](#ref-qwengen) | 2026 | DiT + shared VAE, scene generation | — | — | closed | — |
| [LOUDAR](#ref-loudar) | 2026 | latent operator + diffusion prior | full band, blind | — | code | — |
| [εar-VAE2](#ref-earvae2) | 2026 | complex-STFT autoencoder, band refiner | round trip | — | open³ | ✓ [0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md) |

¹ Only the 1- and 2-split checkpoints are released; the paper's 4-split ensemble is not.
² Its nineteen faults cover EQ, dynamics, reverb, amplitude and stereo; MP3/AAC/Opus and bandwidth damage are not
among them.
³ The open weights are retrained on public datasets, not the paper's internal corpus. By ear they sound clearly worse
than εar-VAE ([ADR-0011](docs/decisions/0011-benchmark-autoencoder-roundtrips-and-mean-damage-subtraction.md)).

Only one model, Apollo, targets MP3 directly, and it is a supervised model. Outside this project, every generative
method either stops at the missing band or has not been tried on MP3, AAC or Opus. Repairing in-band codec damage with
a generative model is still open.

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

<a id="ref-codicodec"></a>
[CoDiCodec] Pasini, Lattner, Fazekas. *CoDiCodec: Unifying Continuous and Discrete Compressed Representations of Audio.* ISMIR 2025. [arXiv:2509.09836](https://arxiv.org/abs/2509.09836) · [code](https://github.com/SonyCSLParis/codicodec) · [weights](https://huggingface.co/SonyCSLParis/codicodec)

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

The code here is MIT-licensed. The third-party code it runs has its own licenses: Apollo CC BY-SA 4.0, A2SB the NVIDIA
Source Code License (non-commercial), εar-VAE and εar-VAE2 Apache-2.0, CoDiCodec CC BY-NC 4.0, and the vendored
trackswitch.js player MIT. The Stable Audio 3 checkpoints are released under the
[Stability AI Community License](https://stability.ai/license), which any weights adapted from them would inherit; it
permits the personal and research use this project is scoped to. The audio on the listening page belongs to its owners
and is not covered by the MIT license.
