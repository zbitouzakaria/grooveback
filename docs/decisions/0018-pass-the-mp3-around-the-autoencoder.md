# 18. Pass the MP3 around the autoencoder with decoder skips

Date: 2026-11-03

## Status

Accepted.

## The autoencoder caps the LoRA below its own input

The LoRA restores in medium-base's latent space, so medium-base's autoencoder, SAME-L, decodes its output. The best the
LoRA can do is the master's own round trip through that autoencoder, and that round trip scores below the MP3 the LoRA
starts from. On the overfit test's 40 training windows at 32 kbps, the round trip scores a median 9.8 dB SDR against
the master, and the MP3 13.6 dB. A LoRA trained to overfit those windows reached 9.8 dB: the ceiling, 3.8 dB under
its input. A better LoRA cannot lift that ceiling, because the loss happens in the decoder.

We want to reach Apollo. On ADR-0016's 50 test clips at 32 kbps, Apollo scores 12.1 dB SDR, against the MP3's 12.9 dB.
It leads on the spectral scores: a spectral SNR of 15.8 dB against the MP3's 13.5 dB, and an LSD of 9.0 dB against
32.3 dB.

## What others did

[SkipDACVAE](https://arxiv.org/abs/2609.29169) gives a frozen audio autoencoder one zero-initialised 1×1 convolution
per encoder level. Each carries the degraded audio's encoder features into the decoder at the same resolution. Only
those convolutions train: the clean audio is encoded, then decoded with skips from its degraded copy, and compared
with the clean audio. In one ablation, their diffusion model uses the new decoder without retraining, since the latent
space does not change. They report distribution scores only, no SDR.

Training a multi-step model on its own loss plus a loss on its decoded output is common in audio.
[BridgeCodec](https://arxiv.org/abs/2610.15453) decodes its estimate through a frozen decoder and adds waveform and
mel losses in a second, 10k-step stage; their gradient reaches the generator through that decoder.
[BareWave](https://arxiv.org/abs/2606.09048) weights its STFT loss toward low noise, and the weight raises UTMOS from
3.88 to 3.97 over the same loss unweighted. One result is a warning for our metrics: in
[Richter et al.](https://arxiv.org/abs/2409.10753), a PESQ loss raised PESQ from 2.91 to 3.70 while SI-SDR fell from
19.4 to 8.3 dB. We found no one who trains decoder skips inside a multi-step fine-tune.

## Inference: the decoder also reads the MP3

```
MP3 ──► encoder ──► MP3 latents ──► DiT + LoRA, 50 Euler steps ──► restored latents ──► decoder ──► restored audio
 │                                                                                         ▲
 └──────────► 256-sample patches, and the encoder's first layer ──────► two skips ─────────┘
```

The sampler does not change. Only the last stage does: the decoder rebuilds the audio from the restored latents and,
through two skips, from the MP3 itself. Where the MP3 still holds the music, below its cut-off, the decoder no longer
has to rebuild it from 256 numbers per 4,096 stereo samples.

SAME-L cuts the audio into patches of 256 samples and turns every 16 patches into one latent frame. Its decoder does
the reverse: each latent frame gets 16 learned query tokens, and 12 transformer layers turn them into 16 patches. The
two skips act at that patch level, the decoder's outermost:

- one adds a learned linear map of the MP3's patches to the decoder's output patches;
- one adds a learned linear map of the encoder's first layer, run on the MP3, to the decoder's query tokens.

Both maps start at zero, so before training the decoder is unchanged and the model restores as checkpoint-33135
does. They add 2.6 M weights next to the decoder's 426 M, which stay frozen.

## Training: a second pass at the sampler's last noise level

We continue the 20 h run from checkpoint-33135. Every step makes two passes over the same batch:

1. The flow loss at the usual noise levels, as today. It trains the LoRA.
2. The same batch at t = 0.138, the noise level of the sampler's last step. The model's estimate of the master's
   latents, `noised - t * velocity`, is decoded with the skips, and an audio loss compares the result with the master.
   Its gradient trains the skips and the LoRA.

The 50-step Euler sampler makes its last call at t = 0.138 and returns exactly this estimate. So the skips learn from
latents of the kind they decode at inference. The usual noise levels would not suit the audio loss: their median is
t = 0.62, where the estimate is the blurred average of every master that fits the MP3. There, an audio loss would pull
the LoRA away from the average the sampler needs. At t = 0.138 the estimate is close to the master, and we expect the
two losses to agree.

The audio loss is an L1 distance on the waveform plus an L1 distance between multi-resolution STFT magnitudes. Both
measure fidelity, given Richter et al.'s result. Its weight against the flow loss is a setting.

## Validation keeps the audio

Validation today scores the flow loss on latents only. It will also keep each window's master and coded audio, and
report the audio loss at t = 0.138, decoded with the skips, next to the flow loss. If the two losses fight, the flow
loss rises above checkpoint-33135's while the audio loss falls. The best checkpoint becomes the one with the lowest
audio loss, and its listening windows are decoded with the skips.

## How we will know

We restore ADR-0016's 50 test clips at 32 kbps with checkpoint-33135 and with this run's best checkpoint. We score them
with ADR-0016's four metrics, next to the untouched MP3, the master's round trip and Apollo.

Two limits are known before we start. Above the MP3's cut-off the skips have nothing to carry, so that band stays as
good as the latents. And in training the skips decode estimates made from the master's noised latents, while at
inference they decode the sampler's own; the test clips show how much that matters.

## How we test it

Eight tests, three of them on the validation set.

Validation set:

- `test_a_validation_window_returns_the_same_stretch_of_the_master_as_its_latents`
- `test_a_validation_window_returns_the_audio_of_its_stored_copy`
- `test_the_audio_loss_is_identical_on_an_unchanged_model`

Skips and training:

- `test_untrained_skips_leave_the_decoded_audio_unchanged`
- `test_trained_skips_make_the_decoded_audio_depend_on_the_mp3`
- `test_one_epoch_with_skips_writes_their_weights_and_logs_both_validation_losses`
- `test_a_resume_from_a_checkpoint_without_skips_starts_them_at_zero`
- `test_saved_skips_reload_to_the_same_validation_audio_loss`
