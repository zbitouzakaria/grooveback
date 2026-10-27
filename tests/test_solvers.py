"""The solvers' wiring: what reaches the model, what comes back.

The real models need gated checkpoints and a GPU, so recording fakes stand
in; the contract under test is ours, not the model authors'.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from grooveback import latents as gl
from grooveback import training as gt
from grooveback.audio import loudness
from grooveback.solvers import SDEDIT_MAX_SECONDS, latent_sub, lora, roundtrip, sdedit

SR = 44_100


class RecordingCodec:
    """A registry-shaped autoencoder fake: encode returns the audio itself as
    latents (so a `(2, samples)` input has 2 latent channels); decode returns
    the latents, or an injected render."""

    def __init__(self, render: np.ndarray | None = None):
        self.encodes = []
        self.decodes = []
        self._render = render

    def encode(self, audio, sample_rate, model):
        self.encodes.append(audio.shape)
        return audio.copy()

    def decode(self, latents, model):
        self.decodes.append(latents.copy())
        return latents if self._render is None else self._render


def register_fake(monkeypatch, codec: RecordingCodec) -> str:
    """Register a fake autoencoder under the name 'fake' for one test."""
    entry = gl._AE(load=None, encode=codec.encode, decode=codec.decode)
    monkeypatch.setitem(gl.AUTOENCODERS, "fake", entry)
    return "fake"


class RecordingPrior:
    """Records every generate() call; returns `batch`, or by default a fixed
    stereo batch whose two channels are distinguishable."""

    def __init__(self, samples: int = 2_000, batch: torch.Tensor | None = None):
        self.calls = []
        if batch is None:
            batch = torch.zeros((1, 2, samples))
            batch[0, 0] = 0.25
            batch[0, 1] = -0.5
        self._batch = batch

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._batch


class EdgeSpikingEarVAE2(torch.nn.Module):
    """εar-VAE2's interface around an identity codec. Like the upstream
    inverse STFT, its decode cannot reconstruct the last three samples it
    returns: they come back at 2.7."""

    samples_per_latent = 1_920

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_audio(self, audio, chunked, chunk_size, overlap, deterministic):
        return audio

    def decode_audio(self, latents, chunked, chunk_size, overlap):
        out = latents.clone()
        out[..., -3:] = 2.7
        return out


def test_sdedit_returns_channels_by_samples_float32_trimmed_to_input():
    model = RecordingPrior(samples=2_000)
    audio = np.zeros((2, 1_500), dtype=np.float32)

    out = sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)

    assert out.shape == (2, 1_500)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], 0.25)
    np.testing.assert_array_equal(out[1], -0.5)


def test_sdedit_passes_input_as_channels_by_samples_tensor():
    """stable-audio-3 reads 2-D *numpy* init audio as (samples, channels) and
    transposes it, so the solver must hand over a torch tensor, which passes
    through untransposed."""
    model = RecordingPrior()
    audio = np.arange(6, dtype=np.float32).reshape(2, 3)

    sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)

    in_sr, tensor = model.calls[0]["init_audio"]
    assert in_sr == SR
    assert isinstance(tensor, torch.Tensor)
    torch.testing.assert_close(
        tensor, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    )


def test_sdedit_maps_noise_level_and_sampling_settings_onto_generate():
    """Classic SDEdit (theta unset): the schedule start and the mix are both
    noise_level."""
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    sdedit(
        model, audio, SR,
        noise_level=0.3, steps=50, cfg_scale=7.0,
        prompt="a prompt", negative_prompt="muffled", seed=7,
    )

    call = model.calls[0]
    assert call["init_noise_level"] == 0.3
    assert call["init_mix_level"] == 0.3
    assert call["steps"] == 50
    assert call["cfg_scale"] == 7.0
    assert call["prompt"] == "a prompt"
    assert call["negative_prompt"] == "muffled"
    assert call["seed"] == 7


def test_sdedit_theta_decouples_schedule_start_from_the_mix():
    """theta sets the schedule time that denoising starts from; noise_level
    sets how much noise is mixed into the input. With noise_level 0 no noise
    is added, so all of the denoising works on the damage itself (ADR-0009)."""
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    sdedit(model, audio, SR, noise_level=0.0, theta=0.1, steps=50, cfg_scale=1.0)

    call = model.calls[0]
    assert call["init_noise_level"] == 0.1
    assert call["init_mix_level"] == 0.0


def test_sdedit_requests_the_inputs_full_length_past_the_default_clamp():
    """generate()'s sample_size defaults to 120 s of samples and silently
    truncates anything longer (ADR-0008), so on a longer input the solver
    must ask for the input's own length plus the schedule's 6 s padding —
    and a duration that covers every input sample."""
    model = RecordingPrior()
    n = 200 * SR  # 200 s, past the 120 s default clamp
    audio = np.zeros((2, n), dtype=np.float32)

    sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)

    call = model.calls[0]
    assert call["sample_size"] >= n + 6 * SR
    assert call["duration"] * SR >= n


def test_sdedit_matches_output_loudness_to_the_input():
    """The waveform baselines' output tracks the input's level by
    construction; the decoder's native level drifts with noise_level, so the
    solver scales its output to the input's integrated loudness. One second
    of tone, because loudness needs at least the meter's 400 ms block."""
    t = np.arange(SR, dtype=np.float32) / SR
    tone = np.sin(2 * np.pi * 997.0 * t, dtype=np.float32)
    loud_input = np.tile(0.5 * tone, (2, 1))
    quiet_render = torch.from_numpy(np.tile(0.05 * tone, (2, 1))).unsqueeze(0)
    model = RecordingPrior(batch=quiet_render)

    out = sdedit(model, loud_input, SR, noise_level=0.5, steps=8, cfg_scale=1.0)

    # Gain-only matching is exact; the tolerance covers float32 roundoff.
    assert loudness(out, SR) == pytest.approx(loudness(loud_input, SR), abs=0.01)


@pytest.mark.parametrize("noise_level", [0.0, -0.2, 1.5])
def test_noise_level_of_zero_or_outside_unit_interval_is_rejected_before_generation(noise_level):
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="noise_level"):
        sdedit(model, audio, SR, noise_level=noise_level, steps=8, cfg_scale=1.0)
    assert model.calls == []


@pytest.mark.parametrize("theta", [0.0, -0.1, 1.5])
def test_theta_of_zero_or_outside_unit_interval_is_rejected_before_generation(theta):
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="theta"):
        sdedit(model, audio, SR, noise_level=0.0, theta=theta, steps=8, cfg_scale=1.0)
    assert model.calls == []


@pytest.mark.parametrize("noise_level", [-0.2, 1.5])
def test_noise_level_outside_unit_interval_is_rejected_when_theta_is_set(noise_level):
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="noise_level"):
        sdedit(
            model, audio, SR,
            noise_level=noise_level, theta=0.1, steps=8, cfg_scale=1.0,
        )
    assert model.calls == []


@pytest.mark.parametrize("steps", [0, -8])
def test_non_positive_steps_is_rejected_before_generation(steps):
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="steps"):
        sdedit(model, audio, SR, noise_level=0.5, steps=steps, cfg_scale=1.0)
    assert model.calls == []


def test_wrong_sample_rate_is_rejected_before_generation():
    model = RecordingPrior()
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="44100 Hz"):
        sdedit(model, audio, sample_rate=48_000, noise_level=0.5, steps=8, cfg_scale=1.0)
    assert model.calls == []


def test_non_finite_input_is_rejected_before_generation():
    model = RecordingPrior()
    audio = np.full((2, 1_000), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN"):
        sdedit(model, audio, SR, noise_level=0.5, steps=8, cfg_scale=1.0)
    assert model.calls == []


def test_input_longer_than_the_cap_is_rejected_before_generation():
    model = RecordingPrior()
    n = int(SDEDIT_MAX_SECONDS * SR) + SR  # one second past the cap

    with pytest.raises(ValueError, match="whole files up to"):
        sdedit(
            model, np.zeros((2, n), dtype=np.float32), SR,
            noise_level=0.5, steps=8, cfg_scale=1.0,
        )
    assert model.calls == []


def test_roundtrip_through_an_identity_codec_returns_the_input(monkeypatch):
    """Latents are the audio and decode returns them, so any wiring change
    shows up as an inequality. The input is shorter than the loudness meter's
    400 ms block, so no gain is applied and equality is exact."""
    codec = RecordingCodec()
    ae = register_fake(monkeypatch, codec)
    audio = np.stack([np.linspace(0, 1, 1_000), np.linspace(0, -1, 1_000)]).astype(np.float32)

    out = roundtrip(object(), audio, SR, ae=ae)

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, audio)


def test_roundtrip_trims_a_decoder_overrun_to_the_input_length(monkeypatch):
    codec = RecordingCodec(render=np.ones((2, 1_005), dtype=np.float32))
    ae = register_fake(monkeypatch, codec)
    audio = np.ones((2, 1_000), dtype=np.float32)

    out = roundtrip(object(), audio, SR, ae=ae)

    assert out.shape == (2, 1_000)


def test_roundtrip_pads_a_resampling_shortfall_with_zeros(monkeypatch):
    """A 48 kHz model's output can come back a couple of samples short after
    resampling back to 44.1 kHz. The render must keep the input's exact
    length, because scripts/build_demo.py refuses a listening pack whose
    tracks differ in length; the missing tail is filled with silence."""
    codec = RecordingCodec(render=np.ones((2, 998), dtype=np.float32))
    ae = register_fake(monkeypatch, codec)
    audio = np.ones((2, 1_000), dtype=np.float32)

    out = roundtrip(object(), audio, SR, ae=ae)

    assert out.shape == (2, 1_000)
    np.testing.assert_array_equal(out[:, 998:], 0.0)


def test_roundtrip_matches_output_loudness_to_the_input(monkeypatch):
    """One second of tone, because loudness needs the meter's 400 ms block;
    the decoder's quiet render must come back at the input's level."""
    t = np.arange(SR, dtype=np.float32) / SR
    tone = np.sin(2 * np.pi * 997.0 * t, dtype=np.float32)
    loud_input = np.tile(0.5 * tone, (2, 1))
    codec = RecordingCodec(render=np.tile(0.05 * tone, (2, 1)))
    ae = register_fake(monkeypatch, codec)

    out = roundtrip(object(), loud_input, SR, ae=ae)

    assert loudness(out, SR) == pytest.approx(loudness(loud_input, SR), abs=0.01)


def test_earvae2_roundtrip_ends_without_the_decoders_edge_spike():
    """Regression: every εar-VAE2 render ended on a click, its last three
    samples rising to 2.7 on a clip that peaks at 0.9. The upstream inverse STFT
    returns a few final samples that no full window covers; torch warns that the
    frames are shorter than the requested length. It shows when the input lasts
    a whole number of the model's 40 ms frames, as 6 s clips and 3-minute
    excerpts do; any other length is padded to the next frame, which holds the
    spike. So the clip is 0.36 s, nine frames, and under the loudness meter's
    400 ms block, so no gain is applied. atol=1e-6 is above the error of the
    44.1 -> 48 -> 44.1 kHz resample on a Hann-faded tone (1.5e-8)."""
    t = np.arange(15_876) / SR  # 0.36 s, 17,280 samples at 48 kHz: nine 1,920-sample frames
    audio = (np.hanning(15_876) * 0.5 * np.sin(2 * np.pi * 1_000 * t)).astype(np.float32)
    audio = np.stack([audio, -audio])

    out = roundtrip(EdgeSpikingEarVAE2(), audio, SR, ae="earvae2")

    np.testing.assert_allclose(out, audio, rtol=0, atol=1e-6)


def test_latent_sub_with_zero_damage_equals_the_round_trip(monkeypatch):
    codec = RecordingCodec()
    ae = register_fake(monkeypatch, codec)
    audio = np.stack([np.linspace(0, 1, 1_000), np.linspace(0, -1, 1_000)]).astype(np.float32)

    subtracted = latent_sub(object(), audio, SR, ae=ae, damage=np.zeros(2, dtype=np.float32))
    plain = roundtrip(object(), audio, SR, ae=ae)

    np.testing.assert_array_equal(subtracted, plain)


def test_latent_sub_subtracts_the_damage_from_every_frame(monkeypatch):
    """With identity encode/decode the output is audio minus the per-channel
    damage value, which makes the arithmetic visible end to end."""
    codec = RecordingCodec()
    ae = register_fake(monkeypatch, codec)
    audio = np.ones((2, 1_000), dtype=np.float32)

    out = latent_sub(
        object(), audio, SR, ae=ae, damage=np.array([0.25, 0.5], dtype=np.float32)
    )

    np.testing.assert_array_equal(out[0], 0.75)
    np.testing.assert_array_equal(out[1], 0.5)


def test_latent_sub_refuses_damage_from_another_autoencoder(monkeypatch):
    codec = RecordingCodec()
    ae = register_fake(monkeypatch, codec)
    audio = np.zeros((2, 1_000), dtype=np.float32)

    with pytest.raises(ValueError, match="different autoencoder"):
        latent_sub(object(), audio, SR, ae=ae, damage=np.zeros(128, dtype=np.float32))
    assert codec.decodes == []


def test_lora_returns_the_input_length_at_the_input_loudness(
    tmp_path, monkeypatch, tiny_checkpoint
):
    """A saved checkpoint of the tiny SA3 (tests/conftest.py), reloaded and run
    on 1 s of noise: 44_100 samples are 689 frames of 64, so the solver pads
    the last 4 samples. The tolerance covers float32 rounding of the gain."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    trained = gt.load_model("small-music-base", device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(gt.add_lora(trained, gt.lora_config(rank=4, lora_alpha=4)))
    gt.save_checkpoint(tmp_path, trained, optimizer, config=gt.lora_config(rank=4, lora_alpha=4),
                       step=1, epoch=1)
    model = gt.load_trained_model("small-music-base", tmp_path, device=torch.device("cpu"))
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2, 44_100)).astype(np.float32)

    out = lora(model, audio, SR, steps=2, inpaint_mask_value=0.0)

    assert out.shape == (2, 44_100)
    assert loudness(out, SR) == pytest.approx(loudness(audio, SR), abs=0.01)


def test_lora_restores_the_last_partial_frame(tmp_path, monkeypatch, tiny_checkpoint):
    """1 s is 689 frames of 64 and 4 samples. Those 4 used to come back as zeros:
    on a 6 s MUSDB test clip, 56 ms of silence that every metric counts."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    trained = gt.load_model("small-music-base", device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(gt.add_lora(trained, gt.lora_config(rank=4, lora_alpha=4)))
    gt.save_checkpoint(tmp_path, trained, optimizer, config=gt.lora_config(rank=4, lora_alpha=4),
                       step=1, epoch=1)
    model = gt.load_trained_model("small-music-base", tmp_path, device=torch.device("cpu"))
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2, 44_100)).astype(np.float32)

    out = lora(model, audio, SR, steps=2, inpaint_mask_value=0.0)

    assert np.abs(out[:, 44_096:]).min() > 0.0


def test_lora_hands_the_sampler_to_sa3(tmp_path, monkeypatch, tiny_checkpoint):
    """SA3 refuses a sampler it does not know, so the name reached it."""
    monkeypatch.setattr(gt, "base_checkpoint", lambda variant: tiny_checkpoint)
    model = gt.load_model("small-music-base", device=torch.device("cpu"))
    gt.add_lora(model, gt.lora_config(rank=4, lora_alpha=4))
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2, 44_100)).astype(np.float32)

    with pytest.raises(ValueError, match="Unknown sampler_type"):
        lora(model, audio, SR, steps=2, inpaint_mask_value=0.0, sampler_type="heun")


def test_lora_rejects_input_longer_than_the_cap_before_encoding():
    model = Mock()
    model.sample_rate = SR
    n = int(SDEDIT_MAX_SECONDS * SR) + SR  # one second past the cap

    with pytest.raises(ValueError, match="whole files up to"):
        lora(model, np.zeros((2, n), dtype=np.float32), SR, steps=8, inpaint_mask_value=0.0)
    model.pretransform.encode.assert_not_called()
