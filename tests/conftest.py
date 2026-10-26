import copy

import pytest
import torch


def _same(decoder: bool) -> dict:
    side = (
        {"out_channels": 32, "conv_mapping": True, "sinusoidal_blocks": [0]}
        if decoder
        else {"in_channels": 32}
    )
    return {
        "type": "taae_v2",
        "config": {
            **side,
            "channels": 16,
            "c_mults": [1],
            "strides": [4],
            "latent_dim": 8,
            "transformer_depths": [1],
            "dim_heads": 8,
            "use_flash": False,
            "variable_stride": True,
            "differential": False,
            "mapping_style": "none",
            "dyt": True,
            "chunk_size": 32,
            "chunk_midpoint_shift": True,
        },
    }


# Stable Audio 3's base config with every size shrunk: SAME's patch of 16
# samples times its stride of 4 makes 64 samples per latent frame, 8 latent
# channels, a 2-block transformer, and a number conditioner in place of the
# gated T5Gemma text encoder.
TINY_SA3 = {
    "model_type": "diffusion_cond_inpaint",
    "sample_size": 4 * 44_100,
    "sample_rate": 44_100,
    "audio_channels": 2,
    "model": {
        "io_channels": 8,
        "pretransform": {
            "type": "autoencoder",
            "config": {
                "pretransform": {"type": "patched", "config": {"patch_size": 16, "channels": 2}},
                "encoder": _same(decoder=False),
                "decoder": _same(decoder=True),
                "bottleneck": {
                    "type": "softnorm",
                    "config": {
                        "dim": 8,
                        "noise_augment_dim": 0,
                        "noise_regularize": False,
                        "auto_scale": False,
                        "freeze": True,
                    },
                },
                "latent_dim": 8,
                "downsampling_ratio": 64,
                "io_channels": 2,
            },
        },
        "conditioning": {
            "cond_dim": 16,
            "configs": [
                {
                    "id": "seconds_total",
                    "type": "number",
                    "config": {"min_val": 0, "max_val": 384, "fourier_features_type": "expo"},
                }
            ],
        },
        "diffusion": {
            "type": "dit",
            "diffusion_objective": "rectified_flow",
            "cross_attention_cond_ids": ["seconds_total"],
            "global_cond_ids": ["seconds_total"],
            "local_add_cond_ids": ["inpaint_mask", "inpaint_masked_input"],
            "use_effective_length_for_schedule": True,
            "mask_padding_attention": True,
            "distribution_shift_options": {"type": "full", "min_length": 256, "max_length": 4096},
            "config": {
                "io_channels": 8,
                "embed_dim": 128,
                "depth": 2,
                "num_heads": 2,
                "cond_token_dim": 16,
                "global_cond_dim": 16,
                "local_add_cond_dim": 9,
                "global_cond_type": "adaLN",
                "timestep_features_type": "expo",
                "attn_kwargs": {"qk_norm": "rms"},
                "norm_type": "rms_norm",
                "ff_kwargs": {"mult": 2.0},
                "num_memory_tokens": 4,
            },
        },
    },
}


@pytest.fixture
def tiny_checkpoint():
    """A seeded, randomly initialised tiny SA3 base checkpoint, (config, weights)."""
    from stable_audio_3.factory import create_diffusion_cond_from_config

    torch.manual_seed(0)
    config = copy.deepcopy(TINY_SA3)
    return config, create_diffusion_cond_from_config(config).state_dict()
