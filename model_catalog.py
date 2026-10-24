"""Curated local Diffusers models supported by the app.

Sizes are approximate model-download estimates from the model cards/repositories;
the actual download can vary by format and repository updates.
"""
from __future__ import annotations

MODEL_CATALOG = {
    "tiny-sd": {
        "name": "Segmind Tiny SD",
        "repo_id": "segmind/tiny-sd",
        "size": "about 1.1 GB",
        "tier": "Small",
        "steps": (10, 50, 25),
        "description": "Lowest storage and memory option; faster, lighter results.",
    },
    "stable-diffusion-v1-5": {
        "name": "Stable Diffusion 1.5",
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "size": "about 4.3 GB of model weights",
        "tier": "Medium",
        "steps": (1, 4, 2),
        "description": "Balanced general-purpose model; default selection.",
    },
    "sd-turbo": {
        "name": "Stable Diffusion Turbo",
        "repo_id": "stabilityai/sd-turbo",
        "size": "about 5.2 GB of model weights",
        "tier": "Medium",
        "description": "Fast generation; designed for very few steps.",
    },
    "dreamshaper-8": {
        "name": "DreamShaper 8",
        "repo_id": "Lykon/dreamshaper-8",
        "size": "about 8 GB",
        "tier": "Large",
        "description": "Versatile illustrative and photorealistic styles.",
    },
    "sdxl-base-1.0": {
        "name": "Stable Diffusion XL Base 1.0",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "size": "about 7 GB of model weights",
        "tier": "Large",
        "description": "Higher detail and native 1024-pixel output; needs more memory.",
    },
}

DEFAULT_MODEL = "stable-diffusion-v1-5"

# Standard Diffusers component folders only. This avoids pulling redundant .ckpt
# and full precision checkpoint files from repositories that publish them beside
# the Diffusers pipeline.
MODEL_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "text_encoder_2/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "unet/**",
    "vae/**",
    "feature_extractor/**",
    "safety_checker/**",
    "image_encoder/**",
    "transformer/**",
]
