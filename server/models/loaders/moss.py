from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch
from diffusers import DiffusionPipeline

from ..inspect import apply_post_load_optimizations, resolve_model_source
from ...infra import config

LoaderKwargValue = Union[str, float, int, bool, None, object]
_QUANTIZED_BACKBONES = ("int4", "int8", "fp8", "gguf")


def load_moss_pipeline(
    model: str,
    download_kwargs: Optional[Dict[str, str]] = None,
    backbone: str = "fp16",
    text_encoder: str = "default",
    delete_file_after_load: bool = False,
) -> DiffusionPipeline:
    """Load MOSS-Image from a Diffusers repository at its published precision."""
    quant_mode = (backbone or "fp16").lower()
    te_mode = (text_encoder or "default").lower()
    if quant_mode in _QUANTIZED_BACKBONES or te_mode != "default":
        raise ValueError("MOSS-Image only loads at its published precision.")
    if not model:
        raise ValueError("MOSS-Image requires a model repository.")

    should_delete = delete_file_after_load or config.DELETE_FILE_AFTER_LOAD
    dl_kwargs = dict(download_kwargs or {})
    resolved, is_single_file, was_downloaded = resolve_model_source(model, dl_kwargs)
    if is_single_file:
        if should_delete and was_downloaded and os.path.exists(resolved):
            os.remove(resolved)
        raise ValueError("MOSS-Image loads from a Diffusers repository, not a single checkpoint file.")

    call_kwargs: Dict[str, LoaderKwargValue] = {
        "custom_pipeline": "pipeline_moss_image",
        "trust_remote_code": True,
        "torch_dtype": {"default": torch.bfloat16, "vae": torch.float32},
        "token": config.HF_HUB_TOKEN,
        "cache_dir": dl_kwargs.get("directory") or config.DOWNLOAD_PATH,
    }
    pipe = DiffusionPipeline.from_pretrained(resolved, **call_kwargs)
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is not None:
        pipe.scheduler = type(scheduler).from_config(scheduler.config)
    return apply_post_load_optimizations(pipe, backbone="fp16", text_encoder="default")
