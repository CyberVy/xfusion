from __future__ import annotations
import os
from typing import Optional, Dict, Union
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

from ..constants import FLUX_CONFIG_PATH, default_flux_transformer_url
from ..inspect import (
    resolve_model_source,
    build_bnb_config,
    build_gguf_config,
    build_pipeline_quant_config,
    apply_post_load_optimizations,
)
from ...infra import config

LoaderKwargValue = Union[str, float, int, bool, None, object]


def load_flux_pipeline(
    model: Optional[str] = None,
    download_kwargs: Optional[Dict[str, str]] = None,
    backbone: str = "fp16",
    text_encoder: str = "default",
    delete_file_after_load: bool = False,
    **kwargs: LoaderKwargValue,
) -> FluxPipeline:
    """Load FluxPipeline from Hugging Face repo, local directory, or single-file checkpoint."""
    model_name = model or default_flux_transformer_url
    quant_mode = (backbone or "fp16").lower()
    te_mode = (text_encoder or "default").lower()
    should_delete = delete_file_after_load or config.DELETE_FILE_AFTER_LOAD
    dl_kwargs = dict(download_kwargs or {})

    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN
    if call_kwargs.get("torch_dtype") is None:
        call_kwargs["torch_dtype"] = torch.float16
    call_kwargs["cache_dir"] = dl_kwargs.get("directory") or config.DOWNLOAD_PATH

    resolved, is_single_file, was_downloaded = resolve_model_source(model_name, dl_kwargs)

    if te_mode == "skip":
        call_kwargs["text_encoder_2"] = None
        call_kwargs["tokenizer_2"] = None

    if is_single_file:
        try:
            if resolved.endswith(".gguf") or quant_mode == "gguf":
                call_kwargs["transformer"] = FluxTransformer2DModel.from_single_file(
                    resolved,
                    config=FLUX_CONFIG_PATH,
                    subfolder="transformer",
                    quantization_config=build_gguf_config(),
                    torch_dtype=call_kwargs["torch_dtype"],
                )
            elif quant_mode in ("int4", "int8"):
                call_kwargs["transformer"] = FluxTransformer2DModel.from_single_file(
                    resolved,
                    config=FLUX_CONFIG_PATH,
                    subfolder="transformer",
                    quantization_config=build_bnb_config(quant_mode, target="diffusers"),
                    torch_dtype=call_kwargs["torch_dtype"],
                )
            pipe = FluxPipeline.from_single_file(resolved, config=FLUX_CONFIG_PATH, **call_kwargs)
        finally:
            if should_delete and was_downloaded and os.path.exists(resolved):
                os.remove(resolved)
    else:
        if te_mode != "skip" and "quantization_config" not in call_kwargs:
            quant_cfg = build_pipeline_quant_config(
                backbone=quant_mode,
                text_encoder=te_mode,
                backbone_key="transformer",
                text_encoder_keys=("text_encoder_2",),
            )
            if quant_cfg is not None:
                call_kwargs["quantization_config"] = quant_cfg
        pipe = FluxPipeline.from_pretrained(resolved, **call_kwargs)

    return apply_post_load_optimizations(pipe, backbone=quant_mode, text_encoder=te_mode)
