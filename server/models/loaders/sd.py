from __future__ import annotations
import os
from typing import Optional, Dict, Union, Tuple, Type
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    UNet2DConditionModel,
    SD3Transformer2DModel,
    ControlNetModel,
    SD3ControlNetModel,
)

from ..constants import (
    default_stable_diffusion_model_url,
    SD_V1_CONFIG_PATH,
    SD_V2_CONFIG_PATH,
    SD_3_CONFIG_PATH,
    SD_XL_CONFIG_PATH,
    SD_V1_V2_CONTROLNET_CONFIG_PATH,
    SD_XL_CONTROLNET_CONFIG_PATH,
)
from ..inspect import (
    infer_model_version,
    resolve_model_source,
    build_bnb_config,
    build_gguf_config,
    build_pipeline_quant_config,
    apply_post_load_optimizations,
)
from ...infra import config

LoaderKwargValue = Union[str, float, int, bool, None, object]


def _normalize_controlnet_version(inferred: str) -> str:
    if inferred == "controlnet_xl_large":
        return "xl"
    if inferred == "controlnet":
        return "1.5"
    raise ValueError(f"Unsupported ControlNet architecture: {inferred}")


def _normalize_sd_version(inferred: str) -> str:
    if "sd3" in inferred:
        return "3"
    if "xl" in inferred:
        return "xl"
    if inferred == "v2":
        return "2"
    if inferred == "v1":
        return "1.5"
    raise ValueError(f"Unsupported Stable Diffusion architecture: {inferred}")


def load_stable_diffusion_controlnet(
    controlnet_model: str,
    model_version: Optional[str] = None,
    file_format: str = "safetensors",
    download_kwargs: Optional[Dict[str, str]] = None,
    delete_file_after_load: bool = False,
    **kwargs: LoaderKwargValue,
) -> Union[ControlNetModel, SD3ControlNetModel]:
    version = str(model_version or "").lower()
    dl_kwargs = dict(download_kwargs or {})
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN
    if call_kwargs.get("torch_dtype") is None:
        call_kwargs["torch_dtype"] = torch.float16
    call_kwargs["cache_dir"] = dl_kwargs.get("directory") or config.DOWNLOAD_PATH

    resolved, is_single_file, was_downloaded = resolve_model_source(
        controlnet_model, dl_kwargs, default_ext=file_format
    )

    if is_single_file:
        if not version:
            version = _normalize_controlnet_version(infer_model_version(resolved))
        try:
            if version in ("1", "1.5", "2"):
                return ControlNetModel.from_single_file(
                    resolved, config=SD_V1_V2_CONTROLNET_CONFIG_PATH, **call_kwargs
                )
            if version in ("xl", "sdxl", "pony"):
                return ControlNetModel.from_single_file(
                    resolved, config=SD_XL_CONTROLNET_CONFIG_PATH, **call_kwargs
                )
            if version in ("3", "3.5"):
                return SD3ControlNetModel.from_single_file(resolved, config=None, **call_kwargs)
            raise ValueError(f"ControlNet version '{version}' is not supported.")
        finally:
            if (delete_file_after_load or config.DELETE_FILE_AFTER_LOAD) and was_downloaded and os.path.exists(resolved):
                os.remove(resolved)

    if version in ("3", "3.5"):
        return SD3ControlNetModel.from_pretrained(resolved, **call_kwargs)

    if version in ("1", "1.5", "2", "xl", "sdxl", "pony", ""):
        if not os.path.isdir(resolved) and "variant" not in call_kwargs:
            try:
                return ControlNetModel.from_pretrained(resolved, variant="fp16", **call_kwargs)
            except (ValueError, OSError):
                pass
        return ControlNetModel.from_pretrained(resolved, **call_kwargs)

    raise ValueError(f"ControlNet version '{version}' is not supported.")


_SINGLE_FILE_SPECS: Dict[str, Tuple[Type[DiffusionPipeline], str, Type[object], str]] = {
    "1": (StableDiffusionPipeline, SD_V1_CONFIG_PATH, UNet2DConditionModel, "unet"),
    "1.5": (StableDiffusionPipeline, SD_V1_CONFIG_PATH, UNet2DConditionModel, "unet"),
    "2": (StableDiffusionPipeline, SD_V2_CONFIG_PATH, UNet2DConditionModel, "unet"),
    "xl": (StableDiffusionXLPipeline, SD_XL_CONFIG_PATH, UNet2DConditionModel, "unet"),
    "sdxl": (StableDiffusionXLPipeline, SD_XL_CONFIG_PATH, UNet2DConditionModel, "unet"),
    "pony": (StableDiffusionXLPipeline, SD_XL_CONFIG_PATH, UNet2DConditionModel, "unet"),
    "3": (StableDiffusion3Pipeline, SD_3_CONFIG_PATH, SD3Transformer2DModel, "transformer"),
    "3.5": (StableDiffusion3Pipeline, SD_3_CONFIG_PATH, SD3Transformer2DModel, "transformer"),
}


def load_stable_diffusion_pipeline(
    model: Optional[str] = None,
    model_version: Optional[str] = None,
    file_format: str = "safetensors",
    download_kwargs: Optional[Dict[str, str]] = None,
    backbone: str = "fp16",
    text_encoder: str = "default",
    delete_file_after_load: bool = False,
    **kwargs: LoaderKwargValue,
) -> Tuple[DiffusionPipeline, str]:
    """Load a Stable Diffusion pipeline and return the resolved version id."""
    target_model = model or default_stable_diffusion_model_url
    version = str(model_version or "").lower()
    quant_mode = (backbone or "fp16").lower()
    te_mode = (text_encoder or "default").lower()
    dl_kwargs = dict(download_kwargs or {})

    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN
    if call_kwargs.get("torch_dtype") is None:
        call_kwargs["torch_dtype"] = torch.float16
    call_kwargs["cache_dir"] = dl_kwargs.get("directory") or config.DOWNLOAD_PATH

    resolved, is_single_file, was_downloaded = resolve_model_source(
        target_model, dl_kwargs, default_ext=file_format
    )

    if te_mode == "skip" and version in ("3", "3.5"):
        call_kwargs["text_encoder_3"] = None
        call_kwargs["tokenizer_3"] = None

    if is_single_file:
        if not version:
            version = _normalize_sd_version(infer_model_version(resolved))
        if te_mode == "skip" and version in ("3", "3.5"):
            call_kwargs["text_encoder_3"] = None
            call_kwargs["tokenizer_3"] = None

        spec = _SINGLE_FILE_SPECS.get(version)
        if spec is None:
            raise ValueError(f"Model version '{version}' is not supported.")
        pipe_cls, cfg_path, backbone_cls, backbone_key = spec

        try:
            if resolved.endswith(".gguf") or quant_mode == "gguf":
                call_kwargs[backbone_key] = backbone_cls.from_single_file(  # type: ignore[attr-defined]
                    resolved,
                    config=cfg_path,
                    subfolder=backbone_key,
                    quantization_config=build_gguf_config(),
                    torch_dtype=call_kwargs["torch_dtype"],
                )
            elif quant_mode in ("int4", "int8"):
                call_kwargs[backbone_key] = backbone_cls.from_single_file(  # type: ignore[attr-defined]
                    resolved,
                    config=cfg_path,
                    subfolder=backbone_key,
                    quantization_config=build_bnb_config(quant_mode, target="diffusers"),
                    torch_dtype=call_kwargs["torch_dtype"],
                )

            pipe = pipe_cls.from_single_file(resolved, config=cfg_path, **call_kwargs)
        finally:
            if (delete_file_after_load or config.DELETE_FILE_AFTER_LOAD) and was_downloaded and os.path.exists(resolved):
                os.remove(resolved)
    else:
        backbone_key = "transformer" if version in ("3", "3.5") else "unet"
        te_keys = ("text_encoder_3",) if version in ("3", "3.5") else ("text_encoder_2",)
        if "quantization_config" not in call_kwargs:
            quant_cfg = build_pipeline_quant_config(
                backbone=quant_mode,
                text_encoder=te_mode,
                backbone_key=backbone_key,
                text_encoder_keys=te_keys,
            )
            if quant_cfg is not None:
                call_kwargs["quantization_config"] = quant_cfg

        if not os.path.isdir(resolved) and "variant" not in call_kwargs:
            try:
                pipe = DiffusionPipeline.from_pretrained(resolved, variant="fp16", **call_kwargs)
            except (ValueError, OSError):
                pipe = DiffusionPipeline.from_pretrained(resolved, **call_kwargs)
        else:
            pipe = DiffusionPipeline.from_pretrained(resolved, **call_kwargs)

    pipe = apply_post_load_optimizations(pipe, backbone=quant_mode, text_encoder=te_mode)
    return pipe, version
