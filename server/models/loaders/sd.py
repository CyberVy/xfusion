import os
import requests
import torch
from typing import Optional, Dict, Any

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
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
from ..inspect import infer_model_version
from ...infra.downloader import download_file
from ...infra import config


def load_stable_diffusion_controlnet(
    controlnet_model: str,
    model_version: Optional[str] = None,
    file_format: str = "safetensors",
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    use_internet = True
    model_version = str(model_version or "").lower()
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if download_kwargs is None:
        download_kwargs = {}

    if (
        controlnet_model.startswith(".")
        or controlnet_model.startswith("/")
        or controlnet_model.startswith("~")
    ):
        use_internet = False

    kwargs.update(cache_dir=download_kwargs.get("directory"))

    if use_internet:
        # from Hugging face repo
        if not (controlnet_model.startswith("http://") or controlnet_model.startswith("https://")):
            if model_version in ["1", "1.5", "2", "xl", "pony"]:
                return ControlNetModel.from_pretrained(controlnet_model, variant="fp16", **kwargs)
            elif model_version in ["3", "3.5"]:
                return SD3ControlNetModel.from_pretrained(controlnet_model, **kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
        # from single file URL
        else:
            file_path = download_file(controlnet_model, **download_kwargs)
            if not (
                file_path.endswith(".safetensors")
                or file_path.endswith(".bin")
                or file_path.endswith(".ckpt")
            ):
                print(f"Warning: Can't get the format of the downloaded file, guess it is .{file_format}")
                renamed = f"{file_path}.{file_format}"
                os.rename(file_path, renamed)
                file_path = renamed

            if not model_version:
                inferred = infer_model_version(file_path)
                if "controlnet_xl_large" == inferred:
                    model_version = "xl"
                elif "controlnet" == inferred:
                    model_version = "1.5"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")
                print(f"Auto detect result: {model_version}.")

            if model_version in ["1", "1.5", "2"]:
                return ControlNetModel.from_single_file(
                    file_path, config=SD_V1_V2_CONTROLNET_CONFIG_PATH, **kwargs
                )
            elif model_version in ["xl", "pony"]:
                return ControlNetModel.from_single_file(
                    file_path, config=SD_XL_CONTROLNET_CONFIG_PATH, **kwargs
                )
            elif model_version in ["3", "3.5"]:
                return SD3ControlNetModel.from_single_file(file_path, config=None, **kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
    else:
        # from local single file
        if (
            controlnet_model.endswith(".safetensors")
            or controlnet_model.endswith(".bin")
            or controlnet_model.endswith(".ckpt")
        ):
            if not model_version:
                inferred = infer_model_version(controlnet_model)
                if "controlnet_xl_large" == inferred:
                    model_version = "xl"
                elif "controlnet" == inferred:
                    model_version = "1.5"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")

            if model_version in ["1", "1.5", "2"]:
                return ControlNetModel.from_single_file(
                    controlnet_model, config=SD_V1_V2_CONTROLNET_CONFIG_PATH, **kwargs
                )
            elif model_version in ["xl", "pony"]:
                return ControlNetModel.from_single_file(
                    controlnet_model, config=SD_XL_CONTROLNET_CONFIG_PATH, **kwargs
                )
            elif model_version in ["3", "3.5"]:
                return SD3ControlNetModel.from_single_file(controlnet_model, config=None, **kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
        # from local directory
        else:
            if model_version in ["1", "1.5", "2", "xl", "pony"]:
                return ControlNetModel.from_pretrained(controlnet_model, **kwargs)
            elif model_version in ["3", "3.5"]:
                return SD3ControlNetModel.from_pretrained(controlnet_model, **kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")


def load_stable_diffusion_pipeline(
    model: Optional[str] = None,
    model_version: Optional[str] = None,
    file_format: str = "safetensors",
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    """Load DiffusionPipeline for SD 1.5, SD 2.x, SDXL, Pony or SD3."""
    use_internet = True
    model = model or default_stable_diffusion_model_url
    model_version = str(model_version or "").lower()
    if download_kwargs is None:
        download_kwargs = {}
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if model.startswith(".") or model.startswith("/") or model.startswith("~"):
        use_internet = False

    kwargs.update(cache_dir=download_kwargs.get("directory"))

    if use_internet:
        # from Hugging Face repo
        if not (model.startswith("http://") or model.startswith("https://")):
            text = requests.get(
                f"https://huggingface.co/{model}/tree/main/vae?not-for-all-audiences=true"
            ).text
            is_fp32 = "diffusion_pytorch_model.safetensors" in text
            is_fp16 = "diffusion_pytorch_model.fp16.safetensors" in text
            if is_fp16:
                kwargs.update({"variant": "fp16"})
            elif is_fp32:
                pass
            else:
                raise Exception(f"Model {model} is not supported.")
            return DiffusionPipeline.from_pretrained(model, **kwargs)
        # from single file URL
        else:
            file_path = download_file(model, **download_kwargs)
            if not (
                file_path.endswith(".safetensors")
                or file_path.endswith(".bin")
                or file_path.endswith(".ckpt")
            ):
                print(f"Warning: Can't get the format of the downloaded file, guess it is .{file_format}")
                renamed = f"{file_path}.{file_format}"
                os.rename(file_path, renamed)
                file_path = renamed

            if not model_version:
                inferred = infer_model_version(file_path)
                if "sd3" in inferred:
                    model_version = "3"
                elif "xl_base" == inferred:
                    model_version = "xl"
                elif "v1" == inferred:
                    model_version = "1"
                elif "v2" == inferred:
                    model_version = "2"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")
                print(f"Auto detect result: {model_version}.")

            if model_version in ["xl", "pony"]:
                return StableDiffusionXLPipeline.from_single_file(
                    file_path, config=SD_XL_CONFIG_PATH, **kwargs
                )
            elif model_version == "3":
                return StableDiffusion3Pipeline.from_single_file(
                    file_path, config=SD_3_CONFIG_PATH, **kwargs
                )
            elif model_version == "2":
                return StableDiffusionPipeline.from_single_file(
                    file_path, config=SD_V2_CONFIG_PATH, **kwargs
                )
            elif model_version in ["1", "1.5"]:
                return StableDiffusionPipeline.from_single_file(
                    file_path, config=SD_V1_CONFIG_PATH, **kwargs
                )
            else:
                raise ValueError(f"{model_version} is not supported yet.")
    else:
        # from local single file
        if (
            model.endswith(".safetensors")
            or model.endswith(".bin")
            or model.endswith(".ckpt")
        ):
            if not model_version:
                inferred = infer_model_version(model)
                if "sd3" in inferred:
                    model_version = "3"
                elif "xl_base" == inferred:
                    model_version = "xl"
                elif "v1" == inferred:
                    model_version = "1"
                elif "v2" == inferred:
                    model_version = "2"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")

            if model_version in ["xl", "pony"]:
                return StableDiffusionXLPipeline.from_single_file(
                    model, config=SD_XL_CONFIG_PATH, **kwargs
                )
            elif model_version == "3":
                return StableDiffusion3Pipeline.from_single_file(
                    model, config=SD_3_CONFIG_PATH, **kwargs
                )
            elif model_version == "2":
                return StableDiffusionPipeline.from_single_file(
                    model, config=SD_V2_CONFIG_PATH, **kwargs
                )
            elif model_version in ["1", "1.5"]:
                return StableDiffusionPipeline.from_single_file(
                    model, config=SD_V1_CONFIG_PATH, **kwargs
                )
            else:
                raise ValueError(f"{model_version} is not supported yet.")
        # from local directory
        else:
            return DiffusionPipeline.from_pretrained(model, **kwargs)
