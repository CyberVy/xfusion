from __future__ import annotations
from collections import OrderedDict
import os
from typing import Dict, Optional, Union, List, Callable, Tuple
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3ControlNetInpaintingPipeline,
)
from diffusers.utils import BaseOutput

from ..base import BasePipelineEngine
from ..schedulers import DIFFUSION_SCHEDULER_MAP
from ..prompt import get_embeds_from_pipeline, skip_clip_layer
from ..types import PipelineMap, InferenceKwargs, TorchTensor, TorchGenerator
from ...infra.image import normalize_image
from ...models.loaders.sd import load_stable_diffusion_pipeline, load_stable_diffusion_controlnet
from ...models.constants import (
    default_sd_v1_v2_controlnet_model_url,
    default_sd_xl_controlnet_model_url,
)
from ...models.inspect import get_tokenizers_and_text_encoders_from_pipeline

SD_PIPELINE_MAP: PipelineMap = {
    "1.5": (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
    ),
    "xl": (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
    ),
    "3": (
        StableDiffusion3Pipeline,
        StableDiffusion3Img2ImgPipeline,
        StableDiffusion3InpaintPipeline,
        StableDiffusion3ControlNetPipeline,
        StableDiffusion3ControlNetPipeline,
        StableDiffusion3ControlNetInpaintingPipeline,
    ),
}

_MAX_EMBED_CACHE_SIZE = 8


class SDPipelineEngine(BasePipelineEngine):
    pipeline_map = SD_PIPELINE_MAP
    scheduler_map = DIFFUSION_SCHEDULER_MAP

    def __init__(self, pipeline: Optional[DiffusionPipeline]):
        super().__init__(pipeline)
        self._embed_cache: "OrderedDict[Tuple, Dict[str, TorchTensor]]" = OrderedDict()
        if pipeline is not None:
            self.tokenizers, self.text_encoders = get_tokenizers_and_text_encoders_from_pipeline(
                self.pipeline
            )
        else:
            self.tokenizers, self.text_encoders = [], []

    def get_embeds(
        self,
        prompt: Union[str, List[str], None],
        negative_prompt: Union[str, List[str], None],
        clip_skip: Optional[int] = None,
    ) -> Dict[str, TorchTensor]:
        active_loras = tuple(sorted((k, v) for k, v in self.lora_dict.items() if v != 0))
        cache_key = (
            str(prompt),
            str(negative_prompt),
            clip_skip or 0,
            self.loras_enabled,
            active_loras,
        )
        exec_device = self.execution_device
        if cache_key in self._embed_cache:
            self._embed_cache.move_to_end(cache_key)
            cached = self._embed_cache[cache_key]
            if exec_device is not None:
                return {k: v.to(exec_device) for k, v in cached.items()}
            return dict(cached)

        skip_clip_layer(self.text_encoders, clip_skip)
        try:
            embeds = get_embeds_from_pipeline(self, prompt, negative_prompt)
        finally:
            skip_clip_layer(self.text_encoders, 0)

        if embeds:
            cpu_copy = {k: v.detach().cpu() for k, v in embeds.items() if hasattr(v, "detach")}
            if cpu_copy:
                self._embed_cache[cache_key] = cpu_copy
                while len(self._embed_cache) > _MAX_EMBED_CACHE_SIZE:
                    self._embed_cache.popitem(last=False)

        return dict(embeds)

    def check_inference_kwargs(
        self, kwargs: InferenceKwargs, use_controlnet: bool = False
    ) -> InferenceKwargs:
        if kwargs.get("negative_prompt") is None:
            kwargs["negative_prompt"] = ""

        width = kwargs.get("width")
        height = kwargs.get("height")
        image = kwargs.get("image")
        default_dim = 1024 if self.model_version in ["xl", "3"] else 512

        if image and isinstance(image, Image.Image):
            if width is None and height is None:
                width = image.width - (image.width % 8)
                height = image.height - (image.height % 8)
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                target_w = width or default_dim
                target_h = height or default_dim
                image = normalize_image(image, target_w * target_h, scale_divisor=8)
                width = image.width
                height = image.height
            kwargs.update(width=width, height=height, image=image)
        else:
            kwargs["width"] = width or default_dim
            kwargs["height"] = height or default_dim

        cur_w = kwargs.get("width", default_dim)
        cur_h = kwargs.get("height", default_dim)

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            if mask_image.size != (cur_w, cur_h):
                mask_image = mask_image.resize((cur_w, cur_h), Image.Resampling.LANCZOS)
            kwargs["mask_image"] = mask_image

        ip_adapter_image = kwargs.get("ip_adapter_image")
        if ip_adapter_image and isinstance(ip_adapter_image, Image.Image):
            kwargs["ip_adapter_image"] = normalize_image(
                ip_adapter_image, cur_w * cur_h, scale_divisor=8
            )

        if use_controlnet:
            control_image = kwargs.get("control_image")
            if control_image and isinstance(control_image, Image.Image):
                if control_image.size != (cur_w, cur_h):
                    control_image = control_image.resize((cur_w, cur_h), Image.Resampling.LANCZOS)
                kwargs["control_image"] = control_image

        return kwargs

    def __call__(
        self,
        task: str = "txt2img",
        use_controlnet: bool = False,
        **kwargs: Union[str, int, float, bool, Image.Image, None, Callable, TorchGenerator],
    ) -> BaseOutput:
        kwargs_dict: InferenceKwargs = self.check_inference_kwargs(
            dict(kwargs), use_controlnet=use_controlnet  # type: ignore
        )

        prompt = kwargs_dict.get("prompt")
        negative_prompt = kwargs_dict.get("negative_prompt")
        prompt_str = f"{prompt or ''} {negative_prompt or ''}"

        skipped_lora_dict = {}
        was_enabled = self.loras_enabled
        if self.loras_enabled and self.lora_dict:
            for lora, weight in list(self.lora_dict.items()):
                if lora not in prompt_str:
                    skipped_lora_dict[lora] = weight
                    self.set_lora_strength(lora, 0)

            if len(skipped_lora_dict) == len(self.lora_dict):
                self.disable_lora()

        try:
            embeds = self.get_embeds(prompt, negative_prompt, kwargs_dict.get("clip_skip"))
            if embeds:
                kwargs_dict.update(embeds)  # type: ignore
                kwargs_dict["prompt"] = None
                kwargs_dict["negative_prompt"] = None
                kwargs_dict.pop("clip_skip", None)

            return super().__call__(task=task, use_controlnet=use_controlnet, **kwargs_dict)
        finally:
            if was_enabled and not self.loras_enabled:
                self.enable_lora()
            if skipped_lora_dict:
                for lora, weight in skipped_lora_dict.items():
                    self.set_lora_strength(lora, weight)

    def load_controlnet(
        self,
        controlnet_model: Optional[str] = None,
        **kwargs: Union[str, float, int, bool],
    ) -> None:
        if self._controlnet is not None:
            return

        download_kwargs = self.download_kwargs.copy()
        directory = download_kwargs.get("directory", "./")

        if self.model_version in ["1", "1.5"]:
            download_kwargs["directory"] = os.path.join(directory, "controlnet_sd_v1_v2")
            model_url = controlnet_model or default_sd_v1_v2_controlnet_model_url
        elif self.model_version in ["xl", "pony"]:
            download_kwargs["directory"] = os.path.join(directory, "controlnet_sdxl")
            model_url = controlnet_model or default_sd_xl_controlnet_model_url
        else:
            raise NotImplementedError(
                f"ControlNet for model version {self.model_version} is not supported."
            )

        self._controlnet = load_stable_diffusion_controlnet(
            model_url, self.model_version, download_kwargs=download_kwargs, **kwargs
        )
        self._task_pipes.clear()
        if self.offload_mode == "none" and self.execution_device is not None:
            self._controlnet = self._controlnet.to(self.execution_device)
        elif self.offload_mode == "model_cpu_offload" and self.execution_device is not None:
            try:
                from accelerate import cpu_offload_with_hook
                _, hook = cpu_offload_with_hook(self._controlnet, execution_device=self.execution_device)
                if self.pipeline is not None and hasattr(self.pipeline, "_all_hooks"):
                    self.pipeline._all_hooks.append(hook)
            except Exception:
                self._controlnet = self._controlnet.to(self.execution_device)
        elif self.offload_mode == "sequential_cpu_offload" and self.execution_device is not None:
            try:
                from accelerate import cpu_offload
                cpu_offload(self._controlnet, execution_device=self.execution_device)
            except Exception:
                self._controlnet = self._controlnet.to(self.execution_device)

    def clear(self) -> None:
        self._embed_cache.clear()
        super().clear()

    @classmethod
    def from_url(
        cls,
        url: Optional[str] = None,
        model_version: Optional[str] = None,
        quantization: str = "none",
        text_encoder_mode: str = "default",
        delete_after_load: bool = False,
        **kwargs: Union[str, float, int, bool],
    ) -> "SDPipelineEngine":
        raw_pipe = load_stable_diffusion_pipeline(
            model=url,
            model_version=model_version,
            quantization=quantization,
            text_encoder_mode=text_encoder_mode,
            delete_after_load=delete_after_load,
            **kwargs,
        )
        engine = cls(raw_pipe)
        engine.quantization = quantization
        engine.text_encoder_mode = text_encoder_mode
        if url:
            engine.model_name = url
        return engine
