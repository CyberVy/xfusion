from __future__ import annotations
from collections import OrderedDict
from typing import Optional, Union, Callable, Dict, Tuple
from PIL import Image

from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxControlPipeline,
    FluxControlImg2ImgPipeline,
    FluxControlInpaintPipeline,
)
from diffusers.utils import BaseOutput

from ..base import BasePipelineEngine
from ..types import PipelineMap, InferenceKwargs, TorchTensor
from ...infra.image import normalize_image
from ...infra.memory import free_memory_to_system
from ...models.loaders.flux import load_flux_pipeline

FLUX_PIPELINE_MAP: PipelineMap = {
    "flux": (
        FluxPipeline,
        FluxImg2ImgPipeline,
        FluxInpaintPipeline,
        FluxControlPipeline,
        FluxControlImg2ImgPipeline,
        FluxControlInpaintPipeline,
    )
}

_MAX_FLUX_EMBED_CACHE_SIZE = 6


class FluxPipelineEngine(BasePipelineEngine):
    pipeline_map = FLUX_PIPELINE_MAP

    def __init__(self, pipeline: Optional[FluxPipeline]):
        super().__init__(pipeline)
        self._embed_cache: "OrderedDict[Tuple, Dict[str, TorchTensor]]" = OrderedDict()

    def _get_cached_flux_embeds(self, prompt: Optional[Union[str, list]]) -> Dict[str, TorchTensor]:
        if not prompt or self.pipeline is None or not hasattr(self.pipeline, "encode_prompt"):
            return {}

        active_loras = tuple(sorted((k, v) for k, v in self.lora_dict.items() if v != 0))
        cache_key = (str(prompt), self.loras_enabled, active_loras)
        exec_device = self.execution_device

        if cache_key in self._embed_cache:
            self._embed_cache.move_to_end(cache_key)
            cached = self._embed_cache[cache_key]
            if exec_device is not None:
                return {k: v.to(exec_device) for k, v in cached.items()}
            return dict(cached)

        try:
            prompt_embeds, pooled_prompt_embeds, _text_ids = self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=exec_device,
            )
            result: Dict[str, TorchTensor] = {}
            if prompt_embeds is not None and pooled_prompt_embeds is not None:
                result = {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                }
                self._embed_cache[cache_key] = {
                    k: v.detach().cpu() for k, v in result.items()
                }
                while len(self._embed_cache) > _MAX_FLUX_EMBED_CACHE_SIZE:
                    self._embed_cache.popitem(last=False)
                free_memory_to_system()
            return result
        except Exception:
            return {}

    def check_inference_kwargs(
        self, kwargs: InferenceKwargs, use_controlnet: bool = False
    ) -> InferenceKwargs:
        width = kwargs.get("width")
        height = kwargs.get("height")
        image = kwargs.get("image")

        if image and isinstance(image, Image.Image):
            if width is None and height is None:
                width = image.width - (image.width % 16)
                height = image.height - (image.height % 16)
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                target_w = width or 1024
                target_h = height or 1024
                image = normalize_image(image, target_w * target_h, scale_divisor=16)
                width = image.width
                height = image.height
            kwargs.update(width=width, height=height, image=image)
        else:
            kwargs["width"] = width or 1024
            kwargs["height"] = height or 1024

        cur_w = kwargs.get("width", 1024)
        cur_h = kwargs.get("height", 1024)

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            if mask_image.size != (cur_w, cur_h):
                mask_image = mask_image.resize((cur_w, cur_h), Image.Resampling.LANCZOS)
            kwargs["mask_image"] = mask_image

        adapters = getattr(self.pipeline, "get_list_adapters", lambda: {})() or {}
        has_control_adapter = "controlnet" in (adapters.get("transformer") or [])
        if use_controlnet or has_control_adapter:
            control_image = kwargs.get("control_image")
            if control_image and isinstance(control_image, Image.Image):
                if control_image.size != (cur_w, cur_h):
                    control_image = control_image.resize((cur_w, cur_h), Image.Resampling.LANCZOS)
                kwargs["control_image"] = control_image

        return kwargs

    def __call__(
        self,
        *args: Union[str, int, float, bool],
        task: str = "txt2img",
        use_controlnet: bool = False,
        **kwargs: Union[str, int, float, bool, Image.Image, None, Callable],
    ) -> BaseOutput:
        call_kwargs: InferenceKwargs = self.check_inference_kwargs(
            dict(kwargs), use_controlnet=use_controlnet  # type: ignore
        )
        prompt = call_kwargs.get("prompt")
        if prompt and call_kwargs.get("prompt_embeds") is None:
            embeds = self._get_cached_flux_embeds(prompt)
            if embeds:
                call_kwargs.update(embeds)  # type: ignore
                call_kwargs["prompt"] = None

        return super().__call__(*args, task=task, use_controlnet=use_controlnet, **call_kwargs)

    def load_controlnet(
        self,
        controlnet_model: Optional[str] = None,
        **kwargs: Union[str, float, int, bool],
    ) -> None:
        adapters = getattr(self.pipeline, "get_list_adapters", lambda: {})() or {}
        if "controlnet" not in (adapters.get("transformer") or []):
            model_url = (
                controlnet_model
                or "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors?download=true"
            )
            self.set_lora(model_url, "controlnet", 1.0)

    def offload_controlnet(self) -> None:
        adapters = getattr(self.pipeline, "get_list_adapters", lambda: {})() or {}
        if "controlnet" in (adapters.get("transformer") or []):
            self.delete_adapters("controlnet")
        free_memory_to_system()

    def clear(self) -> None:
        self._embed_cache.clear()
        super().clear()

    @classmethod
    def from_url(
        cls,
        url: Optional[str] = None,
        quantization: str = "none",
        text_encoder_mode: str = "default",
        delete_after_load: bool = False,
        **kwargs: Union[str, float, int, bool],
    ) -> "FluxPipelineEngine":
        raw_pipe = load_flux_pipeline(
            url,
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
