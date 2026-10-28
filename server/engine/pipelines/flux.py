from typing import Dict, Any, Optional
from PIL import Image

from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxControlPipeline,
    FluxControlImg2ImgPipeline,
    FluxControlInpaintPipeline,
)

from ..base import BasePipelineEngine
from ...infra.image import normalize_image
from ...models.loaders.flux import load_flux_pipeline

FLUX_PIPELINE_MAP = {
    "flux": (
        FluxPipeline,
        FluxImg2ImgPipeline,
        FluxInpaintPipeline,
        FluxControlPipeline,
        FluxControlImg2ImgPipeline,
        FluxControlInpaintPipeline,
    )
}


class FluxPipelineEngine(BasePipelineEngine):
    pipeline_map = FLUX_PIPELINE_MAP

    def check_inference_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            if mask_image.size != (kwargs["width"], kwargs["height"]):
                mask_image = mask_image.resize((kwargs["width"], kwargs["height"]), Image.Resampling.LANCZOS)
            kwargs["mask_image"] = mask_image

        adapters = getattr(self.pipeline, "get_list_adapters", lambda: {})() or {}
        if "controlnet" in (adapters.get("transformer") or []):
            control_image = kwargs.get("control_image")
            if control_image and isinstance(control_image, Image.Image):
                w = kwargs["width"]
                h = kwargs["height"]
                if control_image.size != (w, h):
                    control_image = control_image.resize((w, h), Image.Resampling.LANCZOS)
                kwargs["control_image"] = control_image

        return kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = self.check_inference_kwargs(kwargs)
        return self.pipeline(*args, **kwargs)

    def load_controlnet(self, controlnet_model: Optional[str] = None, **kwargs: Any) -> None:
        adapters = getattr(self.pipeline, "get_list_adapters", lambda: {})() or {}
        if "controlnet" not in (adapters.get("transformer") or []):
            model_url = (
                controlnet_model
                or "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors?download=true"
            )
            self.set_lora(model_url, "controlnet", 1.0)
            components = getattr(self.pipeline, "components", {})
            components.pop("image_encoder", None)
            components.pop("feature_extractor", None)

            engine_cls = self.__class__
            self.text_to_image_controlnet_pipeline = engine_cls(
                FluxControlPipeline(**components), init_sub_pipelines=False
            )
            self.image_to_image_controlnet_pipeline = engine_cls(
                FluxControlImg2ImgPipeline(**components), init_sub_pipelines=False
            )
            self.inpainting_controlnet_pipeline = engine_cls(
                FluxControlInpaintPipeline(**components), init_sub_pipelines=False
            )

            self.sub_pipelines["text_to_image_controlnet_pipeline"] = self.text_to_image_controlnet_pipeline
            self.sub_pipelines["image_to_image_controlnet_pipeline"] = self.image_to_image_controlnet_pipeline
            self.sub_pipelines["inpainting_controlnet_pipeline"] = self.inpainting_controlnet_pipeline
            self._sync_sub_pipelines()

    def offload_controlnet(self) -> None:
        print("Offload controlnet for FLUX is a no-op.")

    @classmethod
    def from_url(cls, url: Optional[str] = None, init_sub_pipelines: bool = True, **kwargs: Any):
        raw_pipe = load_flux_pipeline(url, **kwargs)
        engine = cls(raw_pipe, init_sub_pipelines=init_sub_pipelines)
        if url:
            engine.model_name = url
        return engine
