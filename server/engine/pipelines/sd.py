import os
from typing import Dict, Any, Optional
from PIL import Image

from diffusers import (
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
)

from ..base import BasePipelineEngine
from ..schedulers import DIFFUSION_SCHEDULER_MAP
from ..prompt import get_embeds_from_pipeline, skip_clip_layer
from ...infra.image import normalize_image
from ...models.loaders.sd import load_stable_diffusion_pipeline, load_stable_diffusion_controlnet
from ...models.constants import (
    default_sd_v1_v2_controlnet_model_url,
    default_sd_xl_controlnet_model_url,
)
from ...models.inspect import get_tokenizers_and_text_encoders_from_pipeline

SD_PIPELINE_MAP = {
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
    ),
}


class SDPipelineEngine(BasePipelineEngine):
    pipeline_map = SD_PIPELINE_MAP
    scheduler_map = DIFFUSION_SCHEDULER_MAP

    def __init__(self, pipeline: Any, init_sub_pipelines: bool = True):
        super().__init__(pipeline, init_sub_pipelines=init_sub_pipelines)
        if pipeline is not None:
            self.tokenizers, self.text_encoders = get_tokenizers_and_text_encoders_from_pipeline(self.pipeline)
        else:
            self.tokenizers, self.text_encoders = [], []

    def get_embeds(self, prompt: Any, negative_prompt: Any, clip_skip: Optional[int] = None) -> Dict[str, Any]:
        skip_clip_layer(self.text_encoders, clip_skip)
        embeds = get_embeds_from_pipeline(self, prompt, negative_prompt)
        skip_clip_layer(self.text_encoders, 0)
        return embeds

    def check_inference_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if kwargs.get("negative_prompt") is None:
            kwargs["negative_prompt"] = ""

        width = kwargs.get("width")
        height = kwargs.get("height")
        image = kwargs.get("image")

        if image and isinstance(image, Image.Image):
            if width is None and height is None:
                width = image.width - (image.width % 8)
                height = image.height - (image.height % 8)
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                target_w = width or (1024 if self.model_version in ["xl", "3"] else 512)
                target_h = height or (1024 if self.model_version in ["xl", "3"] else 512)
                image = normalize_image(image, target_w * target_h, scale_divisor=8)
                width = image.width
                height = image.height
            kwargs.update(width=width, height=height, image=image)
        else:
            default_dim = 1024 if self.model_version in ["xl", "3"] else 512
            kwargs["width"] = width or default_dim
            kwargs["height"] = height or default_dim

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            if mask_image.size != (kwargs["width"], kwargs["height"]):
                mask_image = mask_image.resize((kwargs["width"], kwargs["height"]), Image.Resampling.LANCZOS)
            kwargs["mask_image"] = mask_image

        ip_adapter_image = kwargs.get("ip_adapter_image")
        if ip_adapter_image and isinstance(ip_adapter_image, Image.Image):
            ip_adapter_image = normalize_image(ip_adapter_image, kwargs["width"] * kwargs["height"], scale_divisor=8)
            kwargs["ip_adapter_image"] = ip_adapter_image

        if "controlnet" in self.pipeline.__class__.__name__.lower():
            control_image = kwargs.get("control_image")
            if control_image and isinstance(control_image, Image.Image):
                w = kwargs["width"]
                h = kwargs["height"]
                if control_image.size != (w, h):
                    control_image = control_image.resize((w, h), Image.Resampling.LANCZOS)
                kwargs["control_image"] = control_image

        return kwargs

    def __call__(self, **kwargs: Any) -> Any:
        kwargs = self.check_inference_kwargs(kwargs)

        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt")
        prompt_str = f"{prompt or ''} {negative_prompt or ''}"

        skipped_lora_dict = {}
        for lora, weight in self.lora_dict.items():
            if lora not in prompt_str:
                skipped_lora_dict[lora] = weight
                self.set_lora_strength(lora, 0)

        if len(skipped_lora_dict) == len(self.lora_dict) and self.lora_dict:
            self.disable_lora()

        try:
            # Handle prompt embeddings
            embeds = self.get_embeds(prompt, negative_prompt, kwargs.get("clip_skip"))
            if embeds:
                kwargs.update(embeds)
                kwargs["prompt"] = None
                kwargs["negative_prompt"] = None
                kwargs.pop("clip_skip", None)

            return self.pipeline(**kwargs)
        finally:
            if len(skipped_lora_dict) == len(self.lora_dict) and self.lora_dict:
                self.enable_lora()
            if skipped_lora_dict:
                for lora, weight in skipped_lora_dict.items():
                    self.set_lora_strength(lora, weight)

    def load_controlnet(self, controlnet_model: Optional[str] = None, **kwargs: Any) -> None:
        if self._controlnet is not None:
            return

        download_kwargs = self.download_kwargs.copy()
        directory = download_kwargs.get("directory", "./")
        engine_cls = self.__class__
        components = getattr(self.pipeline, "components", {})

        if self.model_version in ["1", "1.5"]:
            directory = os.path.join(directory, "controlnet_sd_v1_v2")
            download_kwargs["directory"] = directory
            model_url = controlnet_model or default_sd_v1_v2_controlnet_model_url

            self._controlnet = load_stable_diffusion_controlnet(
                model_url, self.model_version, download_kwargs=download_kwargs, **kwargs
            )
            if self.device is not None:
                self._controlnet = self._controlnet.to(self.device)

            self.text_to_image_controlnet_pipeline = engine_cls(
                StableDiffusionControlNetPipeline(**components, controlnet=self._controlnet),
                init_sub_pipelines=False,
            )
            self.image_to_image_controlnet_pipeline = engine_cls(
                StableDiffusionControlNetImg2ImgPipeline(**components, controlnet=self._controlnet),
                init_sub_pipelines=False,
            )
            self.inpainting_controlnet_pipeline = engine_cls(
                StableDiffusionControlNetInpaintPipeline(**components, controlnet=self._controlnet),
                init_sub_pipelines=False,
            )
        elif self.model_version in ["xl", "pony"]:
            directory = os.path.join(directory, "controlnet_sdxl")
            download_kwargs["directory"] = directory
            model_url = controlnet_model or default_sd_xl_controlnet_model_url

            self._controlnet = load_stable_diffusion_controlnet(
                model_url, self.model_version, download_kwargs=download_kwargs, **kwargs
            )
            if self.device is not None:
                self._controlnet = self._controlnet.to(self.device)

            self.text_to_image_controlnet_pipeline = engine_cls(
                StableDiffusionXLControlNetPipeline(**components, controlnet=self._controlnet),
                init_sub_pipelines=False,
            )
            self.image_to_image_controlnet_pipeline = engine_cls(
                StableDiffusionXLControlNetImg2ImgPipeline(**components, controlnet=self._controlnet),
                init_sub_pipelines=False,
            )
            self.inpainting_controlnet_pipeline = engine_cls(
                StableDiffusionXLControlNetInpaintPipeline(**components, controlnet=self._controlnet),
                init_sub_pipelines=False,
            )
        else:
            raise NotImplementedError(f"ControlNet for model version {self.model_version} is not supported.")

        self.sub_pipelines["text_to_image_controlnet_pipeline"] = self.text_to_image_controlnet_pipeline
        self.sub_pipelines["image_to_image_controlnet_pipeline"] = self.image_to_image_controlnet_pipeline
        self.sub_pipelines["inpainting_controlnet_pipeline"] = self.inpainting_controlnet_pipeline
        self._sync_sub_pipelines()

    @classmethod
    def from_url(
        cls,
        url: Optional[str] = None,
        model_version: Optional[str] = None,
        init_sub_pipelines: bool = True,
        **kwargs: Any
    ):
        raw_pipe = load_stable_diffusion_pipeline(model=url, model_version=model_version, **kwargs)
        engine = cls(raw_pipe, init_sub_pipelines=init_sub_pipelines)
        if url:
            engine.model_name = url
        return engine
