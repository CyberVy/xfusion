from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Type

from diffusers import (
    DiffusionPipeline,
    StableDiffusion3ControlNetInpaintingPipeline,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from ...models.constants import (
    default_sd_v1_v2_controlnet_model_url,
    default_sd_xl_controlnet_model_url,
)
from ...models.inspect import get_tokenizers_and_text_encoders_from_pipeline
from ...models.loaders.sd import load_stable_diffusion_controlnet, load_stable_diffusion_pipeline
from ..prompt import encode_clip_prompt, skip_clip_layer
from ..schedulers import DIFFUSION_SCHEDULER_MAP
from ..contracts import GenerationRequest, ModelFeatures, ModelLoadSettings

TaskKey = Tuple[str, bool]
PipelineTable = Dict[TaskKey, Type[DiffusionPipeline]]

_SD15: PipelineTable = {
    ("txt2img", False): StableDiffusionPipeline,
    ("img2img", False): StableDiffusionImg2ImgPipeline,
    ("inpaint", False): StableDiffusionInpaintPipeline,
    ("txt2img", True): StableDiffusionControlNetPipeline,
    ("img2img", True): StableDiffusionControlNetImg2ImgPipeline,
    ("inpaint", True): StableDiffusionControlNetInpaintPipeline,
}
_SDXL: PipelineTable = {
    ("txt2img", False): StableDiffusionXLPipeline,
    ("img2img", False): StableDiffusionXLImg2ImgPipeline,
    ("inpaint", False): StableDiffusionXLInpaintPipeline,
    ("txt2img", True): StableDiffusionXLControlNetPipeline,
    ("img2img", True): StableDiffusionXLControlNetImg2ImgPipeline,
    ("inpaint", True): StableDiffusionXLControlNetInpaintPipeline,
}
_SD3: PipelineTable = {
    ("txt2img", False): StableDiffusion3Pipeline,
    ("img2img", False): StableDiffusion3Img2ImgPipeline,
    ("inpaint", False): StableDiffusion3InpaintPipeline,
    ("txt2img", True): StableDiffusion3ControlNetPipeline,
    ("img2img", True): StableDiffusion3ControlNetPipeline,
    ("inpaint", True): StableDiffusion3ControlNetInpaintingPipeline,
}

_SCHEDULERS = tuple(DIFFUSION_SCHEDULER_MAP)


def _table_for(model_version: str) -> PipelineTable:
    if model_version in ("1", "1.5", "2"):
        return _SD15
    if model_version in ("xl", "pony"):
        return _SDXL
    if model_version in ("3", "3.5"):
        return _SD3
    raise ValueError(f"Model version '{model_version}' is not supported.")


class SdFamily:
    prompt_activates_lora = True
    embed_cache_limit = 8

    def matches(self, model_version: str) -> bool:
        return model_version in ("1", "1.5", "2", "xl", "pony", "3", "3.5")

    def features_for(self, model_version: str) -> ModelFeatures:
        version = model_version or "xl"
        adapters = {"lora", "ip_adapter"}
        if version in ("1", "1.5", "xl", "pony"):
            adapters.add("controlnet")
        return ModelFeatures(
            model_version=version,
            tasks=frozenset({"txt2img", "img2img", "inpaint"}),
            adapters=frozenset(adapters),
            schedulers=_SCHEDULERS,
            default_size=512 if version in ("1", "1.5", "2") else 1024,
            default_guidance=7.0,
            accepts_negative_prompt=True,
            accepts_clip_skip=True,
            size_divisor=8,
        )

    def version_of(self, pipeline: DiffusionPipeline) -> Optional[str]:
        name = pipeline.__class__.__name__
        if "StableDiffusion3" in name:
            return "3"
        if "XL" in name:
            return "xl"
        if name.startswith("StableDiffusion"):
            return "1.5"
        return None

    def load_weights(
        self,
        settings: ModelLoadSettings,
        model_version: str,
        download_kwargs: Dict[str, str],
    ) -> Tuple[DiffusionPipeline, str]:
        return load_stable_diffusion_pipeline(
            model=settings.source,
            model_version=model_version or None,
            download_kwargs=download_kwargs,
            backbone=settings.backbone,
            text_encoder=settings.text_encoder,
            dtype=settings.dtype,
            delete_file_after_load=settings.delete_file_after_load,
        )

    def pipeline_for_task(self, model_version: str, task: str, controlnet: bool) -> Type[DiffusionPipeline]:
        table = _table_for(model_version)
        pipeline_cls = table.get((task, controlnet))
        if pipeline_cls is None:
            raise ValueError(f"Task '{task}' is not supported.")
        return pipeline_cls

    def build_generation_inputs(self, model, request: GenerationRequest) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
        }
        if request.task != "txt2img" and request.image is not None:
            kwargs["image"] = request.image
        if request.task != "txt2img" and request.strength is not None:
            kwargs["strength"] = request.strength
        if request.mask is not None:
            kwargs["mask_image"] = request.mask
        if request.control_image is not None:
            if request.task == "txt2img":
                kwargs["image"] = request.control_image
            else:
                kwargs["control_image"] = request.control_image
            if request.control_scale is not None:
                kwargs["controlnet_conditioning_scale"] = request.control_scale
        if request.reference_image is not None:
            kwargs["ip_adapter_image"] = request.reference_image

        embeds = self._encode(model, request)
        if embeds:
            kwargs.update(embeds)
        else:
            kwargs["prompt"] = request.prompt
            kwargs["negative_prompt"] = request.negative_prompt
        return kwargs

    def load_controlnet(self, model, source: Optional[str]) -> None:
        if model._controlnet is not None:
            return
        version = model.features.model_version
        download_kwargs = dict(model.download_kwargs)
        directory = download_kwargs.get("directory", "./")
        if version in ("1", "1.5"):
            download_kwargs["directory"] = os.path.join(directory, "controlnet_sd_v1_v2")
            model_url = source or default_sd_v1_v2_controlnet_model_url
        elif version in ("xl", "pony"):
            download_kwargs["directory"] = os.path.join(directory, "controlnet_sdxl")
            model_url = source or default_sd_xl_controlnet_model_url
        else:
            raise NotImplementedError(f"ControlNet for model version {version} is not supported.")

        controlnet = load_stable_diffusion_controlnet(
            model_url,
            version,
            download_kwargs=download_kwargs,
            delete_file_after_load=model.settings.delete_file_after_load,
        )
        model._controlnet = controlnet
        model._task_pipes.clear()
        device = model.execution_device
        memory = model.settings.memory
        if memory == "gpu" and device is not None:
            model._controlnet = controlnet.to(device)
        elif memory == "offload" and device is not None:
            try:
                from accelerate import cpu_offload_with_hook

                _, hook = cpu_offload_with_hook(controlnet, execution_device=device)
                if model.pipeline is not None and hasattr(model.pipeline, "_all_hooks"):
                    model.pipeline._all_hooks.append(hook)
            except Exception:
                model._controlnet = controlnet.to(device)
        elif memory == "sequential" and device is not None:
            try:
                from accelerate import cpu_offload

                cpu_offload(controlnet, execution_device=device)
            except Exception:
                model._controlnet = controlnet.to(device)

    def clear_controlnet(self, model) -> None:
        from ...infra.memory import delete, free_memory_to_system

        model._task_pipes.clear()
        if model._controlnet is not None:
            delete(model._controlnet)
            model._controlnet = None
        free_memory_to_system()

    def controlnet_is_loaded(self, model) -> bool:
        return model._controlnet is not None

    def _encode(self, model, request: GenerationRequest) -> Dict[str, object]:
        if model.pipeline is None:
            return {}
        active_loras = tuple(sorted((name, weight) for name, weight in model.loras.items() if weight != 0))
        cache_key = (
            str(request.prompt),
            str(request.negative_prompt),
            request.clip_skip or 0,
            model.loras_enabled,
            active_loras,
        )
        tokenizers, text_encoders = get_tokenizers_and_text_encoders_from_pipeline(model.pipeline)

        def encode():
            skip_clip_layer(text_encoders, request.clip_skip)
            try:
                return encode_clip_prompt(
                    model.pipeline,
                    model.features.model_version,
                    request.prompt,
                    request.negative_prompt,
                    model.execution_device,
                )
            finally:
                skip_clip_layer(text_encoders, 0)

        return model.cached_embeds(cache_key, encode)
