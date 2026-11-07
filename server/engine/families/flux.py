from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from diffusers import (
    DiffusionPipeline,
    FluxControlImg2ImgPipeline,
    FluxControlInpaintPipeline,
    FluxControlPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxPipeline,
)

from ...infra.memory import free_memory_to_system
from ...models.loaders.flux import load_flux_pipeline
from ..contracts import GenerationRequest, ModelFeatures, ModelLoadSettings

_DEFAULT_CONTROLNET = (
    "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/"
    "flux1-canny-dev-lora.safetensors?download=true"
)

_PIPELINES: Dict[Tuple[str, bool], Type[DiffusionPipeline]] = {
    ("txt2img", False): FluxPipeline,
    ("img2img", False): FluxImg2ImgPipeline,
    ("inpaint", False): FluxInpaintPipeline,
    ("txt2img", True): FluxControlPipeline,
    ("img2img", True): FluxControlImg2ImgPipeline,
    ("inpaint", True): FluxControlInpaintPipeline,
}


class FluxFamily:
    prompt_activates_lora = False
    embed_cache_limit = 6

    def matches(self, model_version: str) -> bool:
        return model_version == "flux"

    def features_for(self, model_version: str) -> ModelFeatures:
        return ModelFeatures(
            model_version="flux",
            tasks=frozenset({"txt2img", "img2img", "inpaint"}),
            adapters=frozenset({"lora", "controlnet"}),
            schedulers=(),
            default_size=1024,
            default_guidance=3.5,
            accepts_negative_prompt=False,
            accepts_clip_skip=False,
            size_divisor=16,
        )

    def version_of(self, pipeline: DiffusionPipeline) -> Optional[str]:
        if pipeline.__class__.__name__.startswith("Flux"):
            return "flux"
        return None

    def load_weights(
        self,
        settings: ModelLoadSettings,
        model_version: str,
        download_kwargs: Dict[str, str],
    ) -> Tuple[DiffusionPipeline, str]:
        pipe = load_flux_pipeline(
            settings.source,
            download_kwargs=download_kwargs,
            backbone=settings.backbone,
            text_encoder=settings.text_encoder,
            dtype=settings.dtype,
            delete_file_after_load=settings.delete_file_after_load,
        )
        return pipe, "flux"

    def pipeline_for_task(self, model_version: str, task: str, controlnet: bool) -> Type[DiffusionPipeline]:
        pipeline_cls = _PIPELINES.get((task, controlnet))
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
            kwargs["control_image"] = request.control_image
            if self.controlnet_is_loaded(model):
                scale = request.control_scale if request.control_scale is not None else 0.5
                model.set_lora_strength("controlnet", scale)

        embeds = self._encode(model, request.prompt)
        if embeds:
            kwargs.update(embeds)
        else:
            kwargs["prompt"] = request.prompt
        return kwargs

    def load_controlnet(self, model, source: Optional[str]) -> None:
        if not self.controlnet_is_loaded(model):
            model.set_lora(source or _DEFAULT_CONTROLNET, "controlnet", 1.0)

    def clear_controlnet(self, model) -> None:
        if self.controlnet_is_loaded(model):
            model.delete_lora("controlnet")
        free_memory_to_system()

    def controlnet_is_loaded(self, model) -> bool:
        adapters = getattr(model.pipeline, "get_list_adapters", lambda: {})() or {}
        if not isinstance(adapters, dict):
            return False
        return "controlnet" in (adapters.get("transformer") or [])

    def _encode(self, model, prompt: Optional[str]) -> Dict[str, object]:
        if not prompt or model.pipeline is None or not hasattr(model.pipeline, "encode_prompt"):
            return {}
        active_loras = tuple(sorted((name, weight) for name, weight in model.loras.items() if weight != 0))
        cache_key = (str(prompt), model.loras_enabled, active_loras)

        def encode():
            try:
                prompt_embeds, pooled_prompt_embeds, _text_ids = model.pipeline.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    device=model.execution_device,
                )
            except Exception:
                return {}
            if prompt_embeds is None or pooled_prompt_embeds is None:
                return {}
            free_memory_to_system()
            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
            }

        return model.cached_embeds(cache_key, encode)
