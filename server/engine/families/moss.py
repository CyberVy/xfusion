from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from diffusers import DiffusionPipeline

from ...models.loaders.moss import load_moss_pipeline
from ..contracts import FeatureNotSupported, GenerationRequest, ModelFeatures, ModelLoadSettings


class MossFamily:
    prompt_activates_lora = False
    embed_cache_limit = 4

    def __init__(self) -> None:
        self._pipeline_cls: Type[DiffusionPipeline] = DiffusionPipeline

    def matches(self, model_version: str) -> bool:
        return model_version == "moss"

    def features_for(self, model_version: str) -> ModelFeatures:
        return ModelFeatures(
            model_version="moss",
            tasks=frozenset({"txt2img"}),
            adapters=frozenset(),
            schedulers=(),
            default_size=1024,
            default_guidance=3.0,
            accepts_negative_prompt=True,
            accepts_clip_skip=False,
            size_divisor=16,
        )

    def version_of(self, pipeline: DiffusionPipeline) -> Optional[str]:
        if "moss" in pipeline.__class__.__name__.lower():
            return "moss"
        return None

    def generator_device(self, model) -> str:
        return "cpu"

    def load_weights(
        self,
        settings: ModelLoadSettings,
        model_version: str,
        download_kwargs: Dict[str, str],
    ) -> Tuple[DiffusionPipeline, str]:
        pipe = load_moss_pipeline(
            settings.source,
            download_kwargs=download_kwargs,
            backbone=settings.backbone,
            text_encoder=settings.text_encoder,
            delete_file_after_load=settings.delete_file_after_load,
        )
        self._pipeline_cls = type(pipe)
        return pipe, "moss"

    def pipeline_for_task(self, model_version: str, task: str, controlnet: bool) -> Type[DiffusionPipeline]:
        if controlnet or task != "txt2img":
            raise ValueError(f"Task '{task}' is not supported.")
        return self._pipeline_cls

    def build_generation_inputs(self, model, request: GenerationRequest) -> Dict[str, object]:
        return {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
            "uncond_skip_layers": [22],
        }

    def load_controlnet(self, model, source: Optional[str]) -> None:
        raise FeatureNotSupported("controlnet")

    def clear_controlnet(self, model) -> None:
        return None

    def controlnet_is_loaded(self, model) -> bool:
        return False
