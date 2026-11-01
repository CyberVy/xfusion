from typing import Optional, Union
from diffusers import DiffusionPipeline

from .base import BasePipelineEngine
from .pipelines.sd import SDPipelineEngine
from .pipelines.flux import FluxPipelineEngine
from .schedulers import DIFFUSION_SCHEDULER_MAP


def load_engine(
    pipeline_or_model: Optional[Union[str, DiffusionPipeline]] = None,
    **kwargs: Union[str, float, int, bool, dict],
) -> BasePipelineEngine:
    """Factory function to load and wrap Diffusers pipelines into a standard Inference Engine."""
    model_version = str(kwargs.pop("model_version", "")).lower()
    quantization = str(kwargs.pop("quantization", "none") or "none").lower()
    text_encoder_mode = str(kwargs.pop("text_encoder_mode", "default") or "default").lower()
    offload_mode = str(kwargs.pop("offload_mode", "none") or "none").lower()
    delete_after_load = bool(kwargs.pop("delete_after_load", False))
    gpu_id = int(kwargs.pop("gpu_id", 0))
    download_kwargs = kwargs.get("download_kwargs", {})

    if model_version == "flux" or (
        not model_version
        and isinstance(pipeline_or_model, str)
        and "flux" in pipeline_or_model.lower()
    ):
        if isinstance(pipeline_or_model, str):
            engine = FluxPipelineEngine.from_url(
                pipeline_or_model,
                quantization=quantization,
                text_encoder_mode=text_encoder_mode,
                delete_after_load=delete_after_load,
                **kwargs,
            )
            engine.model_name = pipeline_or_model
        elif isinstance(pipeline_or_model, DiffusionPipeline):
            engine = FluxPipelineEngine(pipeline_or_model)
        else:
            engine = FluxPipelineEngine(None)
    elif model_version in ["", "1", "1.5", "2", "3", "3.5", "xl", "sdxl", "pony"]:
        if isinstance(pipeline_or_model, str):
            if model_version in ["sdxl", "pony"]:
                model_version = "xl"
            if model_version == "3.5":
                model_version = "3"
            engine = SDPipelineEngine.from_url(
                pipeline_or_model,
                model_version=model_version,
                quantization=quantization,
                text_encoder_mode=text_encoder_mode,
                delete_after_load=delete_after_load,
                **kwargs,
            )
            engine.model_name = pipeline_or_model
        elif isinstance(pipeline_or_model, DiffusionPipeline):
            engine = SDPipelineEngine(pipeline_or_model)
        else:
            engine = SDPipelineEngine(None)
    else:
        raise ValueError(f"Model version '{model_version}' is not supported yet.")

    if isinstance(download_kwargs, dict):
        engine.set_download_kwargs(**download_kwargs)

    engine.configure_execution(offload_mode=offload_mode, gpu_id=gpu_id)
    return engine


__all__ = [
    "BasePipelineEngine",
    "SDPipelineEngine",
    "FluxPipelineEngine",
    "DIFFUSION_SCHEDULER_MAP",
    "load_engine",
]
