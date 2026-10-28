from typing import Any

try:
    from .base import BasePipelineEngine
    from .pipelines.sd import SDPipelineEngine
    from .pipelines.flux import FluxPipelineEngine
    from .schedulers import DIFFUSION_SCHEDULER_MAP
except (ImportError, AttributeError):
    BasePipelineEngine = None
    SDPipelineEngine = None
    FluxPipelineEngine = None
    DIFFUSION_SCHEDULER_MAP = {}


def load_engine(pipeline_or_model: Any = None, **kwargs: Any) -> Any:
    """Factory function to load and wrap Diffusers pipelines into a standard Inference Engine."""
    if BasePipelineEngine is None:
        raise RuntimeError("Inference engine requires PyTorch and Diffusers dependencies to be installed.")

    model_version = str(kwargs.pop("model_version", "")).lower()
    download_kwargs = kwargs.get("download_kwargs", {})

    if model_version in ["", "1", "1.5", "2", "3", "3.5", "xl", "sdxl", "pony"]:
        if isinstance(pipeline_or_model, str):
            if model_version in ["sdxl", "pony"]:
                model_version = "xl"
            if model_version == "3.5":
                model_version = "3"
            engine = SDPipelineEngine.from_url(pipeline_or_model, model_version=model_version, **kwargs)
            engine.model_name = pipeline_or_model
        else:
            engine = SDPipelineEngine(pipeline_or_model)

        if isinstance(download_kwargs, dict):
            engine.set_download_kwargs(**download_kwargs)
        return engine

    elif model_version in ["flux"]:
        if isinstance(pipeline_or_model, str):
            engine = FluxPipelineEngine.from_url(pipeline_or_model, **kwargs)
            engine.model_name = pipeline_or_model
        else:
            engine = FluxPipelineEngine(pipeline_or_model)

        if isinstance(download_kwargs, dict):
            engine.set_download_kwargs(**download_kwargs)
        return engine

    else:
        raise ValueError(f"Model version '{model_version}' is not supported yet.")


__all__ = [
    "BasePipelineEngine",
    "SDPipelineEngine",
    "FluxPipelineEngine",
    "DIFFUSION_SCHEDULER_MAP",
    "load_engine",
]
