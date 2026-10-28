import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from .models import (
        load_stable_diffusion_pipeline,
        load_stable_diffusion_controlnet,
        load_flux_pipeline,
        infer_model_version,
    )
    from .engine import (
        load_engine,
        SDPipelineEngine,
        FluxPipelineEngine,
    )
except (ImportError, AttributeError):
    load_stable_diffusion_pipeline = None
    load_stable_diffusion_controlnet = None
    load_flux_pipeline = None
    infer_model_version = None
    load_engine = None
    SDPipelineEngine = None
    FluxPipelineEngine = None

__all__ = [
    "load_engine",
    "SDPipelineEngine",
    "FluxPipelineEngine",
    "load_stable_diffusion_pipeline",
    "load_stable_diffusion_controlnet",
    "load_flux_pipeline",
    "infer_model_version",
]
