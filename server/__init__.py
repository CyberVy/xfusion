import os

_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

from .models import (
    load_stable_diffusion_pipeline,
    load_stable_diffusion_controlnet,
    load_flux_pipeline,
    load_moss_pipeline,
    infer_model_version,
)
from .engine import load_model, LoadedModel

__all__ = [
    "load_model",
    "LoadedModel",
    "load_stable_diffusion_pipeline",
    "load_stable_diffusion_controlnet",
    "load_flux_pipeline",
    "load_moss_pipeline",
    "infer_model_version",
]
