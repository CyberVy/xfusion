from __future__ import annotations

from typing import Dict, Optional

from ..models.inspect import apply_execution_device, use_gpu
from .contracts import ModelLoadSettings
from .families import FLUX_FAMILY, MOSS_FAMILY, SD_FAMILY, find_model_family
from .runtime import LoadedModel
from .schedulers import DIFFUSION_SCHEDULER_MAP


def load_model(
    settings: ModelLoadSettings,
    gpu_id: int = 0,
    download_kwargs: Optional[Dict[str, str]] = None,
) -> LoadedModel:
    """Load one model onto one GPU and wrap it in the shared runtime."""
    hint = (settings.model_version or "").strip().lower()
    downloads = dict(download_kwargs or {})
    if hint:
        family = find_model_family(hint)
        requested_version = hint
    elif "moss" in settings.source.lower():
        family = MOSS_FAMILY
        requested_version = "moss"
    elif "flux" in settings.source.lower():
        family = FLUX_FAMILY
        requested_version = "flux"
    else:
        family = SD_FAMILY
        requested_version = ""

    with use_gpu(gpu_id):
        pipeline, resolved_version = family.load_weights(settings, requested_version, downloads)
        if hint:
            resolved_version = hint
        elif not resolved_version:
            resolved_version = family.version_of(pipeline) or "xl"

        features = family.features_for(resolved_version)
        apply_execution_device(pipeline, memory=settings.memory, gpu_id=gpu_id)
    model = LoadedModel(pipeline, family, settings, features, gpu_id)
    if downloads:
        model.set_download_kwargs(**downloads)
    return model


__all__ = [
    "LoadedModel",
    "ModelLoadSettings",
    "DIFFUSION_SCHEDULER_MAP",
    "load_model",
]
