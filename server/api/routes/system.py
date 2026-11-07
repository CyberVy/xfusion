import platform
import sys

from fastapi import APIRouter
from ..schemas import SystemInfoResponse
from ..manager import manager
from ...infra.config import GPU_COUNT, GPU_NAME

router = APIRouter(prefix="/api/v1/system", tags=["System"])


@router.get("/health")
async def health_check():
    """Simple ping endpoint for client connection and health check."""
    return {
        "status": "healthy",
        "service": "xfusion-api",
        "model_loaded": manager.current_model is not None,
    }


@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Retrieve detailed hardware, model, and scheduler state."""
    return SystemInfoResponse(
        status="healthy",
        gpu_count=GPU_COUNT,
        gpu_names=GPU_NAME,
        current_model=manager.current_model,
        model_version=manager.model_version,
        backbone=manager.settings.backbone,
        text_encoder=manager.settings.text_encoder,
        memory=manager.settings.memory,
        dtype=manager.settings.dtype,
        adapters=sorted(manager.models[0].features.adapters) if manager.models else [],
        tasks=sorted(manager.models[0].features.tasks) if manager.models else [],
        schedulers=manager.get_schedulers(),
        default_size=manager.models[0].features.default_size if manager.models else 1024,
        default_guidance=manager.models[0].features.default_guidance if manager.models else 7.0,
        accepts_negative_prompt=manager.models[0].features.accepts_negative_prompt if manager.models else True,
        accepts_clip_skip=manager.models[0].features.accepts_clip_skip if manager.models else True,
        loras=manager.get_loras(),
        loras_enabled=manager.loras_enabled,
        controlnet_loaded=manager.controlnet_is_loaded(),
        ip_adapter=manager.ip_adapter_info(),
        python_version=sys.version.split()[0],
        os_platform=platform.platform(),
    )

