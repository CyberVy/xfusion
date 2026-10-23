from fastapi import APIRouter
from ..schemas import SystemInfoResponse
from ..manager import manager, AVAILABLE_SCHEDULERS
from ...const import GPU_COUNT, GPU_NAME

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
        loras=manager.get_loras(),
        schedulers=AVAILABLE_SCHEDULERS,
    )
