from fastapi import APIRouter, HTTPException
from ..schemas import (
    ModelLoadRequest,
    LoraLoadRequest,
    LoraDeleteRequest,
    SchedulerSetRequest
)
from ..manager import manager

router = APIRouter(prefix="/api/v1", tags=["Models & Adapters"])


@router.post("/models/load")
async def load_model(req: ModelLoadRequest):
    """Load or switch the base diffusion model."""
    try:
        res = await manager.load_model(req.model, req.model_version)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/current")
async def get_current_model():
    """Get the currently loaded base model."""
    return {
        "current_model": manager.current_model,
        "model_version": manager.model_version,
    }


@router.post("/lora/load")
async def load_lora(req: LoraLoadRequest):
    """Load or update a LoRA adapter."""
    try:
        res = await manager.set_lora(req.url, req.lora_name, req.strength)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lora/delete")
async def delete_lora(req: LoraDeleteRequest):
    """Delete a loaded LoRA adapter."""
    try:
        res = await manager.delete_lora(req.lora_name)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lora/list")
async def list_loras():
    """List all active LoRA adapters."""
    return {"loras": manager.get_loras()}


@router.post("/scheduler/set")
async def set_scheduler(req: SchedulerSetRequest):
    """Change the sampler/scheduler on the active pipeline."""
    try:
        res = await manager.set_scheduler(req.scheduler, req.pipeline_type)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
