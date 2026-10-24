from fastapi import APIRouter, HTTPException
from ..schemas import (
    ModelLoadRequest,
    LoraLoadRequest,
    LoraDeleteRequest,
    LoraStrengthRequest,
    LoraEnabledRequest,
    IpAdapterLoadRequest,
    IpAdapterStrengthRequest,
    ControlnetLoadRequest,
    ControlnetPreviewRequest,
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
    return {"loras": manager.get_loras(), "enabled": manager.loras_enabled}


@router.post("/lora/strength")
async def set_lora_strength(req: LoraStrengthRequest):
    """Change the strength of a loaded LoRA without downloading it again."""
    try:
        return await manager.set_lora_strength(req.lora_name, req.strength)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lora/enabled")
async def set_loras_enabled(req: LoraEnabledRequest):
    """Enable or disable every loaded LoRA."""
    try:
        return await manager.set_loras_enabled(req.enabled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ip-adapter/load")
async def load_ip_adapter(req: IpAdapterLoadRequest):
    """Load an IP-Adapter and set its strength."""
    try:
        return await manager.set_ip_adapter(
            req.uri, req.subfolder, req.weight_name, req.image_encoder_folder, req.strength
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ip-adapter/strength")
async def set_ip_adapter_strength(req: IpAdapterStrengthRequest):
    """Change the strength of the loaded IP-Adapter."""
    try:
        return await manager.set_ip_adapter_strength(req.strength)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ip-adapter/delete")
async def delete_ip_adapter():
    """Unload the IP-Adapter."""
    try:
        return await manager.delete_ip_adapter()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/controlnet/load")
async def load_controlnet(req: ControlnetLoadRequest):
    """Load a ControlNet. An empty model uses the pipeline default."""
    try:
        return await manager.load_controlnet(req.model or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/controlnet/offload")
async def offload_controlnet():
    """Offload the loaded ControlNet."""
    try:
        return await manager.offload_controlnet()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/controlnet/preview")
async def preview_controlnet(req: ControlnetPreviewRequest):
    """Render the Canny edge map that ControlNet will receive."""
    try:
        return {"image_base64": manager.preview_canny(req.image, req.low_threshold, req.high_threshold)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/set")
async def set_scheduler(req: SchedulerSetRequest):
    """Change the sampler/scheduler on the active pipeline."""
    try:
        res = await manager.set_scheduler(req.scheduler, req.pipeline_type)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
