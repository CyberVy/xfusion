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
    CannyPreviewRequest,
    SchedulerSetRequest,
)
from ..manager import manager
from ...engine.contracts import ModelLoadSettings

router = APIRouter(prefix="/api/v1", tags=["Models & Adapters"])


@router.post("/models/load")
async def load_model(req: ModelLoadRequest):
    """Load or switch the base diffusion model."""
    try:
        settings = ModelLoadSettings(
            source=req.source,
            model_version=req.model_version,
            backbone=req.backbone,
            text_encoder=req.text_encoder,
            memory=req.memory,
            delete_file_after_load=req.delete_file_after_load,
        )
        return await manager.load_model(settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
        return await manager.set_lora(req.url, req.lora_name, req.strength)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/lora/delete")
async def delete_lora(req: LoraDeleteRequest):
    """Delete a loaded LoRA adapter."""
    try:
        return await manager.delete_lora(req.lora_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/lora/list")
async def list_loras():
    """List all active LoRA adapters."""
    return {"loras": manager.get_loras(), "enabled": manager.loras_enabled}


@router.post("/lora/strength")
async def set_lora_strength(req: LoraStrengthRequest):
    """Change the strength of a loaded LoRA without downloading it again."""
    try:
        return await manager.set_lora_strength(req.lora_name, req.strength)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/lora/enabled")
async def set_loras_enabled(req: LoraEnabledRequest):
    """Enable or disable every loaded LoRA."""
    try:
        return await manager.set_loras_enabled(req.enabled)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ip-adapter/load")
async def load_ip_adapter(req: IpAdapterLoadRequest):
    """Load an IP-Adapter and set its strength."""
    try:
        return await manager.set_ip_adapter(
            req.source, req.folder, req.weight, req.encoder_folder, req.strength
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ip-adapter/strength")
async def set_ip_adapter_strength(req: IpAdapterStrengthRequest):
    """Change the strength of the loaded IP-Adapter."""
    try:
        return await manager.set_ip_adapter_strength(req.strength)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ip-adapter/delete")
async def delete_ip_adapter():
    """Unload the IP-Adapter."""
    try:
        return await manager.delete_ip_adapter()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/controlnet/load")
async def load_controlnet(req: ControlnetLoadRequest):
    """Load a ControlNet. An empty source uses the family default."""
    try:
        return await manager.load_controlnet(req.source or None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/controlnet/offload")
async def offload_controlnet():
    """Unload the loaded ControlNet."""
    try:
        return await manager.clear_controlnet()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/controlnet/preview")
async def preview_controlnet(req: CannyPreviewRequest):
    """Render the Canny edge map that ControlNet will receive."""
    try:
        return {"image_base64": manager.preview_canny(req.image, req.canny_low, req.canny_high)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/scheduler/set")
async def set_scheduler(req: SchedulerSetRequest):
    """Change the sampler on the active model."""
    try:
        return await manager.set_scheduler(req.scheduler)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
