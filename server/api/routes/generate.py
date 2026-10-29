from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..schemas import (
    Txt2ImgRequest,
    Img2ImgRequest,
    InpaintRequest,
    GenerationResponse,
    GenerationProgress,
    GeneratedImage
)
from ..manager import manager

router = APIRouter(prefix="/api/v1/generate", tags=["Generation"])

STREAMING_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"
}


@router.get("/progress", response_model=GenerationProgress)
async def generation_progress():
    """Current denoising step. Safe to poll while a generation request is in flight."""
    return GenerationProgress(**manager.progress)


@router.post("/cancel")
async def cancel_generation():
    """Interrupt currently running generation task and release GPU resources immediately."""
    cancelled = manager.cancel()
    return {"status": "success", "cancelled": cancelled}


@router.post("/txt2img", response_model=GenerationResponse)
async def generate_text_to_image(req: Txt2ImgRequest):
    """Generate image from text prompt."""
    try:
        images_data, elapsed = await manager.generate_txt2img(req)
        return GenerationResponse(
            status="success",
            images=[GeneratedImage(**img) for img in images_data],
            elapsed_seconds=round(elapsed, 3),
            metadata={"type": "txt2img", "prompt": req.prompt}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/txt2img/stream")
async def generate_text_to_image_stream(req: Txt2ImgRequest):
    """Generate image from text prompt with SSE streaming (yields each image as it finishes)."""
    return StreamingResponse(
        manager.generate_txt2img_stream(req),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS
    )


@router.post("/img2img", response_model=GenerationResponse)
async def generate_image_to_image(req: Img2ImgRequest):
    """Generate or stylize image from input image and prompt."""
    try:
        images_data, elapsed = await manager.generate_img2img(req)
        return GenerationResponse(
            status="success",
            images=[GeneratedImage(**img) for img in images_data],
            elapsed_seconds=round(elapsed, 3),
            metadata={"type": "img2img", "strength": req.strength}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/img2img/stream")
async def generate_image_to_image_stream(req: Img2ImgRequest):
    """Generate image from input image and prompt with SSE streaming."""
    return StreamingResponse(
        manager.generate_img2img_stream(req),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS
    )


@router.post("/inpaint", response_model=GenerationResponse)
async def generate_inpainting(req: InpaintRequest):
    """Inpaint a masked area of an image."""
    try:
        images_data, elapsed = await manager.generate_inpaint(req)
        return GenerationResponse(
            status="success",
            images=[GeneratedImage(**img) for img in images_data],
            elapsed_seconds=round(elapsed, 3),
            metadata={"type": "inpaint"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inpaint/stream")
async def generate_inpainting_stream(req: InpaintRequest):
    """Inpaint a masked area with SSE streaming."""
    return StreamingResponse(
        manager.generate_inpaint_stream(req),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS
    )
