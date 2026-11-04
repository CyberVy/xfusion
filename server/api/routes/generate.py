from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..schemas import GenerationRequest, GenerationResponse, GenerationProgress, GeneratedImage
from ..manager import manager

router = APIRouter(prefix="/api/v1/generate", tags=["Generation"])

STREAMING_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _require_image(req: GenerationRequest) -> None:
    if not req.image:
        raise HTTPException(status_code=400, detail="image is required")


def _require_mask(req: GenerationRequest) -> None:
    if not req.mask:
        raise HTTPException(status_code=400, detail="mask is required")


@router.get("/progress", response_model=GenerationProgress)
async def generation_progress():
    """Current denoising step. Safe to poll while a generation request is in flight."""
    return GenerationProgress(**manager.progress)


@router.post("/cancel")
async def cancel_generation():
    """Interrupt the current generation and release GPU resources."""
    cancelled = manager.cancel()
    return {"status": "success", "cancelled": cancelled}


@router.post("/txt2img", response_model=GenerationResponse)
async def generate_text_to_image(req: GenerationRequest):
    """Generate an image from a text prompt."""
    try:
        images_data, elapsed = await manager.generate("txt2img", req)
        return GenerationResponse(
            status="success",
            images=[GeneratedImage(**img) for img in images_data],
            elapsed_seconds=round(elapsed, 3),
            metadata={"type": "txt2img", "prompt": req.prompt},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/txt2img/stream")
async def generate_text_to_image_stream(req: GenerationRequest):
    """Generate an image from a text prompt and stream each result."""
    return StreamingResponse(
        manager.generate_stream("txt2img", req),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS,
    )


@router.post("/img2img", response_model=GenerationResponse)
async def generate_image_to_image(req: GenerationRequest):
    """Generate an image from a source image and a prompt."""
    _require_image(req)
    try:
        images_data, elapsed = await manager.generate("img2img", req)
        return GenerationResponse(
            status="success",
            images=[GeneratedImage(**img) for img in images_data],
            elapsed_seconds=round(elapsed, 3),
            metadata={"type": "img2img", "strength": req.strength},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/img2img/stream")
async def generate_image_to_image_stream(req: GenerationRequest):
    """Generate an image from a source image and stream each result."""
    _require_image(req)
    return StreamingResponse(
        manager.generate_stream("img2img", req),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS,
    )


@router.post("/inpaint", response_model=GenerationResponse)
async def generate_inpainting(req: GenerationRequest):
    """Inpaint a masked area of an image."""
    _require_image(req)
    _require_mask(req)
    try:
        images_data, elapsed = await manager.generate("inpaint", req)
        return GenerationResponse(
            status="success",
            images=[GeneratedImage(**img) for img in images_data],
            elapsed_seconds=round(elapsed, 3),
            metadata={"type": "inpaint"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/inpaint/stream")
async def generate_inpainting_stream(req: GenerationRequest):
    """Inpaint a masked area and stream each result."""
    _require_image(req)
    _require_mask(req)
    return StreamingResponse(
        manager.generate_stream("inpaint", req),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS,
    )
