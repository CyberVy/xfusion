from fastapi import APIRouter, HTTPException
from ..schemas import (
    Txt2ImgRequest,
    Img2ImgRequest,
    InpaintRequest,
    GenerationResponse,
    GeneratedImage
)
from ..manager import manager

router = APIRouter(prefix="/api/v1/generate", tags=["Generation"])


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
