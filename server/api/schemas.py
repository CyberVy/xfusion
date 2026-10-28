from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field

MetadataScalar = Union[str, int, float, bool, None]


class IpAdapterInfo(BaseModel):
    uri: str
    subfolder: str
    weight_name: str
    image_encoder_folder: str
    strength: float


class GenerationExtras(BaseModel):
    control_image: Optional[str] = Field(default=None, description="Control image; converted to Canny on the server")
    controlnet_conditioning_scale: Optional[float] = Field(default=0.5, description="ControlNet scale")
    low_threshold: Optional[int] = Field(default=100, description="Canny low threshold")
    high_threshold: Optional[int] = Field(default=200, description="Canny high threshold")
    ip_adapter_image: Optional[str] = Field(default=None, description="IP-Adapter reference image")


class Txt2ImgRequest(GenerationExtras):
    prompt: str = Field(..., description="Prompt describing the image to generate")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    guidance_scale: Optional[float] = Field(default=7.0, description="Guidance scale / CFG scale")
    num_inference_steps: Optional[int] = Field(default=25, description="Number of denoising steps")
    clip_skip: Optional[int] = Field(default=0, description="CLIP skip value")
    width: Optional[int] = Field(default=1024, description="Image width")
    height: Optional[int] = Field(default=1024, description="Image height")
    seed: Optional[int] = Field(default=None, description="Random seed (None for random)")
    num: Optional[int] = Field(default=1, ge=1, le=10, description="Number of images to generate")
    scheduler: Optional[str] = Field(default=None, description="Scheduler name (e.g. DPM++ 2M KARRAS)")


class Img2ImgRequest(GenerationExtras):
    image: str = Field(..., description="Input image as Base64 Data URL or HTTP URL")
    prompt: str = Field(..., description="Prompt describing the modification")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    strength: Optional[float] = Field(default=0.75, description="Image strength / transformation ratio")
    guidance_scale: Optional[float] = Field(default=7.0, description="Guidance scale / CFG scale")
    num_inference_steps: Optional[int] = Field(default=25, description="Number of denoising steps")
    clip_skip: Optional[int] = Field(default=0, description="CLIP skip value")
    width: Optional[int] = Field(default=None, description="Output width (None to preserve)")
    height: Optional[int] = Field(default=None, description="Output height (None to preserve)")
    seed: Optional[int] = Field(default=None, description="Random seed")
    num: Optional[int] = Field(default=1, ge=1, le=10, description="Number of images to generate")
    scheduler: Optional[str] = Field(default=None, description="Scheduler name")


class InpaintRequest(GenerationExtras):
    image: str = Field(..., description="Original image as Base64 Data URL or HTTP URL")
    mask_image: str = Field(..., description="Mask image as Base64 Data URL or HTTP URL")
    prompt: str = Field(..., description="Prompt for inpainting")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    strength: Optional[float] = Field(default=0.8, description="Inpaint strength")
    guidance_scale: Optional[float] = Field(default=7.0, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(default=25, description="Denoising steps")
    clip_skip: Optional[int] = Field(default=0, description="CLIP skip value")
    width: Optional[int] = Field(default=None, description="Width")
    height: Optional[int] = Field(default=None, description="Height")
    seed: Optional[int] = Field(default=None, description="Random seed")
    num: Optional[int] = Field(default=1, ge=1, le=10, description="Number of images to generate")
    scheduler: Optional[str] = Field(default=None, description="Scheduler name")


class ModelLoadRequest(BaseModel):
    model: str = Field(..., description="Hugging Face repo ID, local path, or Civitai download URL")
    model_version: Optional[str] = Field(default="xl", description="Model architecture version (e.g., 1.5, xl, pony, 3, 3.5, flux, or empty to auto-detect)")


class LoraLoadRequest(BaseModel):
    url: str = Field(..., description="LoRA URL or Hugging Face path")
    lora_name: str = Field(..., description="Identifier name for the LoRA adapter")
    strength: Optional[float] = Field(default=0.4, description="LoRA adapter weight/strength (0.0 - 1.0)")


class LoraDeleteRequest(BaseModel):
    lora_name: str = Field(..., description="Name of the LoRA adapter to remove")


class LoraStrengthRequest(BaseModel):
    lora_name: str
    strength: float = Field(..., ge=0, le=1)


class LoraEnabledRequest(BaseModel):
    enabled: bool


class IpAdapterLoadRequest(BaseModel):
    uri: str
    subfolder: str = ""
    weight_name: str
    image_encoder_folder: str = "image_encoder"
    strength: float = Field(default=0.5, ge=0, le=1)


class IpAdapterStrengthRequest(BaseModel):
    strength: float = Field(..., ge=0, le=1)


class ControlnetLoadRequest(BaseModel):
    model: Optional[str] = None


class ControlnetPreviewRequest(BaseModel):
    image: str
    low_threshold: int = 100
    high_threshold: int = 200


class SchedulerSetRequest(BaseModel):
    scheduler: str = Field(..., description="Scheduler name (e.g. DPM++ 2M KARRAS, EULER A)")
    pipeline_type: Optional[str] = Field(default="txt2img", description="Pipeline type: txt2img, img2img, inpaint")


class GeneratedImage(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image with data URI prefix")
    seed: int = Field(..., description="Random seed used for generation")
    width: int = Field(..., description="Output image width")
    height: int = Field(..., description="Output image height")


class GenerationResponse(BaseModel):
    status: str = "success"
    images: List[GeneratedImage]
    elapsed_seconds: float
    metadata: Dict[str, MetadataScalar] = Field(default_factory=dict)


class GenerationProgress(BaseModel):
    active: bool = False
    step: int = 0
    total_steps: int = 0
    image_index: int = 0
    image_count: int = 0


class SystemInfoResponse(BaseModel):
    status: str
    gpu_count: int
    gpu_names: List[str]
    current_model: Optional[str] = None
    model_version: Optional[str] = None
    loras: Dict[str, float] = {}
    loras_enabled: bool = True
    schedulers: List[str] = []
    controlnet_loaded: bool = False
    ip_adapter: Optional[IpAdapterInfo] = None
    python_version: str = ""
    os_platform: str = ""
