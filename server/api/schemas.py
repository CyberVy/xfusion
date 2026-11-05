from typing import Optional, List, Dict, Literal, Union
from pydantic import BaseModel, Field

MetadataScalar = Union[str, int, float, bool, None]
Backbone = Literal["fp16", "int4", "int8", "fp8", "gguf"]
TextEncoderMode = Literal["default", "int4", "int8", "fp8", "skip"]
MemoryMode = Literal["gpu", "offload", "sequential"]


class LoadedIpAdapterInfo(BaseModel):
    source: str
    folder: str = ""
    weight: str
    encoder_folder: str = "image_encoder"
    strength: float


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    guidance: Optional[float] = None
    steps: int = 25
    clip_skip: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    image_count: int = Field(default=1, ge=1, le=10)
    scheduler: Optional[str] = None
    image: Optional[str] = None
    mask: Optional[str] = None
    strength: Optional[float] = None
    control_image: Optional[str] = None
    control_scale: Optional[float] = None
    canny_low: int = 100
    canny_high: int = 200
    reference_image: Optional[str] = None


class ModelLoadRequest(BaseModel):
    source: str
    model_version: Optional[str] = None
    backbone: Backbone = "fp16"
    text_encoder: TextEncoderMode = "default"
    memory: MemoryMode = "gpu"
    delete_file_after_load: bool = False


class LoraLoadRequest(BaseModel):
    url: str
    lora_name: str
    strength: float = 0.4


class LoraDeleteRequest(BaseModel):
    lora_name: str


class LoraStrengthRequest(BaseModel):
    lora_name: str
    strength: float = Field(..., ge=0, le=1)


class LoraEnabledRequest(BaseModel):
    enabled: bool


class IpAdapterLoadRequest(BaseModel):
    source: str
    weight: str
    folder: str = ""
    encoder_folder: str = "image_encoder"
    strength: float = Field(default=0.5, ge=0, le=1)


class IpAdapterStrengthRequest(BaseModel):
    strength: float = Field(..., ge=0, le=1)


class ControlnetLoadRequest(BaseModel):
    source: Optional[str] = None


class CannyPreviewRequest(BaseModel):
    image: str
    canny_low: int = 100
    canny_high: int = 200


class SchedulerSetRequest(BaseModel):
    scheduler: str


class GeneratedImage(BaseModel):
    image_base64: str
    seed: int
    width: int
    height: int


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
    backbone: str = "fp16"
    text_encoder: str = "default"
    memory: str = "gpu"
    adapters: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    schedulers: List[str] = Field(default_factory=list)
    default_size: int = 1024
    default_guidance: float = 7.0
    accepts_negative_prompt: bool = True
    accepts_clip_skip: bool = True
    loras: Dict[str, float] = Field(default_factory=dict)
    loras_enabled: bool = True
    controlnet_loaded: bool = False
    ip_adapter: Optional[LoadedIpAdapterInfo] = None
    python_version: str = ""
    os_platform: str = ""
