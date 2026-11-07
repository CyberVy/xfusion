from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional
from PIL import Image

Backbone = Literal["fp16", "int4", "int8", "fp8", "gguf"]
TextEncoderMode = Literal["default", "int4", "int8", "fp8", "skip"]
MemoryMode = Literal["gpu", "offload", "sequential"]
DtypeMode = Literal["auto", "fp16", "bf16"]
GenerationTask = Literal["txt2img", "img2img", "inpaint"]

BACKBONES = ("fp16", "int4", "int8", "fp8", "gguf")
TEXT_ENCODER_MODES = ("default", "int4", "int8", "fp8", "skip")
MEMORY_MODES = ("gpu", "offload", "sequential")
DTYPE_MODES = ("auto", "fp16", "bf16")
GENERATION_TASKS = ("txt2img", "img2img", "inpaint")

StepCallback = Callable[[int, int], None]


class FeatureNotSupported(RuntimeError):
    def __init__(self, feature: str):
        super().__init__(f"{feature} is not supported by this model.")
        self.feature = feature


@dataclass(frozen=True)
class ModelLoadSettings:
    source: str
    model_version: Optional[str] = None
    backbone: str = "fp16"
    text_encoder: str = "default"
    memory: str = "gpu"
    dtype: str = "auto"
    delete_file_after_load: bool = False

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"Backbone '{self.backbone}' is not supported.")
        if self.text_encoder not in TEXT_ENCODER_MODES:
            raise ValueError(f"Text encoder mode '{self.text_encoder}' is not supported.")
        if self.memory not in MEMORY_MODES:
            raise ValueError(f"Memory mode '{self.memory}' is not supported.")
        if self.dtype not in DTYPE_MODES:
            raise ValueError(f"Dtype mode '{self.dtype}' is not supported.")


@dataclass(frozen=True)
class ModelFeatures:
    model_version: str
    tasks: frozenset[str]
    adapters: frozenset[str]
    schedulers: tuple[str, ...]
    default_size: int
    default_guidance: float
    accepts_negative_prompt: bool
    accepts_clip_skip: bool
    size_divisor: int


@dataclass(frozen=True)
class GenerationRequest:
    task: str
    prompt: str
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    steps: int = 25
    guidance: Optional[float] = None
    seed: Optional[int] = None
    clip_skip: Optional[int] = None
    scheduler: Optional[str] = None
    image_count: int = 1
    image: Optional[Image.Image] = None
    mask: Optional[Image.Image] = None
    strength: Optional[float] = None
    control_image: Optional[Image.Image] = None
    control_scale: Optional[float] = None
    canny_low: int = 100
    canny_high: int = 200
    reference_image: Optional[Image.Image] = None


@dataclass(frozen=True)
class GenerationResult:
    image: Image.Image
    seed: int


@dataclass(frozen=True)
class IpAdapterSource:
    source: str
    weight: str
    folder: str = ""
    encoder_folder: str = "image_encoder"


@dataclass(frozen=True)
class LoadedIpAdapter:
    source: str
    folder: str
    weight: str
    encoder_folder: str
    strength: float


@dataclass(frozen=True)
class LoadedAdapters:
    loras: Mapping[str, float]
    loras_enabled: bool
    ip_adapter: Optional[LoadedIpAdapter]
    controlnet_loaded: bool
