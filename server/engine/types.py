from __future__ import annotations
from typing import Dict, Tuple, Type, Union, Optional, List, Callable, TypedDict, TYPE_CHECKING
from PIL import Image

from diffusers import DiffusionPipeline, ControlNetModel, SD3ControlNetModel
from diffusers.schedulers import SchedulerMixin

if TYPE_CHECKING:
    import torch
    TorchTensor = torch.Tensor
    TorchGenerator = torch.Generator
    TorchDevice = torch.device
    TorchModule = torch.nn.Module
else:
    try:
        import torch
        TorchTensor = torch.Tensor
        TorchGenerator = torch.Generator
        TorchDevice = torch.device
        TorchModule = torch.nn.Module
    except (ImportError, AttributeError):
        TorchTensor = Union[str, None]
        TorchGenerator = Union[int, None]
        TorchDevice = Union[str, int, None]
        TorchModule = Union[str, None]

StepCallback = Callable[[DiffusionPipeline, int, Union[int, float], Dict[str, Union[float, int, str]]], Dict[str, Union[float, int, str]]]
SchedulerConfigValue = Union[str, bool, int, float]
SchedulerConfigMap = Dict[str, Tuple[Type[SchedulerMixin], Dict[str, SchedulerConfigValue]]]
PipelineMap = Dict[str, Tuple[Type[DiffusionPipeline], ...]]
ControlNetModelType = Union[ControlNetModel, SD3ControlNetModel]


class InferenceKwargs(TypedDict, total=False):
    prompt: Optional[Union[str, List[str]]]
    negative_prompt: Optional[Union[str, List[str]]]
    guidance_scale: float
    num_inference_steps: int
    clip_skip: Optional[int]
    width: int
    height: int
    image: Optional[Image.Image]
    mask_image: Optional[Image.Image]
    control_image: Optional[Image.Image]
    ip_adapter_image: Optional[Image.Image]
    controlnet_conditioning_scale: Optional[float]
    strength: Optional[float]
    generator: Optional[TorchGenerator]
    callback_on_step_end: Optional[StepCallback]
    prompt_embeds: Optional[TorchTensor]
    negative_prompt_embeds: Optional[TorchTensor]
    pooled_prompt_embeds: Optional[TorchTensor]
    negative_pooled_prompt_embeds: Optional[TorchTensor]
