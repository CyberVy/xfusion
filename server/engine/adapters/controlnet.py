from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING
from ...infra.memory import delete, free_memory_to_system
from ..types import ControlNetModelType

if TYPE_CHECKING:
    from ..base import BasePipelineEngine


class ControlnetMixin:
    def __init__(self):
        self._controlnet: Optional[ControlNetModelType] = None
        self.text_to_image_controlnet_pipeline: Optional[BasePipelineEngine] = None
        self.image_to_image_controlnet_pipeline: Optional[BasePipelineEngine] = None
        self.inpainting_controlnet_pipeline: Optional[BasePipelineEngine] = None

    def load_controlnet(
        self,
        controlnet_model: Optional[str] = None,
        **kwargs: Union[str, float, int, bool]
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement 'load_controlnet'")

    def offload_controlnet(self) -> None:
        if self._controlnet is not None:
            delete(self._controlnet)
            self._controlnet = None
        free_memory_to_system()
