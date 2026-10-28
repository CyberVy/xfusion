from typing import Optional, Any
from ...infra.memory import delete, free_memory_to_system


class ControlnetMixin:
    def __init__(self):
        self._controlnet: Optional[Any] = None
        self.text_to_image_controlnet_pipeline: Optional[Any] = None
        self.image_to_image_controlnet_pipeline: Optional[Any] = None
        self.inpainting_controlnet_pipeline: Optional[Any] = None

    def load_controlnet(self, controlnet_model: Optional[str] = None, **kwargs: Any) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement 'load_controlnet'")

    def offload_controlnet(self) -> None:
        if self._controlnet is not None:
            delete(self._controlnet)
            self._controlnet = None
        free_memory_to_system()
