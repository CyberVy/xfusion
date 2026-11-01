from __future__ import annotations
from typing import Optional, Union
from ...infra.memory import delete, free_memory_to_system
from ..types import ControlNetModelType


class ControlnetMixin:
    def __init__(self):
        self._controlnet: Optional[ControlNetModelType] = None

    def load_controlnet(
        self,
        controlnet_model: Optional[str] = None,
        **kwargs: Union[str, float, int, bool]
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement 'load_controlnet'")

    def offload_controlnet(self) -> None:
        task_pipes = getattr(self, "_task_pipes", None)
        if isinstance(task_pipes, dict):
            task_pipes.clear()
        if self._controlnet is not None:
            delete(self._controlnet)
            self._controlnet = None
        free_memory_to_system()

