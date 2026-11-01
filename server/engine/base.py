from __future__ import annotations
import inspect
import random
from typing import Dict, Optional, List, Type, Union, Callable
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import SchedulerMixin
from diffusers.utils import BaseOutput

from .adapters.lora import LoraAdapterMixin
from .adapters.ip_adapter import IPAdapterMixin
from .adapters.controlnet import ControlnetMixin
from .types import PipelineMap, SchedulerConfigMap, SchedulerConfigValue, TorchDevice
from ..models.inspect import apply_post_load_optimizations, apply_execution_device
from ..infra.memory import delete, free_memory_to_system

TASK_BASE_INDEX: Dict[str, int] = {
    "txt2img": 0,
    "img2img": 1,
    "inpaint": 2,
}


class BasePipelineEngine(LoraAdapterMixin, IPAdapterMixin, ControlnetMixin):
    pipeline_map: PipelineMap = {}
    scheduler_map: SchedulerConfigMap = {}

    def __init__(self, pipeline: Optional[DiffusionPipeline]):
        LoraAdapterMixin.__init__(self)
        IPAdapterMixin.__init__(self)
        ControlnetMixin.__init__(self)

        self.pipeline: Optional[DiffusionPipeline] = pipeline
        self._active_pipe: Optional[DiffusionPipeline] = pipeline
        self._task_pipes: Dict[int, DiffusionPipeline] = {}

        self.model_name: Optional[str] = getattr(pipeline, "name_or_path", None) if pipeline else None
        self.model_version: Optional[str] = None
        self.quantization: str = "none"
        self.text_encoder_mode: str = "default"
        self.offload_mode: str = "none"
        self.gpu_id: int = 0
        self._scheduler: Optional[SchedulerMixin] = (
            getattr(pipeline, "scheduler", None) if pipeline is not None else None
        )

        if pipeline is None:
            return

        self.model_version = self._resolve_model_version()
        apply_post_load_optimizations(
            self.pipeline,
            quantization=self.quantization,
            text_encoder_mode=self.text_encoder_mode,
        )

    def get_task_pipeline(self, task: str = "txt2img", use_controlnet: bool = False) -> DiffusionPipeline:
        """Derive a zero-copy Diffusers pipeline for the target task while preserving offload and quantization hooks."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized.")

        task_index = TASK_BASE_INDEX.get(task, 0) + (3 if use_controlnet else 0)
        classes = self.pipeline_map.get(self.model_version or "", ())
        if not classes or task_index >= len(classes):
            return self.pipeline

        target_cls = classes[task_index]
        if self.pipeline.__class__ is target_cls and (not use_controlnet or self._controlnet is None):
            return self.pipeline

        cached = self._task_pipes.get(task_index)
        if cached is not None:
            if hasattr(self.pipeline, "scheduler"):
                cached.scheduler = self.pipeline.scheduler
            if hasattr(self.pipeline, "_all_hooks"):
                cached._all_hooks = self.pipeline._all_hooks
            if hasattr(self.pipeline, "_offload_gpu_id"):
                cached._offload_gpu_id = self.pipeline._offload_gpu_id
            return cached

        sig_params = inspect.signature(target_cls.__init__).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
        raw_components = dict(getattr(self.pipeline, "components", {}) or {})
        if use_controlnet and self._controlnet is not None:
            raw_components["controlnet"] = self._controlnet
        if not accepts_kwargs:
            raw_components = {k: v for k, v in raw_components.items() if k in sig_params}

        derived = target_cls(**raw_components)
        if hasattr(self.pipeline, "_all_hooks"):
            derived._all_hooks = self.pipeline._all_hooks
        if hasattr(self.pipeline, "_offload_gpu_id"):
            derived._offload_gpu_id = self.pipeline._offload_gpu_id
        self._task_pipes[task_index] = derived
        return derived

    def _resolve_model_version(self) -> str:
        if self.pipeline is None:
            return "xl"
        target_cls = self.pipeline.__class__
        for model_ver, class_tuple in self.pipeline_map.items():
            if target_cls in class_tuple:
                return model_ver
        return str(getattr(self.pipeline, "model_version", "xl"))

    def set_scheduler(
        self,
        scheduler_type: Union[str, Type[SchedulerMixin]],
        **kwargs: SchedulerConfigValue,
    ) -> None:
        pipeline_obj = self.pipeline
        if pipeline_obj is None:
            raise RuntimeError("Cannot set scheduler on uninitialized pipeline.")

        base_sched = getattr(self, "_scheduler", getattr(pipeline_obj, "scheduler", None))
        config = getattr(
            base_sched, "config", getattr(getattr(pipeline_obj, "scheduler", None), "config", None)
        )

        if not isinstance(scheduler_type, str):
            from_config = getattr(scheduler_type, "from_config", None)
            if from_config is not None:
                pipeline_obj.scheduler = from_config(config, **kwargs)
        else:
            sched_key = scheduler_type.upper()
            if sched_key in self.scheduler_map:
                sched_cls, default_kwargs = self.scheduler_map[sched_key]
                final_kwargs = {**default_kwargs, **kwargs}
                from_config = getattr(sched_cls, "from_config", None)
                if from_config is not None:
                    pipeline_obj.scheduler = from_config(config, **final_kwargs)
            else:
                raise ValueError(f"Scheduler '{scheduler_type}' is not supported.")

    def reset_scheduler(self) -> None:
        if self._scheduler is not None and self.pipeline is not None:
            self.pipeline.scheduler = self._scheduler

    def configure_execution(self, offload_mode: str = "none", gpu_id: int = 0) -> "BasePipelineEngine":
        self.offload_mode = (offload_mode or "none").lower()
        self.gpu_id = gpu_id
        if self.pipeline is not None:
            apply_execution_device(self.pipeline, offload_mode=self.offload_mode, gpu_id=gpu_id)
        return self

    def interrupt(self) -> None:
        for pipe in (self._active_pipe, self.pipeline):
            if pipe is not None:
                try:
                    pipe._interrupt = True
                except Exception:
                    pass

    def clear(self) -> None:
        self._task_pipes.clear()
        if self.pipeline is not None and hasattr(self.pipeline, "maybe_free_model_hooks"):
            try:
                self.pipeline.maybe_free_model_hooks()
            except Exception:
                pass
        components_dict = getattr(self.pipeline, "components", {})
        components = list(components_dict.values()) if isinstance(components_dict, dict) else []
        for comp in components:
            delete(comp)
        if self._controlnet is not None:
            delete(self._controlnet)
            self._controlnet = None
        if self._active_pipe is not None and self._active_pipe is not self.pipeline:
            delete(self._active_pipe)
        self._active_pipe = None
        if self.pipeline is not None:
            delete(self.pipeline)
            self.pipeline = None
        free_memory_to_system()

    def generate(
        self,
        seed: Optional[int] = None,
        num: int = 1,
        task: str = "txt2img",
        use_controlnet: bool = False,
        **kwargs: Union[str, int, float, bool, Image.Image, None, Callable],
    ) -> List[Image.Image]:
        """Generate images for the requested task."""
        seeds = (
            [seed + i for i in range(num)]
            if seed is not None
            else [random.randint(1, 2**31 - 1) for _ in range(num)]
        )
        images: List[Image.Image] = []
        exec_device = self.execution_device
        for item_seed in seeds:
            call_kwargs = dict(kwargs)
            if exec_device is not None:
                call_kwargs["generator"] = torch.Generator(exec_device).manual_seed(item_seed)
            out = self(task=task, use_controlnet=use_controlnet, **call_kwargs)
            if hasattr(out, "images"):
                out_images = getattr(out, "images")
                if isinstance(out_images, list) and len(out_images) > 0:
                    images.append(out_images[0])
        return images

    @property
    def execution_device(self) -> Optional[TorchDevice]:
        if self.pipeline is None:
            return None
        return getattr(self.pipeline, "_execution_device", None) or getattr(
            self.pipeline, "device", None
        )

    @property
    def device(self) -> Optional[TorchDevice]:
        return getattr(self.pipeline, "device", None)

    @property
    def components(self) -> Dict[str, Union[DiffusionPipeline, SchedulerMixin, None]]:
        comps = getattr(self.pipeline, "components", {})
        return dict(comps) if isinstance(comps, dict) else {}

    def __getattr__(self, name: str) -> Union[DiffusionPipeline, Callable, None]:
        pipeline = object.__getattribute__(self, "pipeline")
        if pipeline is not None and hasattr(pipeline, name):
            return getattr(pipeline, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __call__(
        self,
        *args: Union[str, int, float, bool],
        task: str = "txt2img",
        use_controlnet: bool = False,
        **kwargs: Union[str, int, float, bool, Image.Image, None, Callable],
    ) -> BaseOutput:
        pipe = self.get_task_pipeline(task=task, use_controlnet=use_controlnet)
        self._active_pipe = pipe
        try:
            if callable(pipe):
                return pipe(*args, **kwargs)
            raise TypeError(f"Underlying pipeline {pipe} is not callable.")
        finally:
            self._active_pipe = self.pipeline
            if self.pipeline is not None and hasattr(self.pipeline, "maybe_free_model_hooks"):
                try:
                    self.pipeline.maybe_free_model_hooks()
                except Exception:
                    pass
