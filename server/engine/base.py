from __future__ import annotations
import random
from typing import Dict, Optional, List, Tuple, Type, Union, Callable, TYPE_CHECKING
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.schedulers import SchedulerMixin
from diffusers.utils import BaseOutput

from .adapters.lora import LoraAdapterMixin
from .adapters.ip_adapter import IPAdapterMixin
from .adapters.controlnet import ControlnetMixin
from .types import PipelineMap, SchedulerConfigMap, SchedulerConfigValue, InferenceKwargs
from ..infra.memory import delete, free_memory_to_system

if TYPE_CHECKING:
    import torch
    TorchDevice = torch.device
else:
    try:
        import torch
        TorchDevice = torch.device
    except (ImportError, AttributeError):
        TorchDevice = Union[str, int, None]


class BasePipelineEngine(LoraAdapterMixin, IPAdapterMixin, ControlnetMixin):
    pipeline_map: PipelineMap = {}
    scheduler_map: SchedulerConfigMap = {}

    def __init__(self, pipeline: Optional[DiffusionPipeline], init_sub_pipelines: bool = True):
        LoraAdapterMixin.__init__(self)
        IPAdapterMixin.__init__(self)
        ControlnetMixin.__init__(self)

        self.pipeline: Optional[DiffusionPipeline] = pipeline
        self.sub_pipelines: Dict[str, BasePipelineEngine] = {}
        self.model_name: Optional[str] = getattr(pipeline, "name_or_path", None) if pipeline else None
        self.model_version: Optional[str] = None
        self.pipeline_type: int = 0
        self.pipeline_class: Optional[Type[DiffusionPipeline]] = pipeline.__class__ if pipeline is not None else None
        self._scheduler: Optional[SchedulerMixin] = getattr(pipeline, "scheduler", None) if pipeline is not None else None

        if pipeline is None:
            return

        self.model_version, self.pipeline_type, self.pipeline_class = self._resolve_pipeline_type()

        # VAE tiling optimization
        components_dict = getattr(self.pipeline, "components", {})
        components = dict(components_dict) if isinstance(components_dict, dict) else {}
        vae = components.get("vae")
        if vae is not None and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception:
                pass

        if "flux" in self.pipeline.__class__.__name__.lower():
            components.pop("image_encoder", None)
            components.pop("feature_extractor", None)

        if init_sub_pipelines and self.model_version in self.pipeline_map:
            classes = self.pipeline_map[self.model_version]
            engine_cls = self.__class__

            # txt2img
            if self.pipeline_type != 0 and len(classes) > 0:
                self.text_to_image_pipeline = engine_cls(classes[0](**components), init_sub_pipelines=False)
                self.sub_pipelines["text_to_image_pipeline"] = self.text_to_image_pipeline
            else:
                self.text_to_image_pipeline = self

            # img2img
            if self.pipeline_type != 1 and len(classes) > 1:
                self.image_to_image_pipeline = engine_cls(classes[1](**components), init_sub_pipelines=False)
                self.sub_pipelines["image_to_image_pipeline"] = self.image_to_image_pipeline
            else:
                self.image_to_image_pipeline = self

            # inpainting
            if self.pipeline_type != 2 and len(classes) > 2:
                self.inpainting_pipeline = engine_cls(classes[2](**components), init_sub_pipelines=False)
                self.sub_pipelines["inpainting_pipeline"] = self.inpainting_pipeline
            else:
                self.inpainting_pipeline = self

            self._sync_sub_pipelines()

    def _resolve_pipeline_type(self) -> Tuple[str, int, Type[DiffusionPipeline]]:
        if self.pipeline is None:
            return "xl", 0, DiffusionPipeline
        target_cls = self.pipeline.__class__
        for model_ver, class_tuple in self.pipeline_map.items():
            for p_type, cls in enumerate(class_tuple):
                if cls == target_cls:
                    return model_ver, p_type, cls
        # Default fallback
        fallback_ver = str(getattr(self.pipeline, "model_version", "xl"))
        return fallback_ver, 0, target_cls

    def _sync_sub_pipelines(self) -> None:
        for sub_p in self.sub_pipelines.values():
            if sub_p is not self:
                sub_p.download_kwargs = self.download_kwargs
                sub_p.lora_dict = self.lora_dict
                sub_p.model_name = self.model_name

    def set_scheduler(
        self,
        scheduler_type: Union[str, Type[SchedulerMixin]],
        **kwargs: SchedulerConfigValue
    ) -> None:
        pipeline_obj = self.pipeline
        if pipeline_obj is None:
            raise RuntimeError("Cannot set scheduler on uninitialized pipeline.")

        base_sched = getattr(self, "_scheduler", getattr(pipeline_obj, "scheduler", None))
        config = getattr(base_sched, "config", getattr(getattr(pipeline_obj, "scheduler", None), "config", None))

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

    def to(self, *args: Union[str, int, TorchDevice], **kwargs: Union[str, int, TorchDevice]) -> "BasePipelineEngine":
        if self.pipeline is not None and hasattr(self.pipeline, "to"):
            self.pipeline.to(*args, **kwargs)
        for sub_p in self.sub_pipelines.values():
            if sub_p is not self and getattr(sub_p, "pipeline", None) is not None:
                sub_p.pipeline.to(*args, **kwargs)
        return self

    def clear(self) -> None:
        components_dict = getattr(self.pipeline, "components", {})
        components = list(components_dict.values()) if isinstance(components_dict, dict) else []
        for comp in components:
            delete(comp)
        if self._controlnet is not None:
            delete(self._controlnet)
            self._controlnet = None
        if self.pipeline is not None:
            delete(self.pipeline)
            self.pipeline = None
        free_memory_to_system()

    def generate(
        self,
        seed: Optional[int] = None,
        num: int = 1,
        **kwargs: Union[str, int, float, bool, Image.Image, None, Callable]
    ) -> List[Image.Image]:
        """Generate images without external side-effects."""
        seeds = (
            [seed + i for i in range(num)]
            if seed is not None
            else [random.randint(1, 2**31 - 1) for _ in range(num)]
        )
        images: List[Image.Image] = []
        for item_seed in seeds:
            call_kwargs = dict(kwargs)
            device = getattr(self, "device", None)
            if device is not None:
                import torch
                call_kwargs["generator"] = torch.Generator(device).manual_seed(item_seed)
            out = self(**call_kwargs)
            if hasattr(out, "images"):
                out_images = getattr(out, "images")
                if isinstance(out_images, list) and len(out_images) > 0:
                    images.append(out_images[0])
        return images

    @property
    def device(self) -> Optional[TorchDevice]:
        return getattr(self.pipeline, "device", None)

    @property
    def components(self) -> Dict[str, Union[DiffusionPipeline, SchedulerMixin, None]]:
        comps = getattr(self.pipeline, "components", {})
        return dict(comps) if isinstance(comps, dict) else {}

    def __getattr__(self, name: str) -> Union[DiffusionPipeline, Callable, None]:
        # Delegate unknown attribute lookups directly to the underlying Diffusers pipeline
        pipeline = object.__getattribute__(self, "pipeline")
        if pipeline is not None and hasattr(pipeline, name):
            return getattr(pipeline, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __call__(
        self,
        *args: Union[str, int, float, bool],
        **kwargs: Union[str, int, float, bool, Image.Image, None, Callable]
    ) -> BaseOutput:
        pipe = self.pipeline
        if callable(pipe):
            return pipe(*args, **kwargs)
        raise TypeError(f"Underlying pipeline {pipe} is not callable.")
