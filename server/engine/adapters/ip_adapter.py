from __future__ import annotations
from typing import Dict, Union
from diffusers import DiffusionPipeline

from ...infra import config
from ...infra.memory import free_memory_to_system


class IPAdapterMixin:
    def set_ip_adapter(
        self,
        uri: str,
        subfolder: str,
        weight_name: str,
        image_encoder_folder: str = "image_encoder",
        **kwargs: Union[str, float, int, bool]
    ) -> None:
        target_pipe: DiffusionPipeline = getattr(self, "pipeline", self)
        download_kwargs: Dict[str, str] = getattr(self, "download_kwargs", {})
        cache_dir = download_kwargs.get("directory", config.DOWNLOAD_PATH)

        if not hasattr(target_pipe, "load_ip_adapter"):
            raise AttributeError(f"Pipeline {target_pipe} does not support load_ip_adapter")

        target_pipe.load_ip_adapter(
            uri,
            subfolder=subfolder,
            weight_name=weight_name,
            image_encoder_folder=image_encoder_folder,
            cache_dir=cache_dir,
            token=config.HF_HUB_TOKEN,
            **kwargs
        )

        sub_pipelines: Dict[str, EngineType] = getattr(self, "sub_pipelines", {})
        for sub_p in sub_pipelines.values():
            underlying = getattr(sub_p, "pipeline", sub_p)
            if hasattr(underlying, "register_modules"):
                if hasattr(target_pipe, "image_encoder"):
                    underlying.register_modules(image_encoder=target_pipe.image_encoder)
                if hasattr(target_pipe, "feature_extractor"):
                    underlying.register_modules(feature_extractor=target_pipe.feature_extractor)

    def set_ip_adapter_strength(self, weight: float) -> None:
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "set_ip_adapter_scale"):
            target_pipe.set_ip_adapter_scale(weight)

    def delete_ip_adapter(self) -> None:
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "unload_ip_adapter"):
            target_pipe.unload_ip_adapter()

        sub_pipelines: Dict[str, EngineType] = getattr(self, "sub_pipelines", {})
        for sub_p in sub_pipelines.values():
            underlying = getattr(sub_p, "pipeline", sub_p)
            if hasattr(underlying, "unload_ip_adapter"):
                underlying.unload_ip_adapter()

        free_memory_to_system()
