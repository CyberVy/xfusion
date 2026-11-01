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
        image_encoder = getattr(target_pipe, "image_encoder", None)
        offload_mode = getattr(self, "offload_mode", "none")
        exec_device = getattr(self, "execution_device", None)
        if image_encoder is not None and exec_device is not None:
            if offload_mode == "model_cpu_offload":
                try:
                    from accelerate import cpu_offload_with_hook
                    _, hook = cpu_offload_with_hook(image_encoder, execution_device=exec_device)
                    if hasattr(target_pipe, "_all_hooks"):
                        target_pipe._all_hooks.append(hook)
                except Exception:
                    image_encoder.to(exec_device)
            elif offload_mode == "sequential_cpu_offload":
                try:
                    from accelerate import cpu_offload
                    cpu_offload(image_encoder, execution_device=exec_device)
                except Exception:
                    image_encoder.to(exec_device)
            elif offload_mode == "none":
                image_encoder.to(exec_device)

        task_pipes = getattr(self, "_task_pipes", None)
        if isinstance(task_pipes, dict):
            task_pipes.clear()

    def set_ip_adapter_strength(self, weight: float) -> None:
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "set_ip_adapter_scale"):
            target_pipe.set_ip_adapter_scale(weight)

    def delete_ip_adapter(self) -> None:
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "unload_ip_adapter"):
            target_pipe.unload_ip_adapter()
        task_pipes = getattr(self, "_task_pipes", None)
        if isinstance(task_pipes, dict):
            task_pipes.clear()
        free_memory_to_system()
