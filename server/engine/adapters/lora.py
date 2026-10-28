import os
from typing import Dict, Union, List
from diffusers import DiffusionPipeline

from ...infra.downloader import download_file, DownloadArgumentsMixin
from ...infra.memory import free_memory_to_system


def load_lora_weights(
    pipeline: DiffusionPipeline,
    lora_uri: str,
    lora_name: str,
    download_kwargs: Dict[str, str],
    **kwargs: Union[str, float, int, bool]
) -> None:
    use_internet = not (lora_uri.startswith(".") or lora_uri.startswith("/") or lora_uri.startswith("~"))
    if not hasattr(pipeline, "load_lora_weights"):
        raise AttributeError(f"Pipeline {pipeline} does not support load_lora_weights")

    if use_internet:
        lora_path = download_file(lora_uri, **download_kwargs)
        pipeline.load_lora_weights(lora_path, adapter_name=lora_name, **kwargs)
    else:
        pipeline.load_lora_weights(lora_uri, adapter_name=lora_name, **kwargs)


class LoraAdapterMixin(DownloadArgumentsMixin):
    def __init__(self):
        super().__init__()
        self.lora_dict: Dict[str, float] = {}

    def set_lora(self, lora_uri: str, lora_name: str, weight: float = 0.4) -> None:
        target_pipe: DiffusionPipeline = getattr(self, "pipeline", self)
        if lora_name not in self.lora_dict:
            download_kwargs = self.download_kwargs.copy()
            directory = download_kwargs.get("directory", "./")
            directory = os.path.join(directory, "lora")
            download_kwargs["directory"] = directory

            load_lora_weights(target_pipe, lora_uri, lora_name, download_kwargs)

        self.set_lora_strength(lora_name, weight)

    def set_lora_strength(self, lora_name: str, weight: float) -> None:
        self.lora_dict[lora_name] = weight
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "set_adapters"):
            target_pipe.set_adapters(list(self.lora_dict.keys()), list(self.lora_dict.values()))

    def delete_adapters(self, adapter_names: Union[str, List[str]]) -> None:
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "delete_adapters"):
            target_pipe.delete_adapters(adapter_names)

        names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
        for name in names:
            self.lora_dict.pop(name, None)
        free_memory_to_system()

    def enable_lora(self) -> None:
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "enable_lora"):
            target_pipe.enable_lora()

    def disable_lora(self) -> None:
        target_pipe = getattr(self, "pipeline", self)
        if hasattr(target_pipe, "disable_lora"):
            target_pipe.disable_lora()
