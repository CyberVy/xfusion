from __future__ import annotations

import inspect
import os
import random
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import SchedulerMixin
from PIL import Image

from ..infra import config
from ..infra.downloader import download_file
from ..infra.image import normalize_image
from ..infra.memory import delete, free_memory_to_system
from ..models.inspect import apply_execution_device, use_gpu
from .contracts import (
    FeatureNotSupported,
    GenerationInterrupted,
    GenerationRequest,
    GenerationResult,
    IpAdapterSource,
    LoadedIpAdapter,
    ModelFeatures,
    ModelLoadSettings,
    StepCallback,
)
from .schedulers import DIFFUSION_SCHEDULER_MAP


class _ProgressBarWrapper:
    """Wraps Diffusers progress bar for pipelines that lack native callback support."""

    def __init__(self, pbar: Any, on_step: StepCallback, total_steps: int, pipe: Any):
        self._pbar = pbar
        self._on_step = on_step
        self._total_steps = total_steps
        self._pipe = pipe
        self._current_step = 0

    def __iter__(self):
        it = iter(self._pbar)
        step = 0
        while True:
            if getattr(self._pipe, "_interrupt", False):
                raise GenerationInterrupted("Generation cancelled by user")
            try:
                item = next(it)
            except StopIteration:
                break
            yield item
            step += 1
            if self._on_step is not None:
                self._on_step(step, self._total_steps)

    def __enter__(self):
        if hasattr(self._pbar, "__enter__"):
            self._pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._pbar, "__exit__"):
            return self._pbar.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __len__(self) -> int:
        if hasattr(self._pbar, "__len__"):
            return len(self._pbar)
        return self._total_steps

    def update(self, n: int = 1) -> None:
        if hasattr(self._pbar, "update"):
            self._pbar.update(n)
        if getattr(self._pipe, "_interrupt", False):
            raise GenerationInterrupted("Generation cancelled by user")
        self._current_step += n
        if self._on_step is not None:
            current = getattr(self._pbar, "n", self._current_step)
            self._on_step(current, self._total_steps)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pbar, name)


class LoadedModel:
    """One loaded diffusion model. Callers never see the Diffusers pipeline."""

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        family: Any,
        settings: ModelLoadSettings,
        features: ModelFeatures,
        gpu_id: int,
    ):
        self.pipeline = pipeline
        self.family = family
        self.settings = settings
        self.features = features
        self.model_name = settings.source
        self.gpu_id = gpu_id
        self.download_kwargs: Dict[str, str] = {"directory": config.DOWNLOAD_PATH}
        self.loras: Dict[str, float] = {}
        self.loras_enabled = True
        self.ip_adapter: Optional[LoadedIpAdapter] = None
        self._controlnet = None
        self._task_pipes: Dict[Tuple[str, bool], DiffusionPipeline] = {}
        self._active_pipe: Optional[DiffusionPipeline] = pipeline
        self._embed_cache: OrderedDict[Tuple, Dict[str, torch.Tensor]] = OrderedDict()
        self._scheduler: Optional[SchedulerMixin] = getattr(pipeline, "scheduler", None)

    @property
    def controlnet_loaded(self) -> bool:
        return self.family.controlnet_is_loaded(self)

    @property
    def execution_device(self):
        if self.pipeline is None:
            return None
        return getattr(self.pipeline, "_execution_device", None) or getattr(self.pipeline, "device", None)

    def set_download_kwargs(self, **kwargs: str) -> None:
        self.download_kwargs.update(kwargs)

    def generate(
        self,
        request: GenerationRequest,
        *,
        on_step: Optional[StepCallback] = None,
    ) -> GenerationResult:
        if request.task not in self.features.tasks:
            raise FeatureNotSupported(request.task)
        if request.control_image is not None and "controlnet" not in self.features.adapters:
            raise FeatureNotSupported("controlnet")
        if request.reference_image is not None:
            if "ip_adapter" not in self.features.adapters:
                raise FeatureNotSupported("ip_adapter")
            if self.ip_adapter is None:
                raise RuntimeError("IP-Adapter is not loaded.")
        if request.scheduler:
            self.set_scheduler(request.scheduler)

        prepared = self._fit_request(request)
        seed = prepared.seed if prepared.seed is not None else random.randint(1, 2**31 - 1)
        restore_ip_strength = None
        if self.ip_adapter is not None and prepared.reference_image is None:
            prepared = replace(
                prepared,
                reference_image=Image.new("RGB", (prepared.width, prepared.height), (0, 0, 0)),
            )
            restore_ip_strength = self.ip_adapter.strength
            self.set_ip_adapter_strength(0.0)

        with use_gpu(self.gpu_id), self._prompt_gated_loras(prepared.prompt, prepared.negative_prompt):
            orig_progress_bar = None
            pipe = None
            try:
                call_kwargs = self.family.build_generation_inputs(self, prepared)
                generator_device = self.execution_device
                custom_device = getattr(self.family, "generator_device", None)
                if callable(custom_device):
                    generator_device = custom_device(self)
                if generator_device is not None:
                    call_kwargs["generator"] = torch.Generator(generator_device).manual_seed(seed)

                pipe = self._pipeline_for_task(prepared.task, controlnet=prepared.control_image is not None)
                self._active_pipe = pipe
                if on_step is not None:
                    params = inspect.signature(pipe.__call__).parameters
                    accepts_callback = "callback_on_step_end" in params or any(
                        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
                    )
                    total_steps = int(prepared.steps)
                    if accepts_callback:
                        def callback_on_step_end(_pipe, step, _timestep, callback_kwargs):
                            on_step(int(step) + 1, total_steps)
                            return callback_kwargs

                        call_kwargs["callback_on_step_end"] = callback_on_step_end
                    elif "callback" in params:
                        call_kwargs["callback"] = lambda step, _timestep, _latents: on_step(int(step) + 1, total_steps)
                        if "callback_steps" in params:
                            call_kwargs["callback_steps"] = 1
                    elif hasattr(pipe, "progress_bar"):
                        orig_progress_bar = pipe.progress_bar

                        def custom_progress_bar(iterable=None, total=None):
                            pbar = orig_progress_bar(iterable=iterable, total=total)
                            pbar_total = total or (
                                len(iterable)
                                if iterable is not None and hasattr(iterable, "__len__")
                                else total_steps
                            )
                            return _ProgressBarWrapper(pbar, on_step, pbar_total, pipe)

                        pipe.progress_bar = custom_progress_bar
                output = pipe(**call_kwargs)
            finally:
                if orig_progress_bar is not None and pipe is not None and hasattr(pipe, "progress_bar"):
                    pipe.progress_bar = orig_progress_bar
                self._active_pipe = self.pipeline
                if restore_ip_strength is not None:
                    self.set_ip_adapter_strength(restore_ip_strength)
                if self.pipeline is not None and hasattr(self.pipeline, "maybe_free_model_hooks"):
                    try:
                        self.pipeline.maybe_free_model_hooks()
                    except Exception:
                        pass

        images = getattr(output, "images", None)
        if not isinstance(images, list) or not images:
            raise RuntimeError("The pipeline did not return an image.")
        return GenerationResult(image=images[0], seed=seed)

    def interrupt(self) -> None:
        for pipe in (self._active_pipe, self.pipeline):
            if pipe is not None:
                try:
                    pipe._interrupt = True
                except Exception:
                    pass

    def clear(self) -> None:
        self._embed_cache.clear()
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
        self.loras.clear()
        self.ip_adapter = None
        free_memory_to_system()

    def set_scheduler(self, name: str) -> None:
        if name not in self.features.schedulers:
            raise FeatureNotSupported(f"scheduler {name}")
        if self.pipeline is None:
            raise RuntimeError("Cannot set scheduler on an unloaded model.")
        sched_cls, default_kwargs = DIFFUSION_SCHEDULER_MAP[name]
        config_source = self._scheduler or getattr(self.pipeline, "scheduler", None)
        scheduler_config = getattr(config_source, "config", None)
        from_config = getattr(sched_cls, "from_config", None)
        if from_config is None:
            raise FeatureNotSupported(f"scheduler {name}")
        self.pipeline.scheduler = from_config(scheduler_config, **default_kwargs)

    def set_lora(self, source: str, name: str, strength: float) -> None:
        if "lora" not in self.features.adapters:
            raise FeatureNotSupported("lora")
        if name not in self.loras:
            with use_gpu(self.gpu_id):
                self._load_lora_weights(source, name)
            self.loras[name] = 0.0
        self.set_lora_strength(name, strength)

    def set_lora_strength(self, name: str, strength: float) -> None:
        if name not in self.loras:
            raise RuntimeError(f"LoRA '{name}' is not loaded.")
        previous = self.loras[name]
        self.loras[name] = strength
        try:
            if self.pipeline is not None and hasattr(self.pipeline, "set_adapters"):
                self.pipeline.set_adapters(list(self.loras.keys()), list(self.loras.values()))
        except Exception:
            self.loras[name] = previous
            raise

    def delete_lora(self, name: str) -> None:
        if self.pipeline is not None and hasattr(self.pipeline, "delete_adapters"):
            self.pipeline.delete_adapters(name)
        self.loras.pop(name, None)
        self._embed_cache.clear()
        free_memory_to_system()

    def set_loras_enabled(self, enabled: bool) -> None:
        self.loras_enabled = enabled
        if self.pipeline is None:
            return
        if enabled and hasattr(self.pipeline, "enable_lora"):
            self.pipeline.enable_lora()
        elif not enabled and hasattr(self.pipeline, "disable_lora"):
            self.pipeline.disable_lora()
        self._embed_cache.clear()

    def load_ip_adapter(self, source: IpAdapterSource, strength: float) -> None:
        if "ip_adapter" not in self.features.adapters:
            raise FeatureNotSupported("ip_adapter")
        if self.pipeline is None or not hasattr(self.pipeline, "load_ip_adapter"):
            raise FeatureNotSupported("ip_adapter")

        cache_dir = self.download_kwargs.get("directory", config.DOWNLOAD_PATH)
        with use_gpu(self.gpu_id):
            self.pipeline.load_ip_adapter(
                source.source,
                subfolder=source.folder,
                weight_name=source.weight,
                image_encoder_folder=source.encoder_folder,
                cache_dir=cache_dir,
                token=config.HF_HUB_TOKEN,
            )
            self._place_image_encoder()
        self._task_pipes.clear()
        self._embed_cache.clear()
        self.ip_adapter = LoadedIpAdapter(
            source=source.source,
            folder=source.folder,
            weight=source.weight,
            encoder_folder=source.encoder_folder,
            strength=strength,
        )
        self.set_ip_adapter_strength(strength)

    def set_ip_adapter_strength(self, strength: float) -> None:
        if self.ip_adapter is None:
            raise RuntimeError("No IP-Adapter is loaded.")
        if self.pipeline is not None and hasattr(self.pipeline, "set_ip_adapter_scale"):
            self.pipeline.set_ip_adapter_scale(strength)
        self.ip_adapter = LoadedIpAdapter(
            source=self.ip_adapter.source,
            folder=self.ip_adapter.folder,
            weight=self.ip_adapter.weight,
            encoder_folder=self.ip_adapter.encoder_folder,
            strength=strength,
        )

    def clear_ip_adapter(self) -> None:
        if self.pipeline is not None and hasattr(self.pipeline, "unload_ip_adapter"):
            self.pipeline.unload_ip_adapter()
        self.ip_adapter = None
        self._task_pipes.clear()
        self._embed_cache.clear()
        free_memory_to_system()

    def load_controlnet(self, source: Optional[str] = None) -> None:
        if "controlnet" not in self.features.adapters:
            raise FeatureNotSupported("controlnet")
        with use_gpu(self.gpu_id):
            self.family.load_controlnet(self, source)

    def clear_controlnet(self) -> None:
        self.family.clear_controlnet(self)

    def cached_embeds(self, key: Tuple, encode) -> Dict[str, torch.Tensor]:
        if key in self._embed_cache:
            self._embed_cache.move_to_end(key)
            cached = self._embed_cache[key]
            device = self.execution_device
            if device is not None:
                return {name: tensor.to(device) for name, tensor in cached.items()}
            return dict(cached)

        embeds = encode()
        if not embeds:
            return {}
        cpu_copy = {name: tensor.detach().cpu() for name, tensor in embeds.items() if hasattr(tensor, "detach")}
        if cpu_copy:
            self._embed_cache[key] = cpu_copy
            limit = getattr(self.family, "embed_cache_limit", 8)
            while len(self._embed_cache) > limit:
                self._embed_cache.popitem(last=False)
        return dict(embeds)

    def _fit_request(self, request: GenerationRequest) -> GenerationRequest:
        divisor = self.features.size_divisor
        default_dim = self.features.default_size
        negative = request.negative_prompt if self.features.accepts_negative_prompt else ""
        clip_skip = request.clip_skip if self.features.accepts_clip_skip else None
        width = request.width
        height = request.height
        image = request.image

        if image is not None:
            if width is None and height is None:
                width = image.width - (image.width % divisor)
                height = image.height - (image.height % divisor)
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                target_w = width or default_dim
                target_h = height or default_dim
                image = normalize_image(image, target_w * target_h, scale_divisor=divisor)
                width = image.width
                height = image.height
        else:
            width = width or default_dim
            height = height or default_dim

        mask = _resize_exact(request.mask, width, height)
        control_image = _resize_exact(request.control_image, width, height)
        reference = request.reference_image
        if reference is not None:
            reference = normalize_image(reference, width * height, scale_divisor=divisor)

        guidance = request.guidance if request.guidance is not None else self.features.default_guidance
        return GenerationRequest(
            task=request.task,
            prompt=request.prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            steps=request.steps,
            guidance=guidance,
            seed=request.seed,
            clip_skip=clip_skip,
            scheduler=request.scheduler,
            image_count=1,
            image=image,
            mask=mask,
            strength=request.strength,
            control_image=control_image,
            control_scale=request.control_scale,
            canny_low=request.canny_low,
            canny_high=request.canny_high,
            reference_image=reference,
        )

    def _pipeline_for_task(self, task: str, *, controlnet: bool) -> DiffusionPipeline:
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded.")

        target_cls = self.family.pipeline_for_task(self.features.model_version, task, controlnet)
        if self.pipeline.__class__ is target_cls and (not controlnet or self._controlnet is None):
            return self.pipeline

        cache_key = (task, controlnet)
        cached = self._task_pipes.get(cache_key)
        if cached is not None:
            if hasattr(self.pipeline, "scheduler"):
                cached.scheduler = self.pipeline.scheduler
            if hasattr(self.pipeline, "_all_hooks"):
                cached._all_hooks = self.pipeline._all_hooks
            if hasattr(self.pipeline, "_offload_gpu_id"):
                cached._offload_gpu_id = self.pipeline._offload_gpu_id
            return cached

        sig_params = inspect.signature(target_cls.__init__).parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig_params.values())
        raw_components = dict(getattr(self.pipeline, "components", {}) or {})
        if controlnet and self._controlnet is not None:
            raw_components["controlnet"] = self._controlnet
        if not accepts_kwargs:
            raw_components = {key: value for key, value in raw_components.items() if key in sig_params}

        derived = target_cls(**raw_components)
        if hasattr(self.pipeline, "_all_hooks"):
            derived._all_hooks = self.pipeline._all_hooks
        if hasattr(self.pipeline, "_offload_gpu_id"):
            derived._offload_gpu_id = self.pipeline._offload_gpu_id
        self._task_pipes[cache_key] = derived
        return derived

    def _load_lora_weights(self, source: str, name: str) -> None:
        if self.pipeline is None or not hasattr(self.pipeline, "load_lora_weights"):
            raise FeatureNotSupported("lora")
        download_kwargs = dict(self.download_kwargs)
        directory = download_kwargs.get("directory", "./")
        download_kwargs["directory"] = os.path.join(directory, "lora")
        if source.startswith(("http://", "https://")):
            lora_path = download_file(source, **download_kwargs)
            self.pipeline.load_lora_weights(lora_path, adapter_name=name)
        else:
            self.pipeline.load_lora_weights(os.path.expanduser(source), adapter_name=name)
        self._embed_cache.clear()

    def _place_image_encoder(self) -> None:
        image_encoder = getattr(self.pipeline, "image_encoder", None)
        exec_device = self.execution_device
        if image_encoder is None or exec_device is None or self.pipeline is None:
            return
        memory = self.settings.memory
        if memory == "offload":
            try:
                from accelerate import cpu_offload_with_hook

                _, hook = cpu_offload_with_hook(image_encoder, execution_device=exec_device)
                if hasattr(self.pipeline, "_all_hooks"):
                    self.pipeline._all_hooks.append(hook)
                return
            except Exception:
                pass
        elif memory == "sequential":
            try:
                from accelerate import cpu_offload

                cpu_offload(image_encoder, execution_device=exec_device)
                return
            except Exception:
                pass
        image_encoder.to(exec_device)

    @contextmanager
    def _prompt_gated_loras(self, prompt: str, negative_prompt: str) -> Iterator[None]:
        if not getattr(self.family, "prompt_activates_lora", False) or not self.loras_enabled or not self.loras:
            yield
            return

        prompt_text = f"{prompt or ''} {negative_prompt or ''}"
        skipped: Dict[str, float] = {}
        for name, strength in list(self.loras.items()):
            if name not in prompt_text:
                skipped[name] = strength
                self.set_lora_strength(name, 0)

        disabled_all = False
        if skipped and len(skipped) == len(self.loras):
            self.set_loras_enabled(False)
            disabled_all = True
        try:
            yield
        finally:
            if disabled_all:
                self.set_loras_enabled(True)
            for name, strength in skipped.items():
                self.set_lora_strength(name, strength)


def _resize_exact(image: Optional[Image.Image], width: int, height: int) -> Optional[Image.Image]:
    if image is None:
        return None
    if image.size != (width, height):
        return image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def move_to_gpu(pipeline: DiffusionPipeline, settings: ModelLoadSettings, gpu_id: int) -> DiffusionPipeline:
    return apply_execution_device(pipeline, memory=settings.memory, gpu_id=gpu_id)
