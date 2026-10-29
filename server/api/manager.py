import io
import time
import base64
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Union, TypedDict, TYPE_CHECKING
from PIL import Image

from .schemas import Txt2ImgRequest, Img2ImgRequest, InpaintRequest
from ..infra.config import GPU_COUNT, GPU_NAME, DOWNLOAD_PATH
from ..infra.memory import free_memory_to_system
from ..infra.image import (
    pil_to_base64_data_uri,
    decode_image_data,
    convert_image_to_canny,
    convert_mask_image_to_rgb,
)
try:
    from ..engine import load_engine
except (ImportError, AttributeError):
    load_engine = None

from ..engine.base import BasePipelineEngine
from ..engine.types import InferenceKwargs
from diffusers import DiffusionPipeline

PipelineEngineType = BasePipelineEngine


class IpAdapterDict(TypedDict):
    uri: str
    subfolder: str
    weight_name: str
    image_encoder_folder: str
    strength: float


class ProgressDict(TypedDict):
    active: bool
    step: int
    total_steps: int
    image_index: int
    image_count: int


class GeneratedImageItem(TypedDict):
    image_base64: str
    seed: int
    width: int
    height: int


class GenerationInterrupted(Exception):
    """Raised when in-flight generation is cancelled by user."""
    pass


class PipelineManager:
    def __init__(self):
        self.pipelines: List[PipelineEngineType] = []
        self.current_model: Optional[str] = None
        self.model_version: Optional[str] = None
        self.lock = asyncio.Lock()
        self.download_kwargs: Dict[str, str] = {"directory": DOWNLOAD_PATH}
        self.ip_adapter: Optional[IpAdapterDict] = None
        self.loras_enabled: bool = True
        self.controlnet_model: Optional[str] = None
        self.cancel_requested: bool = False
        self.progress: ProgressDict = {
            "active": False,
            "step": 0,
            "total_steps": 0,
            "image_index": 0,
            "image_count": 0,
        }

    def cancel(self) -> bool:
        """Cancel any active generation and interrupt diffusion pipelines immediately."""
        if not self.progress.get("active"):
            return False
        self.cancel_requested = True
        for pipeline in self.pipelines:
            for sub in getattr(pipeline, "sub_pipelines", {}).values():
                pipe = getattr(sub, "pipeline", None)
                if pipe is not None:
                    try:
                        pipe._interrupt = True
                    except Exception:
                        pass
            pipe = getattr(pipeline, "pipeline", None)
            if pipe is not None:
                try:
                    pipe._interrupt = True
                except Exception:
                    pass
        return True

    def initialize_from_env(self) -> None:
        """Optionally initialize pipelines from environment variables."""
        if load_engine is None:
            print("Note: PyTorch CUDA pipeline is deferred or running in API mode.")
            return

        import os
        model_version = os.environ.get("MODEL_VERSION", "")
        model_url = os.environ.get("MODEL_URL", None)
        if not model_url:
            print("Note: No MODEL_URL provided, skipping auto-initialization.")
            return

        num_gpus = max(GPU_COUNT, 1)

        print(f"Initializing {num_gpus} pipeline(s) for model_version='{model_version}'...")
        try:
            self.pipelines = []
            for i in range(num_gpus):
                engine = load_engine(
                    model_url,
                    model_version=model_version,
                    download_kwargs=self.download_kwargs,
                )
                if GPU_COUNT > 0:
                    engine.to(f"cuda:{i}")
                self.pipelines.append(engine)
            self.model_version = model_version
            self.current_model = model_url
            print("Pipelines successfully initialized.")
        except Exception as e:
            print(f"Deferred pipeline initialization: {e}")

    async def load_model(self, model: str, model_version: str) -> Dict[str, str]:
        async with self.lock:
            loop = asyncio.get_running_loop()

            def _sync_load() -> List[PipelineEngineType]:
                num_gpus = max(GPU_COUNT, 1)
                for p in self.pipelines:
                    if hasattr(p, "clear"):
                        p.clear()

                new_pipelines: List[PipelineEngineType] = []
                for i in range(num_gpus):
                    engine = load_engine(
                        model,
                        model_version=model_version,
                        download_kwargs=self.download_kwargs,
                    )
                    if GPU_COUNT > 0 and hasattr(engine, "to"):
                        engine.to(f"cuda:{i}")
                    new_pipelines.append(engine)
                return new_pipelines

            self.pipelines = await loop.run_in_executor(None, _sync_load)
            self.current_model = model
            self.model_version = model_version
            self.ip_adapter = None
            self.loras_enabled = True
            self.controlnet_model = None
            free_memory_to_system()
            return {"status": "success", "model": model, "model_version": model_version}

    async def set_lora(self, url: str, lora_name: str, strength: float = 0.4) -> Dict[str, Union[str, float]]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model is currently loaded.")
            loop = asyncio.get_running_loop()

            def _sync_set_lora() -> None:
                for p in self.pipelines:
                    if hasattr(p, "set_lora"):
                        p.set_lora(url, lora_name, strength)

            await loop.run_in_executor(None, _sync_set_lora)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def delete_lora(self, lora_name: str) -> Dict[str, str]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model is currently loaded.")
            for p in self.pipelines:
                if hasattr(p, "delete_adapters"):
                    p.delete_adapters(lora_name)
            free_memory_to_system()
            return {"status": "success", "deleted_lora": lora_name}

    def get_loras(self) -> Dict[str, float]:
        if not self.pipelines:
            return {}
        return dict(getattr(self.pipelines[0], "lora_dict", {}))

    async def set_lora_strength(self, lora_name: str, strength: float) -> Dict[str, Union[str, float]]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
                if hasattr(pipeline, "set_lora_strength"):
                    pipeline.set_lora_strength(lora_name, strength)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def set_loras_enabled(self, enabled: bool) -> Dict[str, Union[str, bool]]:
        async with self.lock:
            self._require_pipelines()
            method_name = "enable_lora" if enabled else "disable_lora"
            for pipeline in self.pipelines:
                method = getattr(pipeline, method_name, None)
                if method is None:
                    raise RuntimeError(f"{method_name} is not available on the loaded pipeline.")
                method()
            self.loras_enabled = enabled
            return {"status": "success", "enabled": enabled}

    async def set_ip_adapter(
        self,
        uri: str,
        subfolder: str,
        weight_name: str,
        image_encoder_folder: str,
        strength: float
    ) -> Dict[str, Union[str, IpAdapterDict]]:
        async with self.lock:
            self._require_pipelines()
            if self._is_flux():
                raise RuntimeError("IP-Adapter is not supported for FLUX.")
            loop = asyncio.get_running_loop()

            def _sync_set_ip_adapter() -> None:
                for pipeline in self.pipelines:
                    if hasattr(pipeline, "set_ip_adapter"):
                        pipeline.set_ip_adapter(uri, subfolder, weight_name, image_encoder_folder)
                    if hasattr(pipeline, "set_ip_adapter_strength"):
                        pipeline.set_ip_adapter_strength(strength)

            await loop.run_in_executor(None, _sync_set_ip_adapter)
            adapter_info: IpAdapterDict = {
                "uri": uri,
                "subfolder": subfolder,
                "weight_name": weight_name,
                "image_encoder_folder": image_encoder_folder,
                "strength": strength,
            }
            self.ip_adapter = adapter_info
            return {"status": "success", "ip_adapter": self.ip_adapter}

    async def set_ip_adapter_strength(self, strength: float) -> Dict[str, Union[str, float]]:
        async with self.lock:
            self._require_pipelines()
            if not self.ip_adapter:
                raise RuntimeError("No IP-Adapter is loaded.")
            for pipeline in self.pipelines:
                if hasattr(pipeline, "set_ip_adapter_strength"):
                    pipeline.set_ip_adapter_strength(strength)
            self.ip_adapter = {**self.ip_adapter, "strength": strength}
            return {"status": "success", "strength": strength}

    async def delete_ip_adapter(self) -> Dict[str, str]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
                if hasattr(pipeline, "delete_ip_adapter"):
                    pipeline.delete_ip_adapter()
            self.ip_adapter = None
            free_memory_to_system()
            return {"status": "success"}

    async def load_controlnet(self, model: Optional[str]) -> Dict[str, Union[str, bool, None]]:
        async with self.lock:
            self._require_pipelines()
            self.controlnet_model = model or None
            loop = asyncio.get_running_loop()

            def _sync_load_controlnet() -> None:
                for pipeline in self.pipelines:
                    if hasattr(pipeline, "load_controlnet"):
                        pipeline.load_controlnet(self.controlnet_model)

            await loop.run_in_executor(None, _sync_load_controlnet)
            return {"status": "success", "controlnet_model": self.controlnet_model, "loaded": True}

    async def offload_controlnet(self) -> Dict[str, Union[str, bool]]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
                if hasattr(pipeline, "offload_controlnet"):
                    pipeline.offload_controlnet()
            return {"status": "success", "loaded": self.controlnet_is_loaded()}

    def controlnet_is_loaded(self) -> bool:
        if not self.pipelines:
            return False
        return self._pipeline_has_controlnet(self.pipelines[0])

    def preview_canny(self, image_str: str, low_threshold: int, high_threshold: int) -> str:
        image = decode_image_data(image_str).convert("RGB")
        preview = convert_image_to_canny(image, low_threshold, high_threshold)
        return pil_to_base64_data_uri(preview)

    def _require_pipelines(self) -> None:
        if not self.pipelines:
            raise RuntimeError("No model is currently loaded.")

    def _is_flux(self) -> bool:
        return self.model_version == "flux"

    def _pipeline_has_controlnet(self, pipeline: PipelineEngineType) -> bool:
        if getattr(pipeline, "_controlnet", None) is not None:
            return True
        try:
            get_adapters = getattr(pipeline, "get_list_adapters", None)
            adapters = get_adapters() if callable(get_adapters) else {}
        except Exception:
            return False
        if isinstance(adapters, dict):
            return "controlnet" in (adapters.get("transformer") or [])
        return False

    async def set_scheduler(self, scheduler: str, pipeline_type: str = "txt2img") -> Dict[str, str]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model is currently loaded.")
            for p in self.pipelines:
                target_sub = None
                if pipeline_type == "txt2img":
                    target_sub = getattr(p, "text_to_image_pipeline", None)
                elif pipeline_type == "img2img":
                    target_sub = getattr(p, "image_to_image_pipeline", None)
                elif pipeline_type == "inpaint":
                    target_sub = getattr(p, "inpainting_pipeline", None)

                if target_sub is not None and hasattr(target_sub, "set_scheduler"):
                    target_sub.set_scheduler(scheduler)
            return {"status": "success", "scheduler": scheduler, "pipeline_type": pipeline_type}

    def get_schedulers(self) -> List[str]:
        if not self.pipelines:
            return []
        sched_map = getattr(self.pipelines[0], "scheduler_map", {})
        return list(sched_map.keys()) if isinstance(sched_map, dict) else []

    async def _generate(
        self,
        pipeline_attr: str,
        kwargs: InferenceKwargs,
        seed: Optional[int],
        num: Optional[int],
        scheduler: Optional[str],
        use_controlnet: bool = False,
        controlnet_scale: Optional[float] = None
    ) -> Tuple[List[GeneratedImageItem], float]:
        async with self.lock:
            self._require_pipelines()
            controlnet_attrs = {
                "text_to_image_pipeline": "text_to_image_controlnet_pipeline",
                "image_to_image_pipeline": "image_to_image_controlnet_pipeline",
                "inpainting_pipeline": "inpainting_controlnet_pipeline",
            }
            if use_controlnet:
                for pipeline in self.pipelines:
                    if not self._pipeline_has_controlnet(pipeline) and hasattr(pipeline, "load_controlnet"):
                        pipeline.load_controlnet(self.controlnet_model)
                pipeline_attr = controlnet_attrs[pipeline_attr]
                if self._is_flux():
                    for pipeline in self.pipelines:
                        if hasattr(pipeline, "set_lora_strength"):
                            pipeline.set_lora_strength("controlnet", controlnet_scale if controlnet_scale is not None else 0.5)
                elif controlnet_scale is not None:
                    kwargs["controlnet_conditioning_scale"] = controlnet_scale

            if self._is_flux():
                kwargs.pop("negative_prompt", None)
                kwargs.pop("clip_skip", None)
                kwargs.pop("ip_adapter_image", None)

            sub_pipeline = getattr(self.pipelines[0], pipeline_attr)
            if scheduler and getattr(sub_pipeline, "scheduler_map", None):
                for pipeline in self.pipelines:
                    target = getattr(pipeline, pipeline_attr, None)
                    if target is not None and hasattr(target, "set_scheduler"):
                        target.set_scheduler(scheduler)

            num_count = max(1, min(int(num or 1), 10))
            seeds = [seed + i for i in range(num_count)] if seed else [random.randint(1, 2**31 - 1) for _ in range(num_count)]

            self.cancel_requested = False
            total_steps = int(kwargs.get("num_inference_steps") or 0)
            self.progress = {
                "active": True,
                "step": 0,
                "total_steps": total_steps,
                "image_index": 0,
                "image_count": len(seeds),
            }

            def callback_on_step_end(
                _pipe: DiffusionPipeline,
                step: int,
                _timestep: Union[int, float],
                callback_kwargs: Dict[str, Union[float, int, str]]
            ) -> Dict[str, Union[float, int, str]]:
                if self.cancel_requested:
                    if hasattr(_pipe, "_interrupt"):
                        _pipe._interrupt = True
                    raise GenerationInterrupted("Generation cancelled by user")
                self.progress = {**self.progress, "step": int(step) + 1}
                return callback_kwargs

            call_kwargs = {**kwargs, "callback_on_step_end": callback_on_step_end}
            start_time = time.time()
            loop = asyncio.get_running_loop()

            def run_seed(index: int, item_seed: int, pipeline: PipelineEngineType) -> Tuple[int, int, List[Image.Image]]:
                if self.cancel_requested:
                    raise GenerationInterrupted("Generation cancelled by user")
                self.progress = {**self.progress, "image_index": index, "step": 0}
                sub = getattr(pipeline, pipeline_attr)
                images: List[Image.Image] = sub.generate(seed=item_seed, num=1, **call_kwargs)
                return index, item_seed, images

            ordered: List[Tuple[int, int, List[Image.Image]]] = []
            try:
                if len(self.pipelines) == 1:
                    for index, item_seed in enumerate(seeds):
                        if self.cancel_requested:
                            break
                        res = await loop.run_in_executor(
                            None, run_seed, index, item_seed, self.pipelines[0]
                        )
                        ordered.append(res)
                else:
                    def run_batch() -> List[Tuple[int, int, List[Image.Image]]]:
                        with ThreadPoolExecutor(max_workers=len(self.pipelines)) as pool:
                            futures = [
                                pool.submit(run_seed, index, item_seed, self.pipelines[index % len(self.pipelines)])
                                for index, item_seed in enumerate(seeds)
                            ]
                            batch_results = []
                            for future in futures:
                                try:
                                    batch_results.append(future.result())
                                except GenerationInterrupted:
                                    break
                            return batch_results

                    ordered = await loop.run_in_executor(None, run_batch)
                    ordered.sort(key=lambda item: item[0])
            except GenerationInterrupted:
                pass
            finally:
                self.cancel_requested = False
                self.progress = {**self.progress, "active": False}

            result: List[GeneratedImageItem] = []
            for _index, item_seed, images in ordered:
                for img in images:
                    result.append({
                        "image_base64": pil_to_base64_data_uri(img),
                        "seed": item_seed,
                        "width": img.width,
                        "height": img.height,
                    })

            free_memory_to_system()
            return result, time.time() - start_time

    def _prepare_control_image(
        self,
        req: Union[Txt2ImgRequest, Img2ImgRequest, InpaintRequest],
        kwargs: InferenceKwargs,
        control_field: str
    ) -> bool:
        if not req.control_image:
            return False
        canny = convert_image_to_canny(
            decode_image_data(req.control_image).convert("RGB"),
            req.low_threshold,
            req.high_threshold,
        )
        if self._is_flux():
            kwargs["control_image"] = canny
        else:
            kwargs[control_field] = canny  # type: ignore
        return True

    def _prepare_ip_adapter_image(
        self,
        req: Union[Txt2ImgRequest, Img2ImgRequest, InpaintRequest],
        kwargs: InferenceKwargs
    ) -> None:
        if req.ip_adapter_image and not self._is_flux():
            kwargs["ip_adapter_image"] = decode_image_data(req.ip_adapter_image).convert("RGB")

    async def generate_txt2img(self, req: Txt2ImgRequest) -> Tuple[List[GeneratedImageItem], float]:
        kwargs: InferenceKwargs = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
            "clip_skip": req.clip_skip,
            "width": req.width,
            "height": req.height,
        }
        use_controlnet = self._prepare_control_image(req, kwargs, "image")
        self._prepare_ip_adapter_image(req, kwargs)
        return await self._generate(
            "text_to_image_pipeline",
            kwargs,
            req.seed,
            req.num,
            req.scheduler,
            use_controlnet,
            req.controlnet_conditioning_scale,
        )

    async def generate_img2img(self, req: Img2ImgRequest) -> Tuple[List[GeneratedImageItem], float]:
        kwargs: InferenceKwargs = {
            "image": decode_image_data(req.image).convert("RGB"),
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "strength": req.strength,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
            "clip_skip": req.clip_skip,
            "width": req.width,
            "height": req.height,
        }
        use_controlnet = self._prepare_control_image(req, kwargs, "control_image")
        self._prepare_ip_adapter_image(req, kwargs)
        return await self._generate(
            "image_to_image_pipeline",
            kwargs,
            req.seed,
            req.num,
            req.scheduler,
            use_controlnet,
            req.controlnet_conditioning_scale,
        )

    async def generate_inpaint(self, req: InpaintRequest) -> Tuple[List[GeneratedImageItem], float]:
        mask_raw = decode_image_data(req.mask_image)
        if mask_raw.mode == "RGBA":
            mask_image = convert_mask_image_to_rgb(mask_raw)
        else:
            mask_image = mask_raw.convert("RGB")

        kwargs: InferenceKwargs = {
            "image": decode_image_data(req.image).convert("RGB"),
            "mask_image": mask_image,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "strength": req.strength,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
            "clip_skip": req.clip_skip,
            "width": req.width,
            "height": req.height,
        }
        use_controlnet = self._prepare_control_image(req, kwargs, "control_image")
        self._prepare_ip_adapter_image(req, kwargs)
        return await self._generate(
            "inpainting_pipeline",
            kwargs,
            req.seed,
            req.num,
            req.scheduler,
            use_controlnet,
            req.controlnet_conditioning_scale,
        )


manager = PipelineManager()
