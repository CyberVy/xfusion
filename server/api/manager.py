import time
import random
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Union, TypedDict, AsyncGenerator, Any
from PIL import Image

from .schemas import Txt2ImgRequest, Img2ImgRequest, InpaintRequest
from ..infra.config import GPU_COUNT, DOWNLOAD_PATH
from ..infra.memory import free_memory_to_system
from ..infra.image import (
    pil_to_base64_data_uri,
    decode_image_data,
    convert_image_to_canny,
    convert_mask_image_to_rgb,
)
from ..engine import load_engine
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
        self.quantization: str = "none"
        self.text_encoder_mode: str = "default"
        self.offload_mode: str = "none"
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
            pipeline.interrupt()
        return True

    def initialize_from_env(self) -> None:
        """Optionally initialize pipelines from environment variables."""
        import os
        model_version = os.environ.get("MODEL_VERSION", "")
        model_url = os.environ.get("MODEL_URL", None)
        quantization = os.environ.get("QUANTIZATION", "none")
        text_encoder_mode = os.environ.get("TEXT_ENCODER_MODE", "default")
        offload_mode = os.environ.get("OFFLOAD_MODE", "none")
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
                    quantization=quantization,
                    text_encoder_mode=text_encoder_mode,
                    offload_mode=offload_mode,
                    gpu_id=i,
                    download_kwargs=self.download_kwargs,
                )
                self.pipelines.append(engine)
            resolved_ver = model_version or (
                getattr(self.pipelines[0], "model_version", None) if self.pipelines else ""
            )
            self.model_version = resolved_ver
            self.current_model = model_url
            self.quantization = quantization
            self.text_encoder_mode = text_encoder_mode
            self.offload_mode = offload_mode
            print("Pipelines successfully initialized.")
        except Exception as e:
            print(f"Deferred pipeline initialization: {e}")

    async def load_model(
        self,
        model: str,
        model_version: str = "xl",
        quantization: str = "none",
        text_encoder_mode: str = "default",
        offload_mode: str = "none",
        delete_after_load: bool = False,
    ) -> Dict[str, Union[str, bool]]:
        async with self.lock:
            loop = asyncio.get_running_loop()

            def _sync_load() -> List[PipelineEngineType]:
                num_gpus = max(GPU_COUNT, 1)
                for p in self.pipelines:
                    p.clear()
                self.pipelines = []
                free_memory_to_system()

                new_pipelines: List[PipelineEngineType] = []
                for i in range(num_gpus):
                    engine = load_engine(
                        model,
                        model_version=model_version,
                        quantization=quantization,
                        text_encoder_mode=text_encoder_mode,
                        offload_mode=offload_mode,
                        delete_after_load=delete_after_load,
                        gpu_id=i,
                        download_kwargs=self.download_kwargs,
                    )
                    new_pipelines.append(engine)
                return new_pipelines

            self.pipelines = await loop.run_in_executor(None, _sync_load)
            resolved_ver = model_version or (
                getattr(self.pipelines[0], "model_version", None) if self.pipelines else "xl"
            )
            self.current_model = model
            self.model_version = resolved_ver
            self.quantization = quantization
            self.text_encoder_mode = text_encoder_mode
            self.offload_mode = offload_mode
            self.ip_adapter = None
            self.loras_enabled = True
            self.controlnet_model = None
            free_memory_to_system()
            return {
                "status": "success",
                "model": model,
                "model_version": str(resolved_ver),
                "quantization": quantization,
                "text_encoder_mode": text_encoder_mode,
                "offload_mode": offload_mode,
            }

    async def set_lora(self, url: str, lora_name: str, strength: float = 0.4) -> Dict[str, Union[str, float]]:
        async with self.lock:
            self._require_pipelines()
            loop = asyncio.get_running_loop()

            def _sync_set_lora() -> None:
                for p in self.pipelines:
                    p.set_lora(url, lora_name, strength)

            await loop.run_in_executor(None, _sync_set_lora)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def delete_lora(self, lora_name: str) -> Dict[str, str]:
        async with self.lock:
            self._require_pipelines()
            for p in self.pipelines:
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
                pipeline.set_lora_strength(lora_name, strength)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def set_loras_enabled(self, enabled: bool) -> Dict[str, Union[str, bool]]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
                if enabled:
                    pipeline.enable_lora()
                else:
                    pipeline.disable_lora()
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
                    pipeline.set_ip_adapter(uri, subfolder, weight_name, image_encoder_folder)
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
                pipeline.set_ip_adapter_strength(strength)
            self.ip_adapter = {**self.ip_adapter, "strength": strength}
            return {"status": "success", "strength": strength}

    async def delete_ip_adapter(self) -> Dict[str, str]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
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
                    pipeline.load_controlnet(self.controlnet_model)

            await loop.run_in_executor(None, _sync_load_controlnet)
            return {"status": "success", "controlnet_model": self.controlnet_model, "loaded": True}

    async def offload_controlnet(self) -> Dict[str, Union[str, bool]]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
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

    async def set_scheduler(self, scheduler: str) -> Dict[str, str]:
        async with self.lock:
            self._require_pipelines()
            for p in self.pipelines:
                p.set_scheduler(scheduler)
            return {"status": "success", "scheduler": scheduler}

    def get_schedulers(self) -> List[str]:
        if not self.pipelines:
            return []
        sched_map = getattr(self.pipelines[0], "scheduler_map", {})
        return list(sched_map.keys()) if isinstance(sched_map, dict) else []

    def _prepare_generation_context(
        self,
        kwargs: InferenceKwargs,
        scheduler: Optional[str],
        use_controlnet: bool,
        controlnet_scale: Optional[float],
    ) -> None:
        if use_controlnet:
            for pipeline in self.pipelines:
                if not self._pipeline_has_controlnet(pipeline):
                    pipeline.load_controlnet(self.controlnet_model)
            if self._is_flux():
                for pipeline in self.pipelines:
                    pipeline.set_lora_strength(
                        "controlnet", controlnet_scale if controlnet_scale is not None else 0.5
                    )
            elif controlnet_scale is not None:
                kwargs["controlnet_conditioning_scale"] = controlnet_scale

        if self._is_flux():
            kwargs.pop("negative_prompt", None)
            kwargs.pop("clip_skip", None)
            kwargs.pop("ip_adapter_image", None)

        if scheduler and getattr(self.pipelines[0], "scheduler_map", None):
            for pipeline in self.pipelines:
                pipeline.set_scheduler(scheduler)

    async def _iter_generated_images(
        self,
        task: str,
        kwargs: InferenceKwargs,
        seed: Optional[int],
        num: Optional[int],
        scheduler: Optional[str],
        use_controlnet: bool = False,
        controlnet_scale: Optional[float] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        async with self.lock:
            self._require_pipelines()
            self._prepare_generation_context(kwargs, scheduler, use_controlnet, controlnet_scale)

            num_count = max(1, min(int(num or 1), 10))
            seeds = (
                [seed + i for i in range(num_count)]
                if seed
                else [random.randint(1, 2**31 - 1) for _ in range(num_count)]
            )

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
                callback_kwargs: Dict[str, Union[float, int, str]],
            ) -> Dict[str, Union[float, int, str]]:
                if self.cancel_requested:
                    if hasattr(_pipe, "_interrupt"):
                        _pipe._interrupt = True
                    raise GenerationInterrupted("Generation cancelled by user")
                self.progress = {**self.progress, "step": int(step) + 1}
                return callback_kwargs

            call_kwargs = {**kwargs, "callback_on_step_end": callback_on_step_end}
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()

            num_pipelines = max(1, len(self.pipelines))
            assignments: List[List[Tuple[int, int]]] = [[] for _ in range(num_pipelines)]
            for index, item_seed in enumerate(seeds):
                assignments[index % num_pipelines].append((index, item_seed))

            def run_pipeline_worker(p_idx: int, task_list: List[Tuple[int, int]]) -> None:
                pipeline = self.pipelines[p_idx]
                for index, item_seed in task_list:
                    if self.cancel_requested:
                        break
                    try:
                        t0 = time.time()
                        self.progress = {**self.progress, "image_index": index, "step": 0}
                        images: List[Image.Image] = pipeline.generate(
                            seed=item_seed,
                            num=1,
                            task=task,
                            use_controlnet=use_controlnet,
                            **call_kwargs,
                        )
                        dt = time.time() - t0
                        for img in images:
                            item = {
                                "image_base64": pil_to_base64_data_uri(img),
                                "seed": item_seed,
                                "width": img.width,
                                "height": img.height,
                                "index": index,
                                "elapsed_seconds": round(dt, 3),
                            }
                            loop.call_soon_threadsafe(queue.put_nowait, ("image", item))
                    except GenerationInterrupted:
                        break
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                        break

            active_workers = [p_idx for p_idx in range(num_pipelines) if assignments[p_idx]]
            pool = ThreadPoolExecutor(max_workers=max(1, len(active_workers)))
            for p_idx in active_workers:
                pool.submit(run_pipeline_worker, p_idx, assignments[p_idx])

            completed_count = 0
            total_items = len(seeds)
            try:
                while completed_count < total_items and not self.cancel_requested:
                    msg_type, payload = await queue.get()
                    if msg_type == "image":
                        completed_count += 1
                        yield ("image", payload)
                    elif msg_type == "error":
                        yield ("error", payload)
                        break
            finally:
                self.cancel_requested = True
                pool.shutdown(wait=False, cancel_futures=True)
                self.progress = {**self.progress, "active": False}
                free_memory_to_system()

    async def _generate(
        self,
        task: str,
        req: Union[Txt2ImgRequest, Img2ImgRequest, InpaintRequest],
    ) -> Tuple[List[GeneratedImageItem], float]:
        kwargs, use_controlnet = self._build_request_kwargs(task, req)
        start_time = time.time()
        indexed_items: List[Tuple[int, GeneratedImageItem]] = []
        async for msg_type, payload in self._iter_generated_images(
            task,
            kwargs,
            req.seed,
            req.num,
            req.scheduler,
            use_controlnet,
            req.controlnet_conditioning_scale,
        ):
            if msg_type == "error":
                raise RuntimeError(str(payload))
            if msg_type == "image":
                indexed_items.append(
                    (
                        int(payload["index"]),
                        {
                            "image_base64": payload["image_base64"],
                            "seed": payload["seed"],
                            "width": payload["width"],
                            "height": payload["height"],
                        },
                    )
                )
        indexed_items.sort(key=lambda pair: pair[0])
        return [item for _, item in indexed_items], time.time() - start_time

    async def _generate_stream(
        self,
        task: str,
        req: Union[Txt2ImgRequest, Img2ImgRequest, InpaintRequest],
    ) -> AsyncGenerator[str, None]:
        kwargs, use_controlnet = self._build_request_kwargs(task, req)
        start_time = time.time()
        completed_count = 0
        had_error = False
        async for msg_type, payload in self._iter_generated_images(
            task,
            kwargs,
            req.seed,
            req.num,
            req.scheduler,
            use_controlnet,
            req.controlnet_conditioning_scale,
        ):
            if msg_type == "image":
                completed_count += 1
                yield f"event: image\ndata: {json.dumps(payload)}\n\n"
            elif msg_type == "error":
                had_error = True
                yield f"event: error\ndata: {json.dumps({'detail': payload})}\n\n"
                break

        if not had_error and completed_count > 0:
            done_payload = {
                "status": "success",
                "total_images": completed_count,
                "elapsed_seconds": round(time.time() - start_time, 3),
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    def _build_request_kwargs(
        self,
        task: str,
        req: Union[Txt2ImgRequest, Img2ImgRequest, InpaintRequest],
    ) -> Tuple[InferenceKwargs, bool]:
        kwargs: InferenceKwargs = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
            "clip_skip": req.clip_skip,
            "width": req.width,
            "height": req.height,
        }
        control_field = "image"
        if isinstance(req, (Img2ImgRequest, InpaintRequest)):
            kwargs["image"] = decode_image_data(req.image).convert("RGB")
            kwargs["strength"] = req.strength
            control_field = "control_image"
        if isinstance(req, InpaintRequest):
            mask_raw = decode_image_data(req.mask_image)
            kwargs["mask_image"] = (
                convert_mask_image_to_rgb(mask_raw)
                if mask_raw.mode == "RGBA"
                else mask_raw.convert("RGB")
            )

        use_controlnet = False
        if req.control_image:
            canny = convert_image_to_canny(
                decode_image_data(req.control_image).convert("RGB"),
                req.low_threshold,
                req.high_threshold,
            )
            kwargs["control_image" if self._is_flux() else control_field] = canny  # type: ignore
            use_controlnet = True

        if req.ip_adapter_image and not self._is_flux():
            kwargs["ip_adapter_image"] = decode_image_data(req.ip_adapter_image).convert("RGB")

        return kwargs, use_controlnet

    async def generate_txt2img(self, req: Txt2ImgRequest) -> Tuple[List[GeneratedImageItem], float]:
        return await self._generate("txt2img", req)

    async def generate_img2img(self, req: Img2ImgRequest) -> Tuple[List[GeneratedImageItem], float]:
        return await self._generate("img2img", req)

    async def generate_inpaint(self, req: InpaintRequest) -> Tuple[List[GeneratedImageItem], float]:
        return await self._generate("inpaint", req)

    async def generate_txt2img_stream(self, req: Txt2ImgRequest) -> AsyncGenerator[str, None]:
        async for chunk in self._generate_stream("txt2img", req):
            yield chunk

    async def generate_img2img_stream(self, req: Img2ImgRequest) -> AsyncGenerator[str, None]:
        async for chunk in self._generate_stream("img2img", req):
            yield chunk

    async def generate_inpaint_stream(self, req: InpaintRequest) -> AsyncGenerator[str, None]:
        async for chunk in self._generate_stream("inpaint", req):
            yield chunk


manager = PipelineManager()
