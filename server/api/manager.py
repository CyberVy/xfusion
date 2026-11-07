import time
import random
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Union, TypedDict, AsyncGenerator, Any
from PIL import Image

from .schemas import GenerationRequest
from ..infra.config import GPU_COUNT, DOWNLOAD_PATH, DELETE_FILE_AFTER_LOAD
from ..infra.memory import free_memory_to_system
from ..infra.image import (
    pil_to_base64_data_uri,
    decode_image_data,
    convert_image_to_canny,
    convert_mask_image_to_rgb,
)
from ..engine import load_model
from ..engine.contracts import GenerationRequest as ModelRequest
from ..engine.contracts import GenerationInterrupted, IpAdapterSource, ModelLoadSettings
from ..engine.runtime import LoadedModel


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


class PipelineManager:
    def __init__(self):
        self.models: List[LoadedModel] = []
        self.current_model: Optional[str] = None
        self.settings = ModelLoadSettings(source="")
        self.lock = asyncio.Lock()
        self.download_kwargs: Dict[str, str] = {"directory": DOWNLOAD_PATH}
        self.cancel_requested: bool = False
        self.progress: ProgressDict = {
            "active": False,
            "step": 0,
            "total_steps": 0,
            "image_index": 0,
            "image_count": 0,
        }

    @property
    def model_version(self) -> Optional[str]:
        if not self.models:
            return None
        return self.models[0].features.model_version

    @property
    def loras_enabled(self) -> bool:
        if not self.models:
            return True
        return self.models[0].loras_enabled

    def cancel(self) -> bool:
        if not self.progress.get("active"):
            return False
        self.cancel_requested = True
        for model in self.models:
            model.interrupt()
        return True

    def initialize_from_env(self) -> None:
        source = os.environ.get("MODEL_URL") or ""
        if not source:
            print("Note: No MODEL_URL provided, skipping auto-initialization.")
            return
        settings = ModelLoadSettings(
            source=source,
            model_version=os.environ.get("MODEL_VERSION") or None,
            backbone=os.environ.get("BACKBONE", "fp16"),
            text_encoder=os.environ.get("TEXT_ENCODER", "default"),
            memory=os.environ.get("MEMORY", "gpu"),
            dtype=os.environ.get("DTYPE", "auto"),
            delete_file_after_load=DELETE_FILE_AFTER_LOAD,
        )
        print(f"Initializing {max(GPU_COUNT, 1)} model(s) for model_version='{settings.model_version or ''}'...")
        try:
            self._load_sync(settings)
            print("Models successfully initialized.")
        except Exception as exc:
            print(f"Deferred model initialization: {exc}")

    async def load_model(self, settings: ModelLoadSettings) -> Dict[str, str]:
        async with self.lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_sync, settings)
            return {
                "status": "success",
                "model": settings.source,
                "model_version": self.model_version or "",
                "backbone": settings.backbone,
                "text_encoder": settings.text_encoder,
                "memory": settings.memory,
                "dtype": settings.dtype,
            }

    def _load_sync(self, settings: ModelLoadSettings) -> None:
        for model in self.models:
            model.clear()
        self.models = []
        free_memory_to_system()
        loaded: List[LoadedModel] = []
        try:
            for gpu_id in range(max(GPU_COUNT, 1)):
                loaded.append(load_model(settings, gpu_id=gpu_id, download_kwargs=self.download_kwargs))
                free_memory_to_system()
            self.models = loaded
            self.current_model = settings.source
            self.settings = settings
        except Exception:
            for model in loaded:
                model.clear()
            raise
        free_memory_to_system()

    def _require_models(self) -> None:
        if not self.models:
            raise RuntimeError("No model is currently loaded.")

    def _primary(self) -> LoadedModel:
        self._require_models()
        return self.models[0]

    async def set_lora(self, url: str, lora_name: str, strength: float = 0.4) -> Dict[str, Union[str, float]]:
        async with self.lock:
            self._require_models()
            loop = asyncio.get_running_loop()

            def _sync() -> None:
                for model in self.models:
                    model.set_lora(url, lora_name, strength)

            await loop.run_in_executor(None, _sync)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def delete_lora(self, lora_name: str) -> Dict[str, str]:
        async with self.lock:
            self._require_models()
            for model in self.models:
                model.delete_lora(lora_name)
            free_memory_to_system()
            return {"status": "success", "deleted_lora": lora_name}

    def get_loras(self) -> Dict[str, float]:
        if not self.models:
            return {}
        return dict(self.models[0].loras)

    async def set_lora_strength(self, lora_name: str, strength: float) -> Dict[str, Union[str, float]]:
        async with self.lock:
            self._require_models()
            for model in self.models:
                model.set_lora_strength(lora_name, strength)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def set_loras_enabled(self, enabled: bool) -> Dict[str, Union[str, bool]]:
        async with self.lock:
            self._require_models()
            for model in self.models:
                model.set_loras_enabled(enabled)
            return {"status": "success", "enabled": enabled}

    def ip_adapter_info(self) -> Optional[Dict[str, Union[str, float]]]:
        if not self.models or self.models[0].ip_adapter is None:
            return None
        adapter = self.models[0].ip_adapter
        return {
            "source": adapter.source,
            "folder": adapter.folder,
            "weight": adapter.weight,
            "encoder_folder": adapter.encoder_folder,
            "strength": adapter.strength,
        }

    async def set_ip_adapter(
        self,
        source: str,
        folder: str,
        weight: str,
        encoder_folder: str,
        strength: float,
    ) -> Dict[str, Any]:
        async with self.lock:
            self._require_models()
            loop = asyncio.get_running_loop()
            spec = IpAdapterSource(
                source=source,
                folder=folder,
                weight=weight,
                encoder_folder=encoder_folder,
            )

            def _sync() -> None:
                for model in self.models:
                    model.load_ip_adapter(spec, strength)

            await loop.run_in_executor(None, _sync)
            return {"status": "success", "ip_adapter": self.ip_adapter_info()}

    async def set_ip_adapter_strength(self, strength: float) -> Dict[str, Union[str, float]]:
        async with self.lock:
            self._require_models()
            for model in self.models:
                model.set_ip_adapter_strength(strength)
            return {"status": "success", "strength": strength}

    async def delete_ip_adapter(self) -> Dict[str, str]:
        async with self.lock:
            self._require_models()
            for model in self.models:
                model.clear_ip_adapter()
            free_memory_to_system()
            return {"status": "success"}

    async def load_controlnet(self, source: Optional[str]) -> Dict[str, Union[str, bool, None]]:
        async with self.lock:
            self._require_models()
            loop = asyncio.get_running_loop()

            def _sync() -> None:
                for model in self.models:
                    model.load_controlnet(source)

            await loop.run_in_executor(None, _sync)
            return {"status": "success", "source": source, "loaded": self.controlnet_is_loaded()}

    async def clear_controlnet(self) -> Dict[str, Union[str, bool]]:
        async with self.lock:
            self._require_models()
            for model in self.models:
                model.clear_controlnet()
            return {"status": "success", "loaded": self.controlnet_is_loaded()}

    def controlnet_is_loaded(self) -> bool:
        if not self.models:
            return False
        return self.models[0].controlnet_loaded

    def preview_canny(self, image_str: str, canny_low: int, canny_high: int) -> str:
        image = decode_image_data(image_str).convert("RGB")
        preview = convert_image_to_canny(image, canny_low, canny_high)
        return pil_to_base64_data_uri(preview)

    async def set_scheduler(self, scheduler: str) -> Dict[str, str]:
        async with self.lock:
            self._require_models()
            for model in self.models:
                model.set_scheduler(scheduler)
            return {"status": "success", "scheduler": scheduler}

    def get_schedulers(self) -> List[str]:
        if not self.models:
            return []
        return list(self.models[0].features.schedulers)

    def _engine_request(self, task: str, req: GenerationRequest, seed: Optional[int]) -> ModelRequest:
        image = decode_image_data(req.image).convert("RGB") if req.image else None
        mask = None
        if req.mask:
            mask_raw = decode_image_data(req.mask)
            mask = convert_mask_image_to_rgb(mask_raw) if mask_raw.mode == "RGBA" else mask_raw.convert("RGB")
        control_image = None
        if req.control_image:
            control_image = convert_image_to_canny(
                decode_image_data(req.control_image).convert("RGB"),
                req.canny_low,
                req.canny_high,
            )
        reference_image = decode_image_data(req.reference_image).convert("RGB") if req.reference_image else None
        return ModelRequest(
            task=task,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance=req.guidance,
            seed=seed,
            clip_skip=req.clip_skip,
            scheduler=req.scheduler,
            image=image,
            mask=mask,
            strength=req.strength,
            control_image=control_image,
            control_scale=req.control_scale,
            canny_low=req.canny_low,
            canny_high=req.canny_high,
            reference_image=reference_image,
        )

    async def _iter_generated_images(
        self,
        task: str,
        req: GenerationRequest,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        async with self.lock:
            self._require_models()
            if req.control_image:
                for model in self.models:
                    if not model.controlnet_loaded:
                        model.load_controlnet(None)

            image_count = max(1, min(int(req.image_count or 1), 10))
            seeds = (
                [req.seed + i for i in range(image_count)]
                if req.seed is not None
                else [random.randint(1, 2**31 - 1) for _ in range(image_count)]
            )

            self.cancel_requested = False
            self.progress = {
                "active": True,
                "step": 0,
                "total_steps": int(req.steps or 0),
                "image_index": 0,
                "image_count": len(seeds),
            }

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
            assignments: List[List[Tuple[int, int]]] = [[] for _ in self.models]
            for index, item_seed in enumerate(seeds):
                assignments[index % len(self.models)].append((index, item_seed))

            def run_worker(model_index: int, task_list: List[Tuple[int, int]]) -> None:
                model = self.models[model_index]
                for index, item_seed in task_list:
                    if self.cancel_requested:
                        break
                    try:
                        started = time.time()
                        self.progress = {**self.progress, "image_index": index, "step": 0}

                        def on_step(step: int, total: int) -> None:
                            if self.cancel_requested:
                                model.interrupt()
                                raise GenerationInterrupted("Generation cancelled by user")
                            self.progress = {**self.progress, "step": step, "total_steps": total}

                        result = model.generate(self._engine_request(task, req, item_seed), on_step=on_step)
                        item = {
                            "image_base64": pil_to_base64_data_uri(result.image),
                            "seed": result.seed,
                            "width": result.image.width,
                            "height": result.image.height,
                            "index": index,
                            "elapsed_seconds": round(time.time() - started, 3),
                        }
                        loop.call_soon_threadsafe(queue.put_nowait, ("image", item))
                    except GenerationInterrupted:
                        loop.call_soon_threadsafe(queue.put_nowait, ("cancelled", None))
                        break
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                        break

            active_workers = [index for index in range(len(self.models)) if assignments[index]]
            pool = ThreadPoolExecutor(max_workers=max(1, len(active_workers)))
            for model_index in active_workers:
                pool.submit(run_worker, model_index, assignments[model_index])

            completed_count = 0
            try:
                while completed_count < len(seeds) and not self.cancel_requested:
                    try:
                        msg_type, payload = await asyncio.wait_for(queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if msg_type == "cancelled":
                        break
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

    async def generate(self, task: str, req: GenerationRequest) -> Tuple[List[GeneratedImageItem], float]:
        started = time.time()
        indexed: List[Tuple[int, GeneratedImageItem]] = []
        async for msg_type, payload in self._iter_generated_images(task, req):
            if msg_type == "error":
                raise RuntimeError(str(payload))
            if msg_type == "image":
                indexed.append(
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
        indexed.sort(key=lambda pair: pair[0])
        return [item for _, item in indexed], time.time() - started

    async def generate_stream(self, task: str, req: GenerationRequest) -> AsyncGenerator[str, None]:
        started = time.time()
        completed_count = 0
        had_error = False
        async for msg_type, payload in self._iter_generated_images(task, req):
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
                "elapsed_seconds": round(time.time() - started, 3),
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"


manager = PipelineManager()
