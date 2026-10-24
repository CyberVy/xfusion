import os
import io
import time
import base64
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image

from ..const import GPU_COUNT, GPU_NAME
try:
    from ..enhancement import load_enhancer
except (ImportError, AttributeError):
    load_enhancer = None
from ..utils import free_memory_to_system, convert_mask_image_to_rgb, convert_image_to_canny


def pil_to_base64_data_uri(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def decode_image_data(image_str: str) -> Image.Image:
    """Decode base64 data URI, raw base64 string, or download from URL."""
    if image_str.startswith("http://") or image_str.startswith("https://"):
        import requests
        response = requests.get(image_str, timeout=30)
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    
    if "base64," in image_str:
        image_str = image_str.split("base64,")[1]
    
    image_bytes = base64.b64decode(image_str)
    return Image.open(io.BytesIO(image_bytes))


class PipelineManager:
    def __init__(self):
        self.pipelines: list = []
        self.current_model: Optional[str] = None
        self.model_version: Optional[str] = None
        self.lock = asyncio.Lock()
        self.download_kwargs = {"directory": os.environ.get("DOWNLOAD_PATH", "/xfusion_models")}
        self.telegram_kwargs = {"token": os.environ.get("TG_TOKEN", ""), "chat_id": os.environ.get("TG_ID", "")}
        self.ip_adapter: Optional[Dict[str, Any]] = None
        self.loras_enabled = True
        self.controlnet_model: Optional[str] = None
        self.progress: Dict[str, Any] = {
            "active": False,
            "step": 0,
            "total_steps": 0,
            "image_index": 0,
            "image_count": 0,
        }

    def initialize_from_env(self):
        """Optionally initialize pipelines from environment variables."""
        if load_enhancer is None:
            print("Note: PyTorch CUDA pipeline is deferred or running in API mode.")
            return

        model_version = os.environ.get("MODEL_VERSION", "")
        model_url = os.environ.get("MODEL_URL", None)
        num_gpus = max(GPU_COUNT, 1)

        print(f"Initializing {num_gpus} pipeline(s) for model_version='{model_version}'...")
        try:
            self.pipelines = [
                load_enhancer(
                    model_url,
                    model_version=model_version,
                    download_kwargs=self.download_kwargs,
                    telegram_kwargs=self.telegram_kwargs
                )
                for _ in range(num_gpus)
            ]
            self.model_version = model_version
            self.current_model = model_url
            print("Pipelines successfully initialized.")
        except Exception as e:
            print(f"Deferred pipeline initialization: {e}")

    async def load_model(self, model: str, model_version: str) -> Dict[str, Any]:
        async with self.lock:
            num_gpus = max(GPU_COUNT, 1)
            for p in self.pipelines:
                p.clear()

            self.pipelines = [
                load_enhancer(
                    model,
                    model_version=model_version,
                    download_kwargs=self.download_kwargs,
                    telegram_kwargs=self.telegram_kwargs
                )
                for _ in range(num_gpus)
            ]

            # Distribute across GPUs if multiple exist
            if GPU_COUNT > 1:
                for i, p in enumerate(self.pipelines):
                    p.to(f"cuda:{i}")

            self.current_model = model
            self.model_version = model_version
            self.ip_adapter = None
            self.loras_enabled = True
            self.controlnet_model = None
            free_memory_to_system()
            return {"status": "success", "model": model, "model_version": model_version}

    async def set_lora(self, url: str, lora_name: str, strength: float = 0.4) -> Dict[str, Any]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model is currently loaded.")
            for p in self.pipelines:
                p.set_lora(url, lora_name, strength)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def delete_lora(self, lora_name: str) -> Dict[str, Any]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model is currently loaded.")
            for p in self.pipelines:
                p.delete_adapters(lora_name)
            free_memory_to_system()
            return {"status": "success", "deleted_lora": lora_name}

    def get_loras(self) -> Dict[str, float]:
        if not self.pipelines:
            return {}
        return getattr(self.pipelines[0], "lora_dict", {})

    async def set_lora_strength(self, lora_name: str, strength: float) -> Dict[str, Any]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
                pipeline.set_lora_strength(lora_name, strength)
            return {"status": "success", "lora_name": lora_name, "strength": strength}

    async def set_loras_enabled(self, enabled: bool) -> Dict[str, Any]:
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

    async def set_ip_adapter(self, uri: str, subfolder: str, weight_name: str, image_encoder_folder: str,
                             strength: float) -> Dict[str, Any]:
        async with self.lock:
            self._require_pipelines()
            if self._is_flux():
                raise RuntimeError("IP-Adapter is not supported for FLUX.")
            for pipeline in self.pipelines:
                pipeline.set_ip_adapter(uri, subfolder, weight_name, image_encoder_folder)
                pipeline.set_ip_adapter_strength(strength)
            self.ip_adapter = {
                "uri": uri,
                "subfolder": subfolder,
                "weight_name": weight_name,
                "image_encoder_folder": image_encoder_folder,
                "strength": strength,
            }
            return {"status": "success", "ip_adapter": self.ip_adapter}

    async def set_ip_adapter_strength(self, strength: float) -> Dict[str, Any]:
        async with self.lock:
            self._require_pipelines()
            if not self.ip_adapter:
                raise RuntimeError("No IP-Adapter is loaded.")
            for pipeline in self.pipelines:
                pipeline.set_ip_adapter_strength(strength)
            self.ip_adapter = {**self.ip_adapter, "strength": strength}
            return {"status": "success", "strength": strength}

    async def delete_ip_adapter(self) -> Dict[str, Any]:
        async with self.lock:
            self._require_pipelines()
            for pipeline in self.pipelines:
                pipeline.delete_ip_adapter()
            self.ip_adapter = None
            free_memory_to_system()
            return {"status": "success"}

    async def load_controlnet(self, model: Optional[str]) -> Dict[str, Any]:
        async with self.lock:
            self._require_pipelines()
            self.controlnet_model = model or None
            for pipeline in self.pipelines:
                pipeline.load_controlnet(self.controlnet_model)
            return {"status": "success", "controlnet_model": self.controlnet_model, "loaded": True}

    async def offload_controlnet(self) -> Dict[str, Any]:
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

    def _pipeline_has_controlnet(self, pipeline) -> bool:
        if getattr(pipeline, "_controlnet", None) is not None:
            return True
        try:
            adapters = pipeline.get_list_adapters() or {}
        except Exception:
            return False
        return "controlnet" in (adapters.get("transformer") or [])

    async def set_scheduler(self, scheduler: str, pipeline_type: str = "txt2img") -> Dict[str, Any]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model is currently loaded.")
            for p in self.pipelines:
                if pipeline_type == "txt2img":
                    p.text_to_image_pipeline.set_scheduler(scheduler)
                elif pipeline_type == "img2img":
                    p.image_to_image_pipeline.set_scheduler(scheduler)
                elif pipeline_type == "inpaint":
                    p.inpainting_pipeline.set_scheduler(scheduler)
            return {"status": "success", "scheduler": scheduler, "pipeline_type": pipeline_type}

    def get_schedulers(self) -> List[str]:
        if not self.pipelines:
            return []
        return list(getattr(self.pipelines[0], "scheduler_map", {}))

    async def _generate(self, pipeline_attr: str, kwargs: Dict[str, Any], seed: Optional[int], num: Optional[int],
                        scheduler: Optional[str], use_controlnet: bool = False,
                        controlnet_scale: Optional[float] = None) -> Tuple[List[Dict[str, Any]], float]:
        async with self.lock:
            self._require_pipelines()
            controlnet_attrs = {
                "text_to_image_pipeline": "text_to_image_controlnet_pipeline",
                "image_to_image_pipeline": "image_to_image_controlnet_pipeline",
                "inpainting_pipeline": "inpainting_controlnet_pipeline",
            }
            if use_controlnet:
                for pipeline in self.pipelines:
                    if not self._pipeline_has_controlnet(pipeline):
                        pipeline.load_controlnet(self.controlnet_model)
                pipeline_attr = controlnet_attrs[pipeline_attr]
                if self._is_flux():
                    for pipeline in self.pipelines:
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
                    getattr(pipeline, pipeline_attr).set_scheduler(scheduler)

            # Seeds stay within 31 bits so JavaScript clients can round-trip them exactly.
            num = max(1, min(int(num or 1), 10))
            seeds = [seed + i for i in range(num)] if seed else [random.randint(1, 2 ** 31 - 1) for _ in range(num)]

            total_steps = int(kwargs.get("num_inference_steps") or 0)
            self.progress = {
                "active": True,
                "step": 0,
                "total_steps": total_steps,
                "image_index": 0,
                "image_count": len(seeds),
            }

            def callback_on_step_end(_pipe, step, _timestep, callback_kwargs):
                self.progress = {**self.progress, "step": int(step) + 1}
                return callback_kwargs

            call_kwargs = {**kwargs, "callback_on_step_end": callback_on_step_end}
            start_time = time.time()
            loop = asyncio.get_running_loop()

            def run_seed(index: int, item_seed: int, pipeline):
                self.progress = {**self.progress, "image_index": index, "step": 0}
                images = getattr(pipeline, pipeline_attr).generate_image_and_send_to_telegram(
                    seed=item_seed, num=1, **call_kwargs
                )
                return index, item_seed, images

            try:
                if len(self.pipelines) == 1:
                    ordered = []
                    for index, item_seed in enumerate(seeds):
                        ordered.append(await loop.run_in_executor(
                            None, run_seed, index, item_seed, self.pipelines[0]
                        ))
                else:
                    def run_batch():
                        with ThreadPoolExecutor(max_workers=len(self.pipelines)) as pool:
                            futures = [
                                pool.submit(run_seed, index, item_seed, self.pipelines[index % len(self.pipelines)])
                                for index, item_seed in enumerate(seeds)
                            ]
                            return [future.result() for future in futures]

                    ordered = await loop.run_in_executor(None, run_batch)
                    ordered.sort(key=lambda item: item[0])
            finally:
                self.progress = {**self.progress, "active": False}

            result = []
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

    def _prepare_control_image(self, req, kwargs: Dict[str, Any], control_field: str) -> bool:
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
            kwargs[control_field] = canny
        return True

    def _prepare_ip_adapter_image(self, req, kwargs: Dict[str, Any]) -> None:
        if req.ip_adapter_image and not self._is_flux():
            kwargs["ip_adapter_image"] = decode_image_data(req.ip_adapter_image).convert("RGB")

    async def generate_txt2img(self, req) -> Tuple[List[Dict[str, Any]], float]:
        kwargs = {
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
            "text_to_image_pipeline", kwargs, req.seed, req.num, req.scheduler,
            use_controlnet, req.controlnet_conditioning_scale,
        )

    async def generate_img2img(self, req) -> Tuple[List[Dict[str, Any]], float]:
        kwargs = {
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
            "image_to_image_pipeline", kwargs, req.seed, req.num, req.scheduler,
            use_controlnet, req.controlnet_conditioning_scale,
        )

    async def generate_inpaint(self, req) -> Tuple[List[Dict[str, Any]], float]:
        mask_raw = decode_image_data(req.mask_image)
        if mask_raw.mode == "RGBA":
            mask_image = convert_mask_image_to_rgb(mask_raw)
        else:
            mask_image = mask_raw.convert("RGB")

        kwargs = {
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
            "inpainting_pipeline", kwargs, req.seed, req.num, req.scheduler,
            use_controlnet, req.controlnet_conditioning_scale,
        )


manager = PipelineManager()
