import os
import io
import time
import base64
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image

from ..const import GPU_COUNT, GPU_NAME
try:
    from ..enhancement import load_enhancer
except (ImportError, AttributeError):
    load_enhancer = None
from ..utils import free_memory_to_system, convert_mask_image_to_rgb

AVAILABLE_SCHEDULERS = [
    "DPM++ 2M", "DPM++ 2M KARRAS", "DPM++ 2M SDE", "DPM++ 2M SDE KARRAS",
    "DPM++ 2S A", "DPM++ 2S A KARRAS", "DPM++ SDE", "DPM++ SDE KARRAS",
    "DPM2", "DPM2 KARRAS", "DPM2 A", "DPM2 A KARRAS",
    "EULER", "EULER A",
    "HEUN", "LMS", "LMS KARRAS", "DEIS", "UNIPC"
]


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

    async def generate_txt2img(self, req) -> Tuple[List[Dict[str, Any]], float]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model loaded. Please load a model first.")

            pipeline = self.pipelines[0]
            start_time = time.time()

            kwargs = {
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "guidance_scale": req.guidance_scale,
                "num_inference_steps": req.num_inference_steps,
                "clip_skip": req.clip_skip,
                "width": req.width,
                "height": req.height,
                "seed": req.seed,
                "num": req.num or 1,
            }

            if req.scheduler:
                pipeline.text_to_image_pipeline.set_scheduler(req.scheduler)

            # Run inference in executor thread to prevent blocking the event loop
            loop = asyncio.get_running_loop()
            images = await loop.run_in_executor(
                None,
                lambda: pipeline.text_to_image_pipeline.generate_image_and_send_to_telegram(**kwargs)
            )

            elapsed = time.time() - start_time
            free_memory_to_system()

            result = []
            for i, img in enumerate(images):
                result.append({
                    "image_base64": pil_to_base64_data_uri(img),
                    "seed": (req.seed or 0) + i
                })
            return result, elapsed

    async def generate_img2img(self, req) -> Tuple[List[Dict[str, Any]], float]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model loaded. Please load a model first.")

            pipeline = self.pipelines[0]
            start_time = time.time()

            input_image = decode_image_data(req.image).convert("RGB")

            kwargs = {
                "image": input_image,
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "strength": req.strength,
                "guidance_scale": req.guidance_scale,
                "num_inference_steps": req.num_inference_steps,
                "clip_skip": req.clip_skip,
                "seed": req.seed,
                "num": req.num or 1,
            }
            if req.width:
                kwargs["width"] = req.width
            if req.height:
                kwargs["height"] = req.height
            if req.scheduler:
                pipeline.image_to_image_pipeline.set_scheduler(req.scheduler)

            loop = asyncio.get_running_loop()
            images = await loop.run_in_executor(
                None,
                lambda: pipeline.image_to_image_pipeline.generate_image_and_send_to_telegram(**kwargs)
            )

            elapsed = time.time() - start_time
            free_memory_to_system()

            result = []
            for i, img in enumerate(images):
                result.append({
                    "image_base64": pil_to_base64_data_uri(img),
                    "seed": (req.seed or 0) + i
                })
            return result, elapsed

    async def generate_inpaint(self, req) -> Tuple[List[Dict[str, Any]], float]:
        async with self.lock:
            if not self.pipelines:
                raise RuntimeError("No model loaded. Please load a model first.")

            pipeline = self.pipelines[0]
            start_time = time.time()

            input_image = decode_image_data(req.image).convert("RGB")
            mask_raw = decode_image_data(req.mask_image)
            if mask_raw.mode == "RGBA":
                mask_image = convert_mask_image_to_rgb(mask_raw)
            else:
                mask_image = mask_raw.convert("RGB")

            kwargs = {
                "image": input_image,
                "mask_image": mask_image,
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "strength": req.strength,
                "guidance_scale": req.guidance_scale,
                "num_inference_steps": req.num_inference_steps,
                "clip_skip": req.clip_skip,
                "seed": req.seed,
                "num": req.num or 1,
            }
            if req.width:
                kwargs["width"] = req.width
            if req.height:
                kwargs["height"] = req.height
            if req.scheduler:
                pipeline.inpainting_pipeline.set_scheduler(req.scheduler)

            loop = asyncio.get_running_loop()
            images = await loop.run_in_executor(
                None,
                lambda: pipeline.inpainting_pipeline.generate_image_and_send_to_telegram(**kwargs)
            )

            elapsed = time.time() - start_time
            free_memory_to_system()

            result = []
            for i, img in enumerate(images):
                result.append({
                    "image_base64": pil_to_base64_data_uri(img),
                    "seed": (req.seed or 0) + i
                })
            return result, elapsed


manager = PipelineManager()
