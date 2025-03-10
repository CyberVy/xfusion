from .enhancement_utils import PipelineEnhancerBase
from ..components.flux_components import load_flux_pipeline
from ..ui.flux_ui import load_flux_ui
from ..utils import normalize_image,dict_to_str,free_memory_to_system
from diffusers import FluxPipeline,FluxImg2ImgPipeline,FluxInpaintPipeline
from PIL import Image
import torch
import threading
from random import randint


pipeline_map = {"flux": (FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline)}


def generate_image_and_send_to_telegram(pipeline,prompt,num,seed=None,use_enhancer=True,**kwargs):
    kwargs.update(prompt=prompt)
    seeds = []
    images = []
    if seed:
        seeds.append(seed)
    else:
        for i in range(num):
            seeds.append(randint(-2 ** 63, 2 ** 64 - 1))
    for item in seeds:
        kwargs.update(generator=torch.Generator(pipeline.device).manual_seed(item))
        image = pipeline(**kwargs).images[0] if use_enhancer else pipeline.__oins__(**kwargs).images[0]
        images.append(image)

        kwargs_for_telegram = kwargs.copy()
        kwargs_for_telegram.update(prompt=f"\n{kwargs['prompt'][:768]}")
        kwargs_for_telegram.pop("generator",None)
        caption = dict_to_str(kwargs_for_telegram)
        caption += f"Sampler: {pipeline.scheduler.config._class_name}\nLoRa: {pipeline.lora_dict}\nSize: {image.size}\nSeed: {item}\n\nModel:{pipeline.model_name}"
        threading.Thread(target=lambda: pipeline.send_pil_photo(image, file_name=f"{pipeline.__class__.__name__}.PNG", file_type="PNG", caption=caption)).start()
    return images
class FluxPipelineEnhancer(PipelineEnhancerBase):
    pipeline_map = pipeline_map
    overrides = []

    def __init__(self,__oins__,init_sub_pipelines=True):
        PipelineEnhancerBase.__init__(self,__oins__,init_sub_pipelines=init_sub_pipelines)

    def check_inference_kwargs(self,kwargs):

        width = kwargs.get("width") or 1024
        height = kwargs.get("height") or 1024
        kwargs.update(width=width)
        kwargs.update(height=height)

        image = kwargs.get("image")
        if image and isinstance(image, Image.Image):
            image = normalize_image(image, width * height,scale_divisor=16)
            kwargs.update(width=image.width)
            kwargs.update(height=image.height)
            kwargs.update(image=image)

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            mask_image = normalize_image(mask_image, width * height,scale_divisor=16)
            kwargs.update(mask_image=mask_image)

        return kwargs

    def __call__(self, *args, **kwargs):
        kwargs = self.check_inference_kwargs(kwargs)
        return self.__oins__.__call__(*args,**kwargs)

    def generate_image_and_send_to_telegram(self,
                                            prompt,
                                            guidance_scale,num_inference_steps,
                                            width,height,
                                            num=1,seed=None,use_enhancer=True,**kwargs):
        return generate_image_and_send_to_telegram(self,prompt,
                                                   guidance_scale=guidance_scale,num_inference_steps=num_inference_steps,
                                                   width=width, height=height,
                                                   num=num,seed=seed,
                                                   use_enhancer=use_enhancer,**kwargs)

    def load_controlnet(self,controlnet_model=None):
        if controlnet_model is None:
            controlnet_model = "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors?download=true"
        self.set_lora(controlnet_model,"controlnet",1)

    def offload_controlnet(self):
        # self.delete_adapters("controlnet")
        print("offload controlnet is not implemented now.")
        free_memory_to_system()

    @classmethod
    def from_url(cls,url=None,init_sub_pipelines=True,**kwargs):
        return cls(load_flux_pipeline(url,**kwargs),init_sub_pipelines=init_sub_pipelines)

    def load_ui(self,_globals=None,**kwargs):
        server = load_flux_ui(self,_globals,**kwargs)
        return server
