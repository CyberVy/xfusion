from .enhancement_utils import PipelineEnhancerBase
from ..components.flux_components import load_flux_pipeline
from ..ui.flux_ui import load_flux_ui
from ..utils import normalize_image,dict_to_str
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

        caption += f"Sampler: {pipeline.scheduler.config._class_name}\nLoRa: {pipeline.lora_dict}\nSeed: {item}\n\nModel:{pipeline.model_name}"
        threading.Thread(target=lambda: pipeline.send_pil_photo(image, file_name=f"{pipeline.__class__.__name__}.PNG", file_type="PNG", caption=caption)).start()
    torch.cuda.empty_cache()
    return images
class FluxPipelineEnhancer(PipelineEnhancerBase):
    pipeline_map = pipeline_map
    overrides = []

    def __init__(self,__oins__,init_sub_pipelines=True):
        PipelineEnhancerBase.__init__(self,__oins__,init_sub_pipelines=init_sub_pipelines)

    def check_inference_kwargs(self,kwargs):

        image = kwargs.get("image")
        if image and isinstance(image, Image.Image):
            kwargs.update(image=normalize_image(image, 1024 * 1024))

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            mask_image = normalize_image(mask_image, 1024 * 1024)
            kwargs.update(mask_image=mask_image)
            kwargs.update(width=mask_image.width)
            kwargs.update(height=mask_image.height)

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

    @classmethod
    def from_url(cls,url=None,init_sub_pipelines=True,**kwargs):
        return cls(load_flux_pipeline(url,**kwargs),init_sub_pipelines=init_sub_pipelines)

    def load_ui(self,_globals=None,**kwargs):
        server = load_flux_ui(self,_globals,**kwargs)
        return server
