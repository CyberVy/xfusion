from .enhancement_utils import PipelineEnhancerBase
from ..components.flux_components import load_flux_pipeline
from ..ui.flux_ui import load_flux_ui
from ..utils import image_normalize
from PIL import Image
import torch
import threading
from random import randint


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
        caption = f"Prompt:\n{prompt[:768]}\n\nStep: {kwargs.get('num_inference_steps')}, CFG: {kwargs.get('guidance_scale')}, CLIP Skip: {kwargs.get('clip_skip')}\nSampler: {pipeline.scheduler.config._class_name}\nSeed: {item}\n\nModel:{pipeline.model_name}"
        threading.Thread(target=lambda: pipeline.send_PIL_photo(image,file_name=f"{pipeline.__class__.__name__}.PNG",file_type="PNG",caption=caption)).start()
    return images


class FluxPipelineEnhancer(PipelineEnhancerBase):
    overrides = []

    def __init__(self,__oins__,init_sub_pipelines=True):
        PipelineEnhancerBase.__init__(self,__oins__,init_sub_pipelines=init_sub_pipelines)

    def __call__(self, *args, **kwargs):
        image = kwargs.get("image")
        if image and isinstance(image, Image.Image):
            kwargs.update(image=image_normalize(image, 1024 * 1536))

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            kwargs.update(mask_image=image_normalize(image, 1024 * 1536))
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
        server = load_flux_ui(self,_globals)
        server.launch(quiet=True,**kwargs)
        return server
