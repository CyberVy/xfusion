# limited support to sd3 now

from .enhancement_utils import PipelineEnhancerBase
from ..components.component_utils import get_tokenizers_and_text_encoders_from_pipeline
from ..components import load_stable_diffusion_pipeline
from ..components import load_stable_diffusion_controlnet
from ..ui.stable_diffusion_ui import load_stable_diffusion_ui
from ..utils import normalize_image,dict_to_str,convert_image_to_canny
from compel import Compel,ReturnedEmbeddingsType
import torch
from PIL import Image
import threading
from random import randint
from diffusers import (StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline,
                       StableDiffusionControlNetPipeline,StableDiffusionControlNetImg2ImgPipeline)
from diffusers import (StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,StableDiffusionXLInpaintPipeline,
                       StableDiffusionXLControlNetPipeline,StableDiffusionXLControlNetImg2ImgPipeline)
from diffusers import StableDiffusion3Pipeline,StableDiffusion3Img2ImgPipeline,StableDiffusion3InpaintPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler
from diffusers.schedulers import KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler
from diffusers.schedulers import EulerDiscreteScheduler,EulerAncestralDiscreteScheduler
from diffusers.schedulers import HeunDiscreteScheduler
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.schedulers import DEISMultistepScheduler
from diffusers.schedulers import UniPCMultistepScheduler

# pipeline_type
# 0-> text_to_image, 1 -> image_to_image, 2 -> inpainting
pipeline_map = {
    "1.5":(StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline,StableDiffusionControlNetPipeline),
    "xl":(StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,StableDiffusionXLInpaintPipeline,StableDiffusionXLControlNetPipeline),
    "3":(StableDiffusion3Pipeline,StableDiffusion3Img2ImgPipeline,StableDiffusion3InpaintPipeline)}


# from https://huggingface.co/docs/diffusers/api/schedulers/overview
scheduler_map = {
            "DPM++ 2M": (DPMSolverMultistepScheduler, {}),
            "DPM++ 2M KARRAS": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
            "DPM++ 2M SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
            "DPM++ 2M SDE KARRAS": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
            "DPM++ 2S A": (DPMSolverSinglestepScheduler, {}),
            "DPM++ 2S A KARRAS": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
            "DPM++ SDE": (DPMSolverSinglestepScheduler, {}),
            "DPM++ SDE KARRAS": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
            "DPM2": (KDPM2DiscreteScheduler, {}),
            "DPM2 KARRAS": (KDPM2DiscreteScheduler, {"use_karras_sigmas": True}),
            "DPM2 A": (KDPM2AncestralDiscreteScheduler, {}),
            "DPM2 A KARRAS": (KDPM2AncestralDiscreteScheduler, {"use_karras_sigmas": True}),
            "EULER": (EulerDiscreteScheduler, {}),
            "EULER A": (EulerAncestralDiscreteScheduler, {}),
            "HEUN": (HeunDiscreteScheduler, {}),
            "LMS": (LMSDiscreteScheduler, {}),
            "LMS KARRAS": (LMSDiscreteScheduler, {"use_karras_sigmas": True}),
            "DEIS": (DEISMultistepScheduler, {}),
            "UNIPC": (UniPCMultistepScheduler, {}),
        }

def get_embeds_from_pipeline(pipeline,prompt,negative_prompt):
    """
    To overcome 77 tokens limit, and support high level syntax for prompts.
    limited support for sd1 sd2 sdxl

    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :return:
    """
    tokenizers,text_encoders = get_tokenizers_and_text_encoders_from_pipeline(pipeline)
    tokenizers = [tokenizer for tokenizer in tokenizers if tokenizer]
    text_encoders = [text_encoder for text_encoder in text_encoders if text_encoder]
    try:
        for cls in pipeline_map["1.5"]:
            if isinstance(pipeline,cls):
                compel = Compel(tokenizers,text_encoders,truncate_long_prompts=False)
                conditioning = compel(prompt)
                negative_conditioning = compel(negative_prompt)
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
                return {"prompt_embeds":conditioning,"negative_prompt_embeds":negative_conditioning}
        for cls in pipeline_map["xl"]:
            if isinstance(pipeline, cls):
                compel = Compel(tokenizers,text_encoders,
                                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                                requires_pooled=[False,True],truncate_long_prompts=False)
                conditioning,pooled = compel(prompt)
                negative_conditioning, negative_pooled = compel(negative_prompt)
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
                return {"prompt_embeds":conditioning,"negative_prompt_embeds":negative_conditioning,
                        "pooled_prompt_embeds":pooled,"negative_pooled_prompt_embeds":negative_pooled}
    except KeyError:
        pass
    # compel only supports sd1 sd2 sdxl now
    return {}
class SDCLIPEnhancerMixin:
    # __oins__ here is the pipeline instance to implement.
    __oins__ = None
    overrides = ["tokenizers","text_encoders","skip_clip_layer","get_embeds_from_pipeline","__call__"]

    def __init__(self):
        self.tokenizers, self.text_encoders = get_tokenizers_and_text_encoders_from_pipeline(self)

    def skip_clip_layer(self,n):
        text_encoders = [text_encoder for text_encoder in self.text_encoders if text_encoder]
        if isinstance(n,int) and  n >= 1:
            for text_encoder in text_encoders:
                if not hasattr(text_encoder.text_model.encoder,"backup_layers"):
                    text_encoder.text_model.encoder.backup_layers = text_encoder.text_model.encoder.layers
                text_encoder.text_model.encoder.layers = text_encoder.text_model.encoder.backup_layers[:-n]
        elif n is None or n == 0:
            for text_encoder in text_encoders:
                if hasattr(text_encoder.text_model.encoder,"backup_layers"):
                    text_encoder.text_model.encoder.layers = text_encoder.text_model.encoder.backup_layers
        else:
            raise ValueError("An integer >=0 is required.")

    def get_embeds_from_pipeline(self,prompt,negative_prompt,clip_skip=None):
        self.skip_clip_layer(clip_skip)
        r =  get_embeds_from_pipeline(self,prompt,negative_prompt)
        self.skip_clip_layer(0)
        return r

    def __call__(self,**kwargs):
        kwargs.update(self.get_embeds_from_pipeline(kwargs.get("prompt"),kwargs.get("negative_prompt"),kwargs.get("clip_skip")))
        if kwargs.get("prompt_embeds") is not None:
            kwargs.update(prompt=None,negative_prompt=None,clip_skip=None)
        return self.__oins__.__call__(**kwargs)

def generate_image_and_send_to_telegram(pipeline,prompt,negative_prompt,num,seed=None,use_enhancer=True,**kwargs):
    kwargs.update(prompt=prompt,negative_prompt=negative_prompt)
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
        kwargs_for_telegram.update(prompt=f"\n{kwargs['prompt'][:384]}")
        kwargs_for_telegram.update(negative_prompt=f"\n{kwargs['negative_prompt'][:384]}")
        kwargs_for_telegram.pop("generator",None)
        caption = dict_to_str(kwargs_for_telegram)
        caption += f"Sampler: {pipeline.scheduler.config._class_name}\nLoRa: {pipeline.lora_dict}\nSeed: {item}\n\nModel:{pipeline.model_name}"

        threading.Thread(target=lambda: pipeline.send_PIL_photo(image,file_name=f"{pipeline.__class__.__name__}.PNG",file_type="PNG",caption=caption)).start()
    return images
class SDPipelineEnhancer(SDCLIPEnhancerMixin,PipelineEnhancerBase):
    pipeline_map = pipeline_map
    scheduler_map = scheduler_map
    overrides = []

    def __init__(self,__oins__,init_sub_pipelines=True):
        PipelineEnhancerBase.__init__(self, __oins__,init_sub_pipelines=init_sub_pipelines)
        SDCLIPEnhancerMixin.__init__(self)

    def check_inference_kwargs(self,kwargs):

        if kwargs.get("negative_prompt") is None:
            kwargs.update(negative_prompt="")

        width = kwargs.get("width")
        height = kwargs.get("height")

        if self.model_version in ["xl", "3"]:
            if width is None:
                width = 1024
            if height is None:
                height = 1024
        else:
            if width is None:
                width = 512
            if height is None:
                height = 512

        kwargs.update(width=width)
        kwargs.update(height=height)

        image = kwargs.get("image")
        if image and isinstance(image, Image.Image):
            image = normalize_image(image, width * height)
            kwargs.update(width=image.width)
            kwargs.update(height=image.height)
            kwargs.update(image=image)

        mask_image = kwargs.get("mask_image")
        if mask_image and isinstance(mask_image, Image.Image):
            mask_image = normalize_image(mask_image, width * height)
            kwargs.update(mask_image=mask_image)
            kwargs.update(width=mask_image.width)
            kwargs.update(height=mask_image.height)

        if "controlnet" in self.__oinstype__.__name__.lower():
            kwargs.update(self._check_controlnet_inference_kwargs(kwargs))

        return kwargs

    def _check_controlnet_inference_kwargs(self,kwargs):
        width = kwargs.get("width")
        height = kwargs.get("height")

        control_image = kwargs.get("control_image")
        if control_image and isinstance(control_image, Image.Image):
            control_image = normalize_image(control_image, width * height)
            kwargs.update(control_image=control_image)

        image = kwargs.get("image")
        # create text to image controlnet condition
        if image is not None and control_image is None:
            image = convert_image_to_canny(image)
            kwargs.update(image=image)
            return kwargs

        # todo: create image to image controlnet condition
        elif image is not None and control_image is not None:
            ...
            return kwargs

        else:
            return kwargs

    def __call__(self,**kwargs):
        kwargs = self.check_inference_kwargs(kwargs)

        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt")
        if isinstance(prompt,list) and isinstance(negative_prompt,list):
            prompt_type = list
        elif isinstance(prompt,list) and not isinstance(negative_prompt,list):
            negative_prompt = [negative_prompt for _ in range(len(prompt))]
            prompt_type = list
        elif not isinstance(prompt,list) and isinstance(negative_prompt,list):
            prompt = [prompt for _ in range(len(negative_prompt))]
            prompt_type = list
        elif isinstance(prompt,str) and isinstance(negative_prompt,str):
            prompt_type = str
        else:
            raise ValueError("The type of prompt and negative_prompt need to be str or list.")

        prompt_str = f"{prompt} {negative_prompt}" if prompt_type == str else f"{' '.join(prompt)} {' '.join(negative_prompt)}"

        skipped_lora_dict = {}

        for lora,weight in self.lora_dict.items():
            # set LoRA strength to 0, if the trigger word (lora name) is not in the prompt or negative prompt
            if lora not in prompt_str:
                skipped_lora_dict.update({lora:weight})
                self.set_lora_strength(lora, 0)
                print(f"LoRA {lora}:{weight} is disable due to {lora} is not in prompts.")

        # if all LoRAs are not triggered, disable LoRAs for acceleration
        if len(skipped_lora_dict) == len(self.lora_dict):
            self.disable_lora()

        try:
            r = SDCLIPEnhancerMixin.__call__(self,**kwargs)
        except Exception as e:
            raise e

        finally:
            # enable disabled LoRAs
            if len(skipped_lora_dict) == len(self.lora_dict):
                self.enable_lora()
            # recover the LoRAs' strength
            if skipped_lora_dict:
                for lora,weight in skipped_lora_dict.items():
                    self.set_lora_strength(lora,weight)
        return r

    def generate_image_and_send_to_telegram(self,
                                            prompt,negative_prompt="",
                                            guidance_scale=2.5,num_inference_steps=20,clip_skip=0,
                                            width=None,height=None,
                                            seed=None,num=1,
                                            use_enhancer=True,**kwargs):
        return generate_image_and_send_to_telegram(
               self,
               prompt=prompt,negative_prompt=negative_prompt,
               guidance_scale=guidance_scale,num_inference_steps=num_inference_steps,clip_skip=clip_skip,
               width=width,height=height,
               seed=seed,num=num,
               use_enhancer=use_enhancer,**kwargs)


    def load_controlnet(self,controlnet_model=None,**kwargs):

        if self._controlnet is None:
            if self.model_version == "1.5":
                controlnet_model = controlnet_model or "lllyasviel/sd-controlnet-canny"
                self._controlnet = load_stable_diffusion_controlnet(controlnet_model,self.model_version,
                                                                    download_kwargs=self.download_kwargs,**kwargs)
                self.text_to_image_controlnet_pipeline = self.enhancer_class(
                    StableDiffusionControlNetPipeline(**self.components,controlnet=self._controlnet),init_sub_pipelines=False)
                self.image_to_image_controlnet_pipeline = self.enhancer_class(
                    StableDiffusionControlNetImg2ImgPipeline(**self.components,controlnet=self._controlnet),init_sub_pipelines=False
                )

            elif self.model_version == "xl":
                controlnet_model =  controlnet_model or "diffusers/controlnet-canny-sdxl-1.0"
                self._controlnet = load_stable_diffusion_controlnet(controlnet_model,self.model_version,
                                                                    download_kwargs=self.download_kwargs,**kwargs)
                self.text_to_image_controlnet_pipeline = self.enhancer_class(
                    StableDiffusionXLControlNetPipeline(**self.components,controlnet=self._controlnet),init_sub_pipelines=False)
                self.image_to_image_controlnet_pipeline = self.enhancer_class(
                    StableDiffusionXLControlNetImg2ImgPipeline(**self.components,controlnet=self._controlnet),init_sub_pipelines=False)

            # todo: sd3 controlnet support
            elif self.model_version == "3":
                ...
                raise NotImplementedError
                # self._controlnet = load_stable_diffusion_controlnet(...,self.model_version)

            self.sub_pipelines.update(text_to_image_controlnet_pipeline=self.text_to_image_controlnet_pipeline)
            self.sub_pipelines.update(image_to_image_controlnet_pipeline=self.image_to_image_controlnet_pipeline)
            self.sync_sub_pipelines_mixin_kwargs()
            self._controlnet.to(self.device)
        else:
            print(f"Controlnet is already implemented.")

    @classmethod
    def from_url(cls,url=None,model_version=None,init_sub_pipelines=True,**kwargs):
        return cls(load_stable_diffusion_pipeline(model=url,model_version=model_version,**kwargs),
                   init_sub_pipelines=init_sub_pipelines)

    def reload(self,url,**kwargs):
        supported_model_version_list = [None, "", "sdxl", "xl", "pony", "1.5", "2", "3", "3.5"]
        if kwargs.get("model_version") not in supported_model_version_list:
            raise ValueError(
                f"Model version: {kwargs.get('model_version')} is not supported, {supported_model_version_list[2:]} is expected.")
        PipelineEnhancerBase.reload(self,url,**kwargs)

    def load_ui(self,_globals=None,**kwargs):
        server = load_stable_diffusion_ui(self,_globals)
        server.launch(quiet=True,**kwargs)
        return server
