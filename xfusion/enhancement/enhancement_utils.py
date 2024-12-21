from ..utils import EasyInitSubclass,delete
from ..ui.ui_utils import UIMixin
from ..download import DownloadArgumentsMixin,download_file
from ..message import TGBotMixin
from diffusers.schedulers import DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler
from diffusers.schedulers import KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler
from diffusers.schedulers import EulerDiscreteScheduler,EulerAncestralDiscreteScheduler
from diffusers.schedulers import HeunDiscreteScheduler
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.schedulers import DEISMultistepScheduler
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers import FluxPipeline,FluxImg2ImgPipeline,FluxInpaintPipeline
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline
from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,StableDiffusionXLInpaintPipeline
from diffusers import StableDiffusion3Pipeline,StableDiffusion3Img2ImgPipeline,StableDiffusion3InpaintPipeline

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

# pipeline_type 0,1,2 -> text_to_image, image_to_image, inpainting
pipeline_map = {
    "1.5":(StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline),
    "xl":(StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,StableDiffusionXLInpaintPipeline),
    "3":(StableDiffusion3Pipeline,StableDiffusion3Img2ImgPipeline,StableDiffusion3InpaintPipeline),
    "flux": (FluxPipeline,FluxImg2ImgPipeline,FluxInpaintPipeline)
}

class FromURLMixin:
    overrides = ["from_url"]

    @classmethod
    def from_url(cls,url,**kwargs):
        raise NotImplementedError(f"{cls} not implement 'from_url'")


def load_lora(pipeline,lora_uri,lora_name,download_kwargs=None):

    use_internet = True
    if lora_uri.startswith(".") or lora_uri.startswith("/") or lora_uri.startswith("~"):
        use_internet = False
    if download_kwargs is None:
        download_kwargs = {}
    if use_internet:
        lora_path = download_file(lora_uri,**download_kwargs)
        pipeline.load_lora_weights(lora_path,adapter_name=lora_name)
    else:
        pipeline.load_lora_weights(lora_uri,adapter_name=lora_name)
class LoraEnhancerMixin(DownloadArgumentsMixin,EasyInitSubclass):
    # __oins__ here is the pipeline instance to implement.
    __oins__ = None
    overrides = ["lora_dict","set_lora","set_lora_strength","delete_adapters"]

    def __init__(self):
        DownloadArgumentsMixin.__init__(self)
        self.lora_dict = {}
        self.download_kwargs.update(directory="./lora")

    def set_lora(self,lora_uri,lora_name,weight=0.4):
        if lora_name not in self.lora_dict:
            load_lora(self,lora_uri,lora_name,self.download_kwargs)

        self.set_lora_strength(lora_name,weight)

    def set_lora_strength(self,lora_name,weight):
        self.lora_dict.update({lora_name:weight})
        self.set_adapters(list(self.lora_dict.keys()),list(self.lora_dict.values()))

    def delete_adapters(self,adapter_names):
        self.__oins__.delete_adapters(adapter_names)
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for name in adapter_names:
            self.lora_dict.pop(name)


class PipelineEnhancerBase(LoraEnhancerMixin,TGBotMixin,FromURLMixin,UIMixin,EasyInitSubclass):
    overrides = ["enhancer_class","model_version","pipeline_type","pipeline_class",
                 "model_name","_scheduler","scheduler_map","sub_pipelines",
                 "image_to_image_pipeline","inpainting_pipeline",
                 "check_original_pipeline","set_scheduler","reset_scheduler",
                 "to","enable_model_cpu_offload",
                 "image_normalize","clear"]

    def __init__(self,__oins__,init_sub_pipelines=True):
        EasyInitSubclass.__init__(self,__oins__)
        TGBotMixin.__init__(self)
        LoraEnhancerMixin.__init__(self)
        self.enhancer_class = object.__getattribute__(self,"__class__")
        # pipeline_type 0,1,2 -> text_to_image, image_to_image, inpainting
        self.model_version,self.pipeline_type,self.pipeline_class = self.check_original_pipeline()
        self.model_name = self.name_or_path
        self._scheduler = self.scheduler
        self.scheduler_map = scheduler_map

        self.sub_pipelines = []
        if init_sub_pipelines:
            if self.pipeline_type != 0:
                self.text_to_image_pipeline = self.enhancer_class(pipeline_map[self.model_version][0](**self.components),
                                                                  init_sub_pipelines=False)
                self.text_to_image_pipeline.telegram_kwargs = self.telegram_kwargs
                self.text_to_image_pipeline.lora_dict = self.lora_dict
                self.text_to_image_pipeline.download_kwargs = self.download_kwargs
                self.sub_pipelines.append(self.text_to_image_pipeline)
            else:
                self.text_to_image_pipeline = self

            if self.pipeline_type != 1:
                self.image_to_image_pipeline = self.enhancer_class(pipeline_map[self.model_version][1](**self.components),
                                                                   init_sub_pipelines=False)
                self.image_to_image_pipeline.telegram_kwargs = self.telegram_kwargs
                self.image_to_image_pipeline.lora_dict = self.lora_dict
                self.image_to_image_pipeline.download_kwargs = self.download_kwargs
                self.sub_pipelines.append(self.image_to_image_pipeline)
            else:
                self.image_to_image_pipeline = self

            if self.pipeline_type != 2:
                self.inpainting_pipeline =  self.enhancer_class(pipeline_map[self.model_version][2](**self.components),
                                                                init_sub_pipelines=False)
                self.inpainting_pipeline.telegram_kwargs = self.telegram_kwargs
                self.inpainting_pipeline.lora_dict = self.lora_dict
                self.inpainting_pipeline.download_kwargs = self.download_kwargs
                self.sub_pipelines.append(self.inpainting_pipeline)
            else:
                self.inpainting_pipeline = self

    def check_original_pipeline(self):
        for model_version, pipeline_class_tuple in pipeline_map.items():
            for pipeline_type,pipeline_class in enumerate(pipeline_class_tuple):
                if pipeline_class == self.__oinstype__:
                    return model_version,pipeline_type,pipeline_class
        raise TypeError(f"{self.__oinstype__} is not supported yet.")

    def set_scheduler(self,scheduler_type,**kwargs):
        if not isinstance(scheduler_type,str):
            self.scheduler = scheduler_type.from_config(self.scheduler.config,**kwargs)
        else:
            if scheduler_type.upper() in self.scheduler_map:
                self.scheduler = self.scheduler_map[scheduler_type.upper()][0].from_config(self.scheduler.config,**self.scheduler_map[scheduler_type.upper()][1])
            else:
                print(f"{scheduler_type} is not supported yet.")

    def reset_scheduler(self):
        self.scheduler = self._scheduler

    def to(self, *args, **kwargs):
        self.__oins__ = self.__oins__.to(*args, **kwargs)
        return self

    def enable_model_cpu_offload(self,*args,**kwargs):
        self.__oins__.enable_model_cpu_offload(self,*args,**kwargs)
        for pipeline in self.sub_pipelines:
            pipeline.__oins__.enable_model_cpu_offload(*args,**kwargs)

    # todo:
    def image_normalize(self,image):
        ...

    def clear(self):
        for component in self.components.values():
            delete(component)
