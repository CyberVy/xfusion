from ..utils import EasyInitSubclass
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

pipeline_map = {
    "1.5":(StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline),
    "xl":(StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,StableDiffusionXLInpaintPipeline),
    "3":(StableDiffusion3Pipeline,StableDiffusion3Img2ImgPipeline,StableDiffusion3InpaintPipeline),
    "flux": (FluxPipeline,FluxImg2ImgPipeline,FluxInpaintPipeline)
}

class PipelineEnhancerBase(EasyInitSubclass):
    overrides = ["enhancer_class","model_version","pipeline_class","model_name","_scheduler","scheduler_map",
                 "check_original_pipeline","set_scheduler","reset_scheduler","to","load_i2i_pipeline","load_inpainting_pipeline"]

    def __init__(self,__oins__):
        EasyInitSubclass.__init__(self,__oins__)
        self.enhancer_class = object.__getattribute__(self,"__class__")
        self.model_version,self.pipeline_class = self.check_original_pipeline()
        self.model_name = self.name_or_path
        self._scheduler = self.scheduler
        self.scheduler_map = scheduler_map

    def check_original_pipeline(self):
        for model_version, pipeline_class_tuple in pipeline_map.items():
            for pipeline_class in pipeline_class_tuple:
                if pipeline_class == self.__oinstype__:
                    return model_version,pipeline_class
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

    def load_i2i_pipeline(self,**kwargs):
        return self.enhancer_class(pipeline_map[self.model_version][1](**self.components,**kwargs))

    def load_inpainting_pipeline(self,**kwargs):
        return self.enhancer_class(pipeline_map[self.model_version][2](**self.components, **kwargs))

class FromURLMixin:
    overrides = ["from_url"]

    @classmethod
    def from_url(cls,url,**kwargs):
        raise NotImplementedError(f"{cls} not implement 'from_url'")
