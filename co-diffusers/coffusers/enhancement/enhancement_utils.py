from ..utils import EasyInitSubclass
from diffusers.schedulers import DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler
from diffusers.schedulers import KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler
from diffusers.schedulers import EulerDiscreteScheduler,EulerAncestralDiscreteScheduler
from diffusers.schedulers import HeunDiscreteScheduler
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.schedulers import DEISMultistepScheduler
from diffusers.schedulers import UniPCMultistepScheduler


class PipelineEnhancerBase(EasyInitSubclass):
    overrides = ["set_scheduler"]

    def __init__(self,__oins__):
        EasyInitSubclass.__init__(self,__oins__)
        self._scheduler = self.scheduler

        # from https://huggingface.co/docs/diffusers/api/schedulers/overview
        self.scheduler_map = {
            "DPM++ 2M": (DPMSolverMultistepScheduler, {}),
            "DPM++ 2M KARRAS": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
            "DPM++ 2M SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
            "DPM++ 2M SDE KARRAS": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
            "DPM++ 2S a": (DPMSolverSinglestepScheduler, {}),
            "DPM++ 2S a KARRAS": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
            "DPM++ SDE": (DPMSolverSinglestepScheduler, {}),
            "DPM++ SDE KARRAS": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
            "DPM2": (KDPM2DiscreteScheduler, {}),
            "DPM2 KARRAS": (KDPM2DiscreteScheduler, {"use_karras_sigmas": True}),
            "DPM2 a": (KDPM2AncestralDiscreteScheduler, {}),
            "DPM2 a KARRAS": (KDPM2AncestralDiscreteScheduler, {"use_karras_sigmas": True}),
            "Euler": (EulerDiscreteScheduler, {}),
            "Euler a": (EulerAncestralDiscreteScheduler, {}),
            "Heun": (HeunDiscreteScheduler, {}),
            "LMS": (LMSDiscreteScheduler, {}),
            "LMS KARRAS": (LMSDiscreteScheduler, {"use_karras_sigmas": True}),
            "DEIS": (DEISMultistepScheduler, {}),
            "UNIPC": (UniPCMultistepScheduler, {}),
        }

    def set_scheduler(self,scheduler_type,**kwargs):
        if not isinstance(scheduler_type,str):
            self.scheduler = scheduler_type.from_config(self.scheduler.config,**kwargs)
        else:
            scheduler_map = self.scheduler_map
            if scheduler_type.upper() in scheduler_map:
                self.scheduler = scheduler_map[scheduler_type.upper()][0].from_config(self.scheduler.config,**scheduler_map[scheduler_type.upper()][1])
            else:
                print(f"{scheduler_type} is not supported yet.")
