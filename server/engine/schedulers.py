from typing import Dict, Tuple, Type, Union
from diffusers.schedulers import (
    SchedulerMixin,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
)
from .types import SchedulerConfigMap, SchedulerConfigValue

DIFFUSION_SCHEDULER_MAP: SchedulerConfigMap = {
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
