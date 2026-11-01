import os

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

SD_V1_CONFIG_PATH = os.path.join(CONFIGS_DIR, "SD_V1")
SD_V2_CONFIG_PATH = os.path.join(CONFIGS_DIR, "SD_V2")
SD_XL_CONFIG_PATH = os.path.join(CONFIGS_DIR, "SD_XL")
SD_3_CONFIG_PATH = os.path.join(CONFIGS_DIR, "SD_3")

SD_V1_V2_CONTROLNET_CONFIG_PATH = os.path.join(CONFIGS_DIR, "SD_V1_V2_CONTROLNET")
SD_XL_CONTROLNET_CONFIG_PATH = os.path.join(CONFIGS_DIR, "SD_XL_CONTROLNET")

FLUX_CONFIG_PATH = os.path.join(CONFIGS_DIR, "FLUX")

default_flux_transformer_url: str = (
    "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=pruned&fp=fp8"
)

default_stable_diffusion_model_url: str = (
    "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
)

default_sd_v1_v2_controlnet_model_url: str = (
    "https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
)
default_sd_xl_controlnet_model_url: str = (
    "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
)
