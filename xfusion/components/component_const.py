# components_utils.py

SD_V1_CONFIG_PATH = f"{'/'.join(__file__.split('/')[:-1])}/CONFIG/SD_V1"
SD_V2_CONFIG_PATH =  f"{'/'.join(__file__.split('/')[:-1])}/CONFIG/SD_V2"
SD_XL_CONFIG_PATH =  f"{'/'.join(__file__.split('/')[:-1])}/CONFIG/SD_XL"
SD_3_CONFIG_PATH =  f"{'/'.join(__file__.split('/')[:-1])}/CONFIG/SD_3"

SD_V1_V2_CONTROLNET_CONFIG_PATH = f"{'/'.join(__file__.split('/')[:-1])}/CONFIG/SD_V1_V2_CONTROLNET"
SD_XL_CONTROLNET_CONFIG_PATH = f"{'/'.join(__file__.split('/')[:-1])}/CONFIG/SD_XL_CONTROLNET"

FLUX_CONFIG_PATH =  f"{'/'.join(__file__.split('/')[:-1])}/CONFIG/FLUX"


t5_tokenizer_url_list = [
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/special_tokens_map.json?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/spiece.model?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/tokenizer.json?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/tokenizer_config.json?download=true",
]

t5_encoder_url_list = [
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model.safetensors.index.json?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/config.json?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model-00001-of-00002.safetensors?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model-00002-of-00002.safetensors?download=true"
]

clip_tokenizer_url_list = [
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/merges.txt?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/special_tokens_map.json?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/tokenizer_config.json?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/vocab.json?download=true"
]

clip_encoder_url_list = [
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/config.json?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/model.safetensors?download=true"
]

# flux_components.py

default_flux_transformer_url = "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=pruned&fp=fp8"

flux_vae_url_list = [
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/config.json?download=true",
]
flux_scheduler_url_list = [
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/scheduler/scheduler_config.json?download=true"
]

# stable_diffusion_components.py

# https://civitai.com/models/277058/epicrealism-xl
default_stable_diffusion_model_url = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"

default_sd_xl_controlnet_model_url = "https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
default_sd_v1_v2_controlnet_model_url = "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"

# collection

# https://civitai.com/models/134778/xxmix0731girl
lora_sdxl_xxmix_girl = {"name":"xxmix_girl","url":"https://civitai.com/api/download/models/148469?type=Model&format=SafeTensor"}
# https://civitai.com/models/200255?modelVersionId=997426
lora_sdxl_hand = {"name":"hand","url":"https://civitai.com/api/download/models/997426?type=Model&format=SafeTensor"}
