import os
os.environ["CIVITAI_TOKEN"] = "CIVITAI_TOKEN"
token = "HF_HUB_TOKEN"

from xfusion.download import download_file
from xfusion.utils import delete
from diffusers import FluxTransformer2DModel
from diffusers.loaders.single_file_utils import load_state_dict
import torch
import shutil
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig


def convert_civitai_flux_transformer_and_upload_to_hf(url,repo_name,token=None):

    if not url:
        raise ValueError("A valid URL is required.")

    if token is None:
        raise ValueError("A valid token is required.")

    file_path = download_file(url,filename="diffusion_pytorch_model.safetensors",directory="/flux-transformer-converter")
    state_dict = load_state_dict(file_path)
    torch_dtype = state_dict[list(state_dict.keys())[0]].dtype
    transformer = FluxTransformer2DModel.from_single_file(file_path,torch_dtype=torch_dtype,token=token)
    r = transformer.push_to_hub(repo_name,token=token)
    delete(transformer)
    torch.cuda.empty_cache()
    return r

def convert_flux_transformer_into_4bit_and_upload_to_hf(repo_id,repo_name,target_dtype=None,token=None):

    if token is None:
        raise ValueError("A valid token is required.")

    if target_dtype is None:
        target_dtype = torch.float16

    quantization_config = {"load_in_4bit":True,"bnb_4bit_compute_dtype":target_dtype,"bnb_4bit_quant_type":"nf4"}
    quantization_config = DiffusersBitsAndBytesConfig(**quantization_config)
    transformer_4bit = FluxTransformer2DModel.from_pretrained(repo_id,torch_dtype=target_dtype,cache_dir="/flux-transformer-converter",quantization_config=quantization_config)
    r =  transformer_4bit.push_to_hub(repo_name,token=token)
    delete(transformer_4bit)
    torch.cuda.empty_cache()
    return r
