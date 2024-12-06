import os
import torch
from .component_utils import load_t5_tokenizer,load_t5_encoder
from .component_utils import load_clip_tokenizer,load_clip_encoder
from .component_utils import get_t5_encoder_files
from ..download import download_file
from ..const import hf_token
from ..utils import threads_execute
from diffusers import AutoencoderKL,FluxPipeline,FluxTransformer2DModel,FlowMatchEulerDiscreteScheduler
from optimum.quanto import freeze,qfloat8,quantize


def get_flux_transformer_files(directory,
    uri="https://civitai.com/api/download/models/979329?type=Model&format=SafeTensor&size=full&fp=fp16",
    **kwargs):
    file_list = [download_file(uri, directory=directory,**kwargs)]
    return file_list


def get_flux_vae_files(directory,**kwargs):
    url_list = [
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/config.json?download=true",
    ]
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def get_flux_scheduler_files(directory,**kwargs):
    url_list = [
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/scheduler/scheduler_config.json?download=true"
    ]
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def load_flux_transformer(
        directory=None,uri="https://civitai.com/api/download/models/979329?type=Model&format=SafeTensor&size=full&fp=fp16",
        use_local_files=False, delete_internet_files=True):
    """

    :param directory: the transformer where to download
    :param uri: the uri of the transformer
    :param use_local_files:
    :param delete_internet_files:
    :return:
    """
    if use_local_files and directory is not None:
        print("Warning: The parameter 'directory' is the transformer where to download, when 'use_local_files' is true, please make sure it is None. ")

    directory = "./transformer" if directory is None else directory

    if use_local_files and uri.startswith("http"):
        raise ValueError("The uri is not a local file path, a local file path is  required.")

    file_list = []
    if not use_local_files:
        file_list = get_flux_transformer_files(directory,uri)
        transformer = FluxTransformer2DModel.from_single_file(file_list[0], torch_dtype=torch.float16, token=hf_token)
    else:
        transformer = FluxTransformer2DModel.from_single_file(uri, torch_dtype=torch.float16, token=hf_token)
    print("Transformer ready.")

    if not use_local_files and delete_internet_files:
        os.remove(file_list[0])
    return transformer

def load_flux_vae(directory=None,use_local_files=False,delete_internet_files=True):
    directory = "./vae" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_flux_vae_files(directory)

    vae = AutoencoderKL.from_pretrained(directory, torch_dtype=torch.float16)
    print("VAE ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return vae

def load_flux_scheduler(directory=None,use_local_files=False,delete_internet_files=True):
    directory = "./scheduler" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_flux_scheduler_files(directory)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(directory)
    print("Scheduler ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return scheduler

def load_flux_pipeline(uri="https://civitai.com/api/download/models/979329?type=Model&format=SafeTensor&size=full&fp=fp16",**kwargs):
    def q(model):
        quantize(model,weights=qfloat8)
        freeze(model)

    _t5_thread = threads_execute(get_t5_encoder_files,("./t5_encoder",),_await=False)[0]

    transformer = load_flux_transformer("~/",uri=uri)
    threads_execute(q,(transformer,),_await=False)

    _t5_thread.join()
    t5_tokenizer, t5_encoder = load_t5_tokenizer(), load_t5_encoder()
    threads_execute(q,(t5_encoder,),_await=True)

    clip_tokenizer, clip_encoder = load_clip_tokenizer(), load_clip_encoder()
    vae = load_flux_vae()
    scheduler = load_flux_scheduler()

    pipeline = FluxPipeline(transformer=transformer, vae=vae, scheduler=scheduler,
                            text_encoder=clip_encoder, text_encoder_2=t5_encoder,
                            tokenizer=clip_tokenizer, tokenizer_2=t5_tokenizer,**kwargs)
    print("FLUX Pipeline ready.")
    return pipeline
