import os
import torch
from .component_const import flux_vae_url_list
from .component_const import default_flux_transformer_url
from .component_const import flux_scheduler_url_list
from .component_utils import load_t5_tokenizer,load_t5_encoder
from .component_utils import load_clip_tokenizer,load_clip_encoder
from ..download import download_file
from ..const import HF_HUB_TOKEN
from diffusers import FluxPipeline
from diffusers import AutoencoderKL,FluxTransformer2DModel,FlowMatchEulerDiscreteScheduler
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig


def load_flux_pipeline(model=None,download_kwargs=None, **kwargs):
    use_internet = True
    model = "eramth/flux-4bit" if not model else model
    if download_kwargs is None:
        download_kwargs = {}
    if kwargs.get("token") is None:
        kwargs.update(token=HF_HUB_TOKEN)
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if model.startswith(".") or model.startswith("/") or model.startswith("~"):
        use_internet = False
    
    kwargs.update(cache_dir=download_kwargs.get("directory"))

    if use_internet:
        # from Hugging face
        if not (model.startswith("http://") or model.startswith("https://")):
            return FluxPipeline.from_pretrained(model,**kwargs)
        # todo: single file support
        else:
            raise ValueError("Only support load the Flux model from huggingface so far.")
    else:
        if not model.endswith(".safetensors"):
            return FluxPipeline.from_pretrained(model,**kwargs)
        # todo: single file support
        else:
            raise ValueError("Only support load the Flux model from huggingface so far.")


def get_flux_transformer_files(directory,
    uri=default_flux_transformer_url,
    **kwargs):
    file_list = [download_file(uri, directory=directory,**kwargs)]
    return file_list

def get_flux_vae_files(directory,**kwargs):
    url_list = flux_vae_url_list
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def get_flux_scheduler_files(directory,**kwargs):
    url_list = flux_scheduler_url_list
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def load_flux_transformer(
        directory=None,uri=default_flux_transformer_url,
        delete_internet_files=False,download_kwargs=None,**kwargs):
    """
    :param directory: the transformer where to download
    :param uri: the uri of the transformer
    :param delete_internet_files:
    :param download_kwargs:
    :param kwargs: passed in FluxTransformer2DModel.from_single_file
    :return:
    """
    use_internet = True
    if uri.startswith(".") or uri.startswith("/") or uri.startswith("~"):
        use_internet = False
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=HF_HUB_TOKEN)

    directory = "./transformer" if directory is None else directory

    if use_internet:
        # from single file
        if uri.startswith("http://") or uri.startswith("https://"):
            file_list = get_flux_transformer_files(directory,uri,**download_kwargs)
            transformer = FluxTransformer2DModel.from_single_file(file_list[0],**kwargs)
            if delete_internet_files:
                os.remove(file_list[0])
        # from huggingface
        else:
            transformer = FluxTransformer2DModel.from_pretrained(uri,subfolder="transformer",**kwargs)
    else:
        if uri.endswith(".safetensors"):
            transformer = FluxTransformer2DModel.from_single_file(uri,**kwargs)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(uri,subfolder="transformer",**kwargs)

    print("Transformer ready.")

    return transformer

def load_flux_vae(directory=None,use_local_files=False,delete_internet_files=False,download_kwargs=None,**kwargs):
    """
    :param directory:
    :param use_local_files:
    :param delete_internet_files:
    :param download_kwargs:
    :param kwargs: passed in AutoencoderKL.from_pretrained
    :return:
    """
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=HF_HUB_TOKEN)

    directory = "./vae" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_flux_vae_files(directory,**download_kwargs)

    vae = AutoencoderKL.from_pretrained(directory,**kwargs)
    print("VAE ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return vae

def load_flux_scheduler(directory=None,use_local_files=False,delete_internet_files=False,download_kwargs=None,**kwargs):
    """
    :param directory:
    :param use_local_files:
    :param delete_internet_files:
    :param download_kwargs:
    :param kwargs: passed in FlowMatchEulerDiscreteScheduler.from_pretrained
    :return:
    """
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=HF_HUB_TOKEN)

    directory = "./scheduler" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_flux_scheduler_files(directory,**download_kwargs)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(directory,**kwargs)
    print("Scheduler ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return scheduler

def _load_flux_pipeline_by_components(uri=None,delete_internet_files=False,download_kwargs=None,**kwargs):
    uri = default_flux_transformer_url if uri is None else uri
    download_kwargs = {} if download_kwargs is None else download_kwargs

    quantization_config = kwargs.pop("quantization_config",{"load_in_4bit":True,"bnb_4bit_compute_dtype":torch.float16})

    transformer = load_flux_transformer("/transformer",uri=uri,download_kwargs=download_kwargs,
                                        delete_internet_files=delete_internet_files,
                                        quantization_config=DiffusersBitsAndBytesConfig(**quantization_config) if quantization_config else None,
                                        **kwargs)

    t5_tokenizer = load_t5_tokenizer(download_kwargs=download_kwargs,**kwargs)
    t5_encoder = load_t5_encoder(download_kwargs=download_kwargs,
                                 quantization_config=TransformersBitsAndBytesConfig(**quantization_config) if quantization_config else None,
                                 **kwargs)

    clip_tokenizer = load_clip_tokenizer(download_kwargs=download_kwargs,delete_internet_files=delete_internet_files,**kwargs)
    clip_encoder = load_clip_encoder(download_kwargs=download_kwargs,delete_internet_files=delete_internet_files,**kwargs)

    vae = load_flux_vae(download_kwargs=download_kwargs,delete_internet_files=delete_internet_files,**kwargs)
    scheduler = load_flux_scheduler(download_kwargs=download_kwargs,delete_internet_files=delete_internet_files,**kwargs)

    pipeline = FluxPipeline(transformer=transformer, vae=vae, scheduler=scheduler,
                            text_encoder=clip_encoder, text_encoder_2=t5_encoder,
                            tokenizer=clip_tokenizer, tokenizer_2=t5_tokenizer)
    print("FLUX Pipeline ready.")
    return pipeline
