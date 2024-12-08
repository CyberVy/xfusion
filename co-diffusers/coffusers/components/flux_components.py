import os
import torch
from .component_const import flux_vae_url_list
from .component_const import default_flux_transformer_url
from .component_const import flux_scheduler_url_list
from .component_utils import load_t5_tokenizer,load_t5_encoder
from .component_utils import load_clip_tokenizer,load_clip_encoder
from .component_utils import get_t5_encoder_files
from ..download import download_file
from ..const import HF_HUB_TOKEN
from ..utils import threads_execute
from diffusers import AutoencoderKL,FluxPipeline,FluxTransformer2DModel,FlowMatchEulerDiscreteScheduler
from optimum.quanto import freeze,qfloat8,quantize


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
        use_local_files=False, delete_internet_files=True,download_kwargs=None,**kwargs):
    """
    :param directory: the transformer where to download
    :param uri: the uri of the transformer
    :param use_local_files:
    :param delete_internet_files:
    :param download_kwargs:
    :param kwargs: passed in FluxTransformer2DModel.from_single_file
    :return:
    """
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=HF_HUB_TOKEN)

    if use_local_files and directory is not None:
        print("Warning: The parameter 'directory' is the transformer where to download, when 'use_local_files' is true, please make sure it is None. ")

    directory = "./transformer" if directory is None else directory

    if use_local_files and uri.startswith("http"):
        raise ValueError("The uri is not a local file path, a local file path is  required.")

    file_list = []
    if not use_local_files:
        file_list = get_flux_transformer_files(directory,uri,**download_kwargs)
        transformer = FluxTransformer2DModel.from_single_file(file_list[0],**kwargs)
    else:
        transformer = FluxTransformer2DModel.from_single_file(uri,**kwargs)
    print("Transformer ready.")

    if not use_local_files and delete_internet_files:
        os.remove(file_list[0])
    return transformer

def load_flux_vae(directory=None,use_local_files=False,delete_internet_files=True,download_kwargs=None,**kwargs):
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

def load_flux_scheduler(directory=None,use_local_files=False,delete_internet_files=True,download_kwargs=None,**kwargs):
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

def load_flux_pipeline(uri=default_flux_transformer_url,download_kwargs=None,**kwargs):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    def q(model):
        quantize(model,weights=qfloat8)
        freeze(model)

    def _get_t5_encoder_files_mute(directory):
        return get_t5_encoder_files(directory,mute=True,)

    _t5_thread = threads_execute(_get_t5_encoder_files_mute,("./t5_encoder",),_await=False)[0]

    transformer = load_flux_transformer("/transformer",uri=uri,download_kwargs=download_kwargs,**kwargs)
    threads_execute(q,(transformer,),_await=True)

    _t5_thread.join()
    t5_tokenizer, t5_encoder = load_t5_tokenizer(download_kwargs=download_kwargs,**kwargs), load_t5_encoder(directory="./t5_encoder",use_local_files=True,**kwargs)
    threads_execute(q,(t5_encoder,),_await=True)

    clip_tokenizer, clip_encoder = load_clip_tokenizer(download_kwargs=download_kwargs,**kwargs), load_clip_encoder(download_kwargs=download_kwargs,**kwargs)
    vae = load_flux_vae(download_kwargs=download_kwargs,**kwargs)
    scheduler = load_flux_scheduler(download_kwargs=download_kwargs,**kwargs)

    pipeline = FluxPipeline(transformer=transformer, vae=vae, scheduler=scheduler,
                            text_encoder=clip_encoder, text_encoder_2=t5_encoder,
                            tokenizer=clip_tokenizer, tokenizer_2=t5_tokenizer)
    print("FLUX Pipeline ready.")
    return pipeline
