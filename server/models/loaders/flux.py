import os
import torch
from typing import Optional, Dict, Any, List

from diffusers import (
    FluxPipeline,
    AutoencoderKL,
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import (
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
    T5EncoderModel,
    T5Tokenizer,
    CLIPTextModel,
    CLIPTokenizer,
)

from ..constants import (
    default_flux_transformer_url,
    flux_vae_url_list,
    flux_scheduler_url_list,
    t5_tokenizer_url_list,
    t5_encoder_url_list,
    clip_tokenizer_url_list,
    clip_encoder_url_list,
)
from ...infra.downloader import download_file
from ...infra.utils import naive_async
from ...infra import config


def load_flux_pipeline(
    model: Optional[str] = None,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    use_internet = True
    model = model or "eramth/flux-4bit"
    if download_kwargs is None:
        download_kwargs = {}
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if model.startswith(".") or model.startswith("/") or model.startswith("~"):
        use_internet = False

    kwargs.update(cache_dir=download_kwargs.get("directory"))

    if use_internet:
        if not (model.startswith("http://") or model.startswith("https://")):
            return FluxPipeline.from_pretrained(model, **kwargs)
        else:
            raise ValueError("Only loading FLUX from Hugging Face repo is supported directly.")
    else:
        if not model.endswith(".safetensors"):
            return FluxPipeline.from_pretrained(model, **kwargs)
        else:
            raise ValueError("Only loading FLUX from directory or Hugging Face is supported.")


def get_flux_transformer_files(directory: str, uri: str = default_flux_transformer_url, **kwargs):
    return [download_file(uri, directory=directory, **kwargs)]


def get_flux_vae_files(directory: str, **kwargs) -> List[str]:
    return [download_file(url, directory=directory, **kwargs) for url in flux_vae_url_list]


def get_flux_scheduler_files(directory: str, **kwargs) -> List[str]:
    return [download_file(url, directory=directory, **kwargs) for url in flux_scheduler_url_list]


def load_flux_transformer(
    directory: Optional[str] = None,
    uri: str = default_flux_transformer_url,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    use_internet = True
    if uri.startswith(".") or uri.startswith("/") or uri.startswith("~"):
        use_internet = False
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)

    directory = directory or "./transformer"

    if use_internet:
        if uri.startswith("http://") or uri.startswith("https://"):
            file_list = get_flux_transformer_files(directory, uri, **download_kwargs)
            transformer = FluxTransformer2DModel.from_single_file(file_list[0], **kwargs)
            if delete_internet_files:
                os.remove(file_list[0])
        else:
            transformer = FluxTransformer2DModel.from_pretrained(uri, subfolder="transformer", **kwargs)
    else:
        if uri.endswith(".safetensors"):
            transformer = FluxTransformer2DModel.from_single_file(uri, **kwargs)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(uri, subfolder="transformer", **kwargs)

    return transformer


def load_t5_tokenizer(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)

    directory = directory or "./t5_tokenizer"
    file_list = []
    if not use_local_files:
        file_list = [download_file(url, directory=directory, **download_kwargs) for url in t5_tokenizer_url_list]

    tokenizer = T5Tokenizer.from_pretrained(directory, **kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return tokenizer


def load_t5_encoder(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)

    directory = directory or "./t5_encoder"
    file_list = []
    if not use_local_files:
        os.makedirs(directory, exist_ok=True)
        futures = [
            naive_async(download_file)(url, directory=directory, mute=(i != 0), **download_kwargs)
            for i, url in enumerate(t5_encoder_url_list)
        ]
        file_list = [fut.result() for fut in futures]

    encoder = T5EncoderModel.from_pretrained(directory, **kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return encoder


def load_clip_tokenizer(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)

    directory = directory or "./clip_tokenizer"
    file_list = []
    if not use_local_files:
        file_list = [download_file(url, directory=directory, **download_kwargs) for url in clip_tokenizer_url_list]

    tokenizer = CLIPTokenizer.from_pretrained(directory, **kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return tokenizer


def load_clip_encoder(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)

    directory = directory or "./clip_encoder"
    file_list = []
    if not use_local_files:
        file_list = [download_file(url, directory=directory, **download_kwargs) for url in clip_encoder_url_list]

    encoder = CLIPTextModel.from_pretrained(directory, **kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return encoder


def load_flux_vae(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)

    directory = directory or "./vae"
    file_list = []
    if not use_local_files:
        file_list = get_flux_vae_files(directory, **download_kwargs)

    vae = AutoencoderKL.from_pretrained(directory, **kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return vae


def load_flux_scheduler(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=config.HF_HUB_TOKEN)

    directory = directory or "./scheduler"
    file_list = []
    if not use_local_files:
        file_list = get_flux_scheduler_files(directory, **download_kwargs)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(directory, **kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return scheduler


def load_flux_pipeline_by_components(
    uri: Optional[str] = None,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
):
    uri = uri or default_flux_transformer_url
    download_kwargs = download_kwargs or {}
    quant_cfg = kwargs.pop("quantization_config", {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16})

    transformer = load_flux_transformer(
        "/transformer",
        uri=uri,
        download_kwargs=download_kwargs,
        delete_internet_files=delete_internet_files,
        quantization_config=DiffusersBitsAndBytesConfig(**quant_cfg) if quant_cfg else None,
        **kwargs
    )
    t5_tok = load_t5_tokenizer(download_kwargs=download_kwargs, **kwargs)
    t5_enc = load_t5_encoder(
        download_kwargs=download_kwargs,
        quantization_config=TransformersBitsAndBytesConfig(**quant_cfg) if quant_cfg else None,
        **kwargs
    )
    clip_tok = load_clip_tokenizer(download_kwargs=download_kwargs, delete_internet_files=delete_internet_files, **kwargs)
    clip_enc = load_clip_encoder(download_kwargs=download_kwargs, delete_internet_files=delete_internet_files, **kwargs)
    vae = load_flux_vae(download_kwargs=download_kwargs, delete_internet_files=delete_internet_files, **kwargs)
    sched = load_flux_scheduler(download_kwargs=download_kwargs, delete_internet_files=delete_internet_files, **kwargs)

    return FluxPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=sched,
        text_encoder=clip_enc,
        text_encoder_2=t5_enc,
        tokenizer=clip_tok,
        tokenizer_2=t5_tok,
    )
