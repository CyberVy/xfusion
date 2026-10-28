from __future__ import annotations
import os
from typing import Optional, Dict, List, Union

try:
    import torch
except (ImportError, AttributeError):
    torch = None  # type: ignore

LoaderKwargValue = Union[str, float, int, bool, None]

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
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> FluxPipeline:
    use_internet = True
    model_name = model or "eramth/flux-4bit"
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN
    if call_kwargs.get("torch_dtype") is None and torch is not None:
        call_kwargs["torch_dtype"] = torch.float16
    if model_name.startswith(".") or model_name.startswith("/") or model_name.startswith("~"):
        use_internet = False

    call_kwargs["cache_dir"] = dl_kwargs.get("directory")

    if use_internet:
        if not (model_name.startswith("http://") or model_name.startswith("https://")):
            return FluxPipeline.from_pretrained(model_name, **call_kwargs)
        else:
            raise ValueError("Only loading FLUX from Hugging Face repo is supported directly.")
    else:
        if not model_name.endswith(".safetensors"):
            return FluxPipeline.from_pretrained(model_name, **call_kwargs)
        else:
            raise ValueError("Only loading FLUX from directory or Hugging Face is supported.")


def get_flux_transformer_files(
    directory: str,
    uri: str = default_flux_transformer_url,
    **kwargs: LoaderKwargValue
) -> List[str]:
    return [download_file(uri, directory=directory, **kwargs)]


def get_flux_vae_files(directory: str, **kwargs: LoaderKwargValue) -> List[str]:
    return [download_file(url, directory=directory, **kwargs) for url in flux_vae_url_list]


def get_flux_scheduler_files(directory: str, **kwargs: LoaderKwargValue) -> List[str]:
    return [download_file(url, directory=directory, **kwargs) for url in flux_scheduler_url_list]


def load_flux_transformer(
    directory: Optional[str] = None,
    uri: str = default_flux_transformer_url,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> FluxTransformer2DModel:
    use_internet = True
    if uri.startswith(".") or uri.startswith("/") or uri.startswith("~"):
        use_internet = False
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("torch_dtype") is None and torch is not None:
        call_kwargs["torch_dtype"] = torch.float16
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN

    target_dir = directory or "./transformer"

    if use_internet:
        if uri.startswith("http://") or uri.startswith("https://"):
            file_list = get_flux_transformer_files(target_dir, uri, **dl_kwargs)
            transformer = FluxTransformer2DModel.from_single_file(file_list[0], **call_kwargs)
            if delete_internet_files and os.path.exists(file_list[0]):
                os.remove(file_list[0])
        else:
            transformer = FluxTransformer2DModel.from_pretrained(uri, subfolder="transformer", **call_kwargs)
    else:
        if uri.endswith(".safetensors"):
            transformer = FluxTransformer2DModel.from_single_file(uri, **call_kwargs)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(uri, subfolder="transformer", **call_kwargs)

    return transformer


def load_t5_tokenizer(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> T5Tokenizer:
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN

    target_dir = directory or "./t5_tokenizer"
    file_list: List[str] = []
    if not use_local_files:
        file_list = [download_file(url, directory=target_dir, **dl_kwargs) for url in t5_tokenizer_url_list]

    tokenizer = T5Tokenizer.from_pretrained(target_dir, **call_kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return tokenizer


def load_t5_encoder(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> T5EncoderModel:
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("torch_dtype") is None and torch is not None:
        call_kwargs["torch_dtype"] = torch.float16
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN

    target_dir = directory or "./t5_encoder"
    file_list: List[str] = []
    if not use_local_files:
        os.makedirs(target_dir, exist_ok=True)
        futures = [
            naive_async(download_file)(url, directory=target_dir, mute=(i != 0), **dl_kwargs)
            for i, url in enumerate(t5_encoder_url_list)
        ]
        file_list = [fut.result() for fut in futures]

    encoder = T5EncoderModel.from_pretrained(target_dir, **call_kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return encoder


def load_clip_tokenizer(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> CLIPTokenizer:
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN

    target_dir = directory or "./clip_tokenizer"
    file_list: List[str] = []
    if not use_local_files:
        file_list = [download_file(url, directory=target_dir, **dl_kwargs) for url in clip_tokenizer_url_list]

    tokenizer = CLIPTokenizer.from_pretrained(target_dir, **call_kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return tokenizer


def load_clip_encoder(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> CLIPTextModel:
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("torch_dtype") is None and torch is not None:
        call_kwargs["torch_dtype"] = torch.float16
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN

    target_dir = directory or "./clip_encoder"
    file_list: List[str] = []
    if not use_local_files:
        file_list = [download_file(url, directory=target_dir, **dl_kwargs) for url in clip_encoder_url_list]

    encoder = CLIPTextModel.from_pretrained(target_dir, **call_kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return encoder


def load_flux_vae(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> AutoencoderKL:
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("torch_dtype") is None and torch is not None:
        call_kwargs["torch_dtype"] = torch.float16
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN

    target_dir = directory or "./vae"
    file_list: List[str] = []
    if not use_local_files:
        file_list = get_flux_vae_files(target_dir, **dl_kwargs)

    vae = AutoencoderKL.from_pretrained(target_dir, **call_kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return vae


def load_flux_scheduler(
    directory: Optional[str] = None,
    use_local_files: bool = False,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> FlowMatchEulerDiscreteScheduler:
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    if call_kwargs.get("torch_dtype") is None and torch is not None:
        call_kwargs["torch_dtype"] = torch.float16
    if call_kwargs.get("token") is None:
        call_kwargs["token"] = config.HF_HUB_TOKEN

    target_dir = directory or "./scheduler"
    file_list: List[str] = []
    if not use_local_files:
        file_list = get_flux_scheduler_files(target_dir, **dl_kwargs)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(target_dir, **call_kwargs)
    if not use_local_files and delete_internet_files:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
    return scheduler


def load_flux_pipeline_by_components(
    uri: Optional[str] = None,
    delete_internet_files: bool = False,
    download_kwargs: Optional[Dict[str, str]] = None,
    **kwargs: LoaderKwargValue
) -> FluxPipeline:
    target_uri = uri or default_flux_transformer_url
    dl_kwargs = download_kwargs or {}
    call_kwargs: Dict[str, LoaderKwargValue] = dict(kwargs)
    quant_cfg = call_kwargs.pop("quantization_config", None)
    if quant_cfg is None:
        compute_dtype = torch.float16 if torch is not None else None
        quant_dict = {"load_in_4bit": True, "bnb_4bit_compute_dtype": compute_dtype}
    elif isinstance(quant_cfg, dict):
        quant_dict = quant_cfg
    else:
        quant_dict = {}

    transformer = load_flux_transformer(
        "/transformer",
        uri=target_uri,
        download_kwargs=dl_kwargs,
        delete_internet_files=delete_internet_files,
        quantization_config=DiffusersBitsAndBytesConfig(**quant_dict) if quant_dict else None,
        **call_kwargs
    )
    t5_tok = load_t5_tokenizer(download_kwargs=dl_kwargs, **call_kwargs)
    t5_enc = load_t5_encoder(
        download_kwargs=dl_kwargs,
        quantization_config=TransformersBitsAndBytesConfig(**quant_dict) if quant_dict else None,
        **call_kwargs
    )
    clip_tok = load_clip_tokenizer(download_kwargs=dl_kwargs, delete_internet_files=delete_internet_files, **call_kwargs)
    clip_enc = load_clip_encoder(download_kwargs=dl_kwargs, delete_internet_files=delete_internet_files, **call_kwargs)
    vae = load_flux_vae(download_kwargs=dl_kwargs, delete_internet_files=delete_internet_files, **call_kwargs)
    sched = load_flux_scheduler(download_kwargs=dl_kwargs, delete_internet_files=delete_internet_files, **call_kwargs)

    return FluxPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=sched,
        text_encoder=clip_enc,
        text_encoder_2=t5_enc,
        tokenizer=clip_tok,
        tokenizer_2=t5_tok,
    )
