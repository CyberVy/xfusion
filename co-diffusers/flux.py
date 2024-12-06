import os
import torch
from transformers import T5EncoderModel,T5Tokenizer,CLIPTextModel,CLIPTokenizer
from diffusers import AutoencoderKL,FluxPipeline,FluxTransformer2DModel,FlowMatchEulerDiscreteScheduler
from coffusers.download import download_file
from coffusers.const import hf_token
from coffusers.utils import threads_execute
from optimum.quanto import freeze,qfloat8,quantize


def get_flux_transformer_files(directory,
    uri="https://civitai.com/api/download/models/979329?type=Model&format=SafeTensor&size=full&fp=fp16",
    **kwargs):
    file_list = [download_file(uri, directory=directory,**kwargs)]
    return file_list

def get_t5_tokenizer_files(directory,**kwargs):
    url_list = [
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/special_tokens_map.json?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/spiece.model?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/tokenizer.json?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/tokenizer_config.json?download=true",
    ]
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def get_t5_encoder_files(directory,**kwargs):
    url_list = [
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model.safetensors.index.json?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/config.json?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model-00001-of-00002.safetensors?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model-00002-of-00002.safetensors?download=true"
    ]
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def get_clip_tokenizer_files(directory,**kwargs):
    url_list = [
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/merges.txt?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/special_tokens_map.json?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/tokenizer_config.json?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/vocab.json?download=true"
    ]
    file_list = []
    for url in url_list:
        file_list.append(download_file(url,directory=directory,**kwargs))
    return file_list

def get_clip_encoder_files(directory,**kwargs):
    url_list = [
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/config.json?download=true",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/model.safetensors?download=true"
    ]
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def get_vae_files(directory,**kwargs):
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

def load_t5_tokenizer(directory=None, use_local_files=False, delete_internet_files=True):
    directory = "./t5_tokenizer" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_t5_tokenizer_files(directory)

    t5_tokenizer = T5Tokenizer.from_pretrained(directory)
    print("T5 Tokenizer ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return t5_tokenizer

def load_t5_encoder(directory=None, use_local_files=False, delete_internet_files=True):
    directory = "./t5_encoder" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_t5_encoder_files(directory)

    t5_encoder = T5EncoderModel.from_pretrained(directory, torch_dtype=torch.float16)
    print("T5 Encoder ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return t5_encoder

def load_clip_tokenizer(directory=None,use_local_files=False,delete_internet_files=True):
    directory = "./clip_tokenizer" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_clip_tokenizer_files(directory)

    clip_tokenizer = CLIPTokenizer.from_pretrained(directory)
    print("CLIP Tokenizer ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)

    return clip_tokenizer

def load_clip_encoder(directory=None,use_local_files=False,delete_internet_files=True):
    directory = "./clip_encoder" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_clip_encoder_files(directory)

    clip_encoder = CLIPTextModel.from_pretrained(directory,torch_dtype=torch.float16)
    print("CLIP Encoder ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return clip_encoder

def load_flux_vae(directory=None,use_local_files=False,delete_internet_files=True):
    directory = "./vae" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_vae_files(directory)

    vae = AutoencoderKL.from_pretrained(directory, torch_dtype=torch.float16)
    print("VAE ready")

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

def load_flux(uri="https://civitai.com/api/download/models/979329?type=Model&format=SafeTensor&size=full&fp=fp16"):
    def f(model):
        quantize(model,weights=qfloat8)
        freeze(model)

    transformer = load_flux_transformer(uri=uri)
    threads_execute(f,(transformer,),_await=False)

    t5_tokenizer, t5_encoder = load_t5_tokenizer(), load_t5_encoder()
    _thread_t5 = threads_execute(f,(t5_encoder,),_await=False)[0]

    clip_tokenizer, clip_encoder = load_clip_tokenizer(), load_clip_encoder()
    vae = load_flux_vae()
    scheduler = load_flux_scheduler()
    _thread_t5.join()
    pipeline = FluxPipeline(transformer=transformer, vae=vae, scheduler=scheduler,
                            text_encoder=clip_encoder, text_encoder_2=t5_encoder,
                            tokenizer=clip_tokenizer, tokenizer_2=t5_tokenizer)
    pipeline.enable_model_cpu_offload()
    print("FLUX Pipeline ready.")
    return pipeline

def get_clip_prompt_embeds(tokenizer,text_encoder,prompt,num_images_per_prompt: int = 1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(prompt,padding="max_length",max_length=tokenizer.model_max_length,
        truncation=True,return_overflowing_tokens=False,return_length=False,return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)
        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds

def get_t5_prompt_embeds(tokenizer_2, text_encoder_2,prompt = None,num_images_per_prompt: int = 1,max_sequence_length: int = 512):

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer_2(prompt,padding="max_length",max_length=max_sequence_length,
        truncation=True,return_length=False, return_overflowing_tokens=False,return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer_2.batch_decode(untruncated_ids[:, tokenizer_2.model_max_length - 1 : -1])
        print(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    with torch.no_grad():
        prompt_embeds = text_encoder_2(text_input_ids, output_hidden_states=False)[0]
        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

