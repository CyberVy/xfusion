import os
import torch
from .component_const import t5_tokenizer_url_list,t5_encoder_url_list
from .component_const import clip_tokenizer_url_list,clip_encoder_url_list
from ..download import download_file
from ..const import hf_token
from transformers import T5EncoderModel,T5Tokenizer,CLIPTextModel,CLIPTokenizer
from diffusers import AutoencoderKL


def get_t5_tokenizer_files(directory,**kwargs):
    url_list = t5_tokenizer_url_list
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def get_t5_encoder_files(directory,**kwargs):
    url_list = t5_encoder_url_list
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def load_t5_tokenizer(directory=None, use_local_files=False, delete_internet_files=True,download_kwargs=None,**kwargs):
    """
    :param directory:
    :param use_local_files:
    :param delete_internet_files:
    :param download_kwargs:
    :param kwargs: passed in T5Tokenizer.from_pretrained
    :return:
    """
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("token") is None:
        kwargs.update(token=hf_token)

    directory = "./t5_tokenizer" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_t5_tokenizer_files(directory,**download_kwargs)

    t5_tokenizer = T5Tokenizer.from_pretrained(directory,**kwargs)
    print("T5 Tokenizer ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return t5_tokenizer

def load_t5_encoder(directory=None, use_local_files=False, delete_internet_files=True,download_kwargs=None,**kwargs):
    """
    :param directory:
    :param use_local_files:
    :param delete_internet_files:
    :param download_kwargs:
    :param kwargs: passed in T5EncoderModel.from_pretrained
    :return:
    """
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=hf_token)

    directory = "./t5_encoder" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_t5_encoder_files(directory,**download_kwargs)

    t5_encoder = T5EncoderModel.from_pretrained(directory,**kwargs)
    print("T5 Encoder ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return t5_encoder

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

def get_clip_tokenizer_files(directory,**kwargs):
    url_list = clip_tokenizer_url_list
    file_list = []
    for url in url_list:
        file_list.append(download_file(url,directory=directory,**kwargs))
    return file_list

def get_clip_encoder_files(directory,**kwargs):
    url_list = clip_encoder_url_list
    file_list = []
    for url in url_list:
        file_list.append(download_file(url, directory=directory,**kwargs))
    return file_list

def load_clip_tokenizer(directory=None,use_local_files=False,delete_internet_files=True,download_kwargs=None,**kwargs):
    """
    :param directory:
    :param use_local_files:
    :param delete_internet_files:
    :param download_kwargs:
    :param kwargs: passed in CLIPTokenizer.from_pretrained
    :return:
    """
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("token") is None:
        kwargs.update(token=hf_token)

    directory = "./clip_tokenizer" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_clip_tokenizer_files(directory,**download_kwargs)

    clip_tokenizer = CLIPTokenizer.from_pretrained(directory,**kwargs)
    print("CLIP Tokenizer ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)

    return clip_tokenizer

def load_clip_encoder(directory=None,use_local_files=False,delete_internet_files=True,download_kwargs=None,**kwargs):
    download_kwargs = {} if download_kwargs is None else download_kwargs
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if kwargs.get("token") is None:
        kwargs.update(token=hf_token)

    directory = "./clip_encoder" if directory is None else directory
    file_list = []
    if not use_local_files:
        file_list = get_clip_encoder_files(directory,**download_kwargs)

    clip_encoder = CLIPTextModel.from_pretrained(directory,**kwargs)
    print("CLIP Encoder ready.")

    if not use_local_files and delete_internet_files:
        for file in file_list:
            os.remove(file)
    return clip_encoder

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

def get_clip_from_pipeline(pipeline):

    tokenizers = []
    text_encoders = []
    tokenizer_names = ["tokenizer", "tokenizer_2", "tokenizer_3"]
    text_encoder_names = ["text_encoder", "text_encoder_2", "text_encoder_3"]
    for i, item in enumerate(tokenizer_names):
        if hasattr(pipeline, item):
            tokenizers.append(getattr(pipeline, item))
            text_encoders.append(getattr(pipeline, text_encoder_names[i]))
    return tokenizers,text_encoders

def load_vae(vae_uri,directory=None,download_kwargs=None,**kwargs):

    use_internet = True
    if download_kwargs is None:
        download_kwargs = {}

    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})

    if vae_uri.startswith(".") or vae_uri.startswith("/") or vae_uri.startswith("~"):
        use_internet = False

    if use_internet:
        file_path = download_file(vae_uri,directory=directory,**download_kwargs)
        return AutoencoderKL.from_single_file(file_path,**kwargs)
    else:
        return AutoencoderKL.from_single_file(vae_uri,**kwargs)
