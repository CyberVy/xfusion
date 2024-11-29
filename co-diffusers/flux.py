import os
import torch
from transformers import T5EncoderModel,T5Tokenizer,CLIPTextModel,CLIPTokenizer
from diffusers import AutoencoderKL,FluxPipeline
from coffusers.download import download_file
from coffusers.const import hf_token
from optimum.quanto import freeze,qfloat8,quantize


def load_clip_tokenizer():
    _1 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/merges.txt?download=true",
        directory="./clip_tokenizer")
    _2 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/special_tokens_map.json?download=true",
        directory="./clip_tokenizer")
    _3 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/tokenizer_config.json?download=true",
        directory="./clip_tokenizer")
    _4 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer/vocab.json?download=true",
        directory="./clip_tokenizer")
    clip_tokenizer = CLIPTokenizer.from_pretrained("./clip_tokenizer")
    print("CLIP Tokenizer ready.")
    os.remove(_1);os.remove(_2);os.remove(_3);os.remove(_4)
    return clip_tokenizer

def load_clip_encoder():
    _1 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/config.json?download=true",
        directory="./clip_encoder")
    _2 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/model.safetensors?download=true",
        directory="./clip_encoder")
    clip_encoder = CLIPTextModel.from_pretrained("./clip_encoder")
    print("CLIP Encoder ready.")
    os.remove(_1);os.remove(_2)
    return clip_encoder

def load_t5_encoder():
    _1 = download_file(
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model.safetensors.index.json?download=true",
        directory="./t5_encoder")
    _2 = download_file(
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/config.json?download=true",
        directory="./t5_encoder")
    _3 = download_file(
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model-00001-of-00002.safetensors?download=true",
        directory="./t5_encoder")
    _4 = download_file(
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/model-00002-of-00002.safetensors?download=true",
        directory="./t5_encoder")
    t5_encoder = T5EncoderModel.from_pretrained("./t5_encoder")
    print("T5 Encoder ready.")
    os.remove(_1);os.remove(_2);os.remove(_3); os.remove(_4)
    return t5_encoder

def load_t5_tokenizer():
    _1 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/special_tokens_map.json?download=true",
            directory="./t5_tokenizer")
    _2 =download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/spiece.model?download=true",
            directory="./t5_tokenizer")
    _3 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/tokenizer.json?download=true",
            directory="./t5_tokenizer")
    _4 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/tokenizer_2/tokenizer_config.json?download=true",
            directory="./t5_tokenizer")
    t5_tokenizer = T5Tokenizer.from_pretrained("./t5_tokenizer")
    print("T5 Tokenizer ready.")
    os.remove(_1);os.remove(_2);os.remove(_3);os.remove(_4)
    return t5_tokenizer

def load_vae():
    _1 =  download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true",
            directory="./vae")
    _2 = download_file("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/config.json?download=true",
            directory="./vae")
    vae = AutoencoderKL.from_pretrained("./vae", torch_dtype=torch.bfloat16)
    print("VAE ready")
    os.remove(_1);os.remove(_2)
    return vae

def load_flux(uri="https://civitai.com/api/download/models/979329?type=Model&format=SafeTensor&size=full&fp=fp16"):
    _1 = download_file(uri)
    pipeline = FluxPipeline.from_single_file(_1,vae=None,text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,token=hf_token,
            torch_dtype=torch.bfloat16)
    print("FLUX Pipeline ready.")
    os.remove(_1)
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



pipeline = load_flux()
pipeline.to("cuda")

clip_tokenizer,clip_encoder = load_clip_tokenizer(),load_clip_encoder()
t5_tokenizer,t5_encoder = load_t5_tokenizer(),load_t5_encoder()
vae = load_vae()
vae.to("cuda")
pipeline.vae = vae
print("All ready.")

prompt_embeds = get_t5_prompt_embeds(t5_tokenizer,t5_encoder,'hello').to(torch.float16).to("cuda")
pooled_prompt_embeds = get_clip_prompt_embeds(clip_tokenizer,clip_encoder,'hello').to(torch.float16).to("cuda")
print("Embeds ready.")

image = pipeline(prompt_embeds=prompt_embeds,pooled_prompt_embeds=pooled_prompt_embeds,num_inference_steps=20)
