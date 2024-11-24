# 0
#!mkdir ./whl
#!curl -L -o ./whl/co_diffusers-0.1.0-py3-none-any.whl "https://github.com/CyberVy/co-diffusers/raw/refs/heads/main/co-diffusers/dist/co_diffusers-0.1.0-py3-none-any.whl"
#!pip install ./whl/co_diffusers-0.1.0-py3-none-any.whl

# 1
from coffusers.enhancement import get_enhancer
from coffusers.download import download_file
from coffusers.message import send_PIL_photo
from coffusers.const import *
from random import randint
import torch,threading,sys,gc


model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = get_enhancer(model,download_kwargs={"headers":{"cookie":cookie}}).to("cuda")

# 2
def generate(prompt,negative_prompt,num,seed=None,**kwargs):
    kwargs.update(prompt=prompt,negative_prompt=negative_prompt)
    seeds = []
    if seed:
        seeds.append(seed)
    else:
        for i in range(num):
            seeds.append(randint(-2 ** 63, 2 ** 64 - 1))
    for item in seeds:
        kwargs.update(generator=torch.manual_seed(item))
        image = pipeline(**kwargs).images[0]
        caption = f"Prompt: {prompt}\n\nNegative Prompt: {negative_prompt}\n\nStep: {kwargs.get('num_inference_steps')},CFG: {kwargs.get('guidance_scale')},CLIP Skip: {kwargs.get('clip_skip')}\nSampler: {pipeline.scheduler.config._class_name}\nLoRa: {pipeline.lora_dict}\nSeed: {item}\n\nModel:{pipeline.model_name}"
        threading.Thread(target=lambda: send_PIL_photo(image,file_name="Colab.PNG",file_type="PNG",caption=caption)).start()
        torch.cuda.empty_cache();gc.collect()


prompt = "young white woman with dramatic makeup resembling a melted clown, deep black smokey eyes, smeared red lipstick, and white face paint streaks, wet hair falling over shoulders, dark and intense aesthetic, fashion editorial style, aged around 20 years, inspired by rick genest's zombie boy look, best quality"
negative_prompt = "bad hands, malformed limbs, malformed fingers, bad anatomy, fat fingers, ugly, unreal, cgi, airbrushed, watermark, low resolution"
num = 5

num_inference_steps = 30
guidance_scale = 1.5
clip_skip = 1

seed = 13743883683399229202

width = None
height = None

pipeline.set_lora("https://civitai.com/api/download/models/997426?type=Model&format=SafeTensor","hand",0.2)


generate(prompt=prompt,negative_prompt=negative_prompt,num=num,seed=seed,width=width,height=height,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,clip_skip=clip_skip)
