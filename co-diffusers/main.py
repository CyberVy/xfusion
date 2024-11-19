# 0
#!mkdir ./whl
#!curl -L -o ./whl/co_diffusers-0.1.0-py3-none-any.whl "https://github.com/CyberVy/co-diffusers/raw/refs/heads/main/co-diffusers/dist/co_diffusers-0.1.0-py3-none-any.whl"
#!pip install ./whl/co_diffusers-0.1.0-py3-none-any.whl

# 1
from coffusers.components import get_pipeline,get_vae,set_lora
from coffusers.enhancement import get_enhancer,get_embeds_from_pipeline
from coffusers.download import download_file
from coffusers.message import send_PIL_photo
from coffusers.const import *
from random import randint
import torch,threading,sys

model = "https://civitai.com/api/download/models/798204?type=Model&format=SafeTensor&size=full&fp=fp16"
pipeline = get_enhancer(model,download_kwargs={"headers":{"cookie":cookie}}).to("cuda")

lora ="https://civitai.com/api/download/models/997426?type=Model&format=SafeTensor"
set_lora(pipeline,lora,lora_name="hands",scale=0.4,download_kwargs={"headers":{"cookie":cookie},"directory":"lora"})

# 2
def generate(prompt,negative_prompt,num,seed=None,**kwargs):
    # kwargs.update(get_embeds_from_pipeline(pipeline,prompt,negative_prompt))
    kwargs.update({"prompt":prompt,"negative_prompt":negative_prompt})
    seeds = []
    if seed:
        seeds.append(seed)
    else:
        for i in range(num):
            seeds.append(randint(-2 ** 63, 2 ** 64 - 1))
    for item in seeds:
        kwargs.update(generator=torch.manual_seed(item))
        image = pipeline(**kwargs).images[0]
        torch.cuda.empty_cache()
        threading.Thread(target=lambda: send_PIL_photo(image,file_name="Colab.PNG",file_type="PNG",caption=f"Prompt:{prompt}\n\nNegative Prompt:{negative_prompt}\n\nStep:{num_inference_steps},CFG:{guidance_scale},CLIP Skip:{clip_skip}\nSampler:{pipeline.scheduler.config._class_name}\nSeed:{item}")).start()


prompt = "detailed portrait Photo of a beautiful 20yo woman who is an instgram influencer, detailed rich background by Slim Aarons.She is holding a rose in her hands."
negative_prompt = "malformed limbs, malformed fingers, bad anatomy, fat fingers, ugly, unreal, cgi, airbrushed, watermark, low resolution"
num = 5

num_inference_steps = 10
guidance_scale = 1
clip_skip = 1
seed = -4100840620450523389

width = None
height = None

pipeline.set_adapters(["hands"], [0.4])

generate(prompt,negative_prompt,num=num,seed=seed,width=width,height=height,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,clip_skip=clip_skip)
