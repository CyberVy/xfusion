# 0
#!mkdir ./whl
#!curl -L -o ./whl/co_diffusers-0.1.0-py3-none-any.whl "https://github.com/CyberVy/co-diffusers/raw/refs/heads/main/co-diffusers/dist/co_diffusers-0.1.0-py3-none-any.whl"
#!pip install ./whl/co_diffusers-0.1.0-py3-none-any.whl

# 1
from coffusers.user_tools import text_to_image_and_send_to_telegram
from coffusers.enhancement import get_enhancer
from coffusers.download import download_file
from coffusers.message import send_PIL_photo
from coffusers.const import *
from random import randint
import torch,threading,sys,gc


model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = get_enhancer(model,download_kwargs={"headers":{"cookie":cookie}}).to("cuda")

# 2
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

text_to_image_and_send_to_telegram(pipeline,prompt=prompt,negative_prompt=negative_prompt,num=num,seed=seed,width=width,height=height,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,clip_skip=clip_skip)
