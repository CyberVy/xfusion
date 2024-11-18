# 0
#!mkdir ./whl
#!curl -L -o ./whl/co_diffusers-0.1.0-py3-none-any.whl "https://github.com/CyberVy/co-diffusers/raw/refs/heads/main/co-diffusers/dist/co_diffusers-0.1.0-py3-none-any.whl"
#!pip install ./whl/co_diffusers-0.1.0-py3-none-any.whl

# 1
from coffusers.components import get_pipeline,get_vae,set_lora
from coffusers.enhancement import get_embeds_from_pipeline
from coffusers.download import download_file
from coffusers.message import send_PIL_photo
from random import randint
import torch,threading,sys


civitai_cookie = "__Secure-civitai-token=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..o9E0O2vKpNnCWtB6.EOfQnHJr-gRewFTTErAboA38UymwyuDED35bicH12oBnFrFQ4_EMOZluxjMKi0g3Lm-mzOzzJbw0tBigTY8iAr6OGl81KPAAHIGtU-Y98eXT3P52cHFuo3Y-9FvwswxtsHCGtRm64MmTxGJKUivPBITpg7y89dzJ0YNp7OiPo8PN2qTVpHl2Jrl5Td591Z3ikMNIGHTEky8N_rX6QoGhVbDCrUW-MqcjUlE5enEMggUyYyCQxCYllMGUW4P1T4Ev5uXFRQWP221AHZI5gtOWQQMmaI98gYSL6x-q_ZnToa0UK-KQYIR_xIQFjvBrMk67ZFJLticbBs65q055teu2FbklMd38rrYe7NgHycrG2yaeQInIW20NBKaM0XGqu2IWCZAJpoUFR6ZKuwo-G_UDDFJJ7l09Rgk3Ww6_1g1KB88PpT_l1oN5OGyny_REG_eqWmNgeLO-vSP_ogwDiXxqgNCmlMfyXruK3lYO4ro1glHsgH2PwMUBQ2Rpjtao5rO7dzHdXOw5lVRbQvOk4ChuhmIQSXJRtQE8s55DNX6yQEPl0pLZsRewuQ-n5qJHOOJ4MB92q6i-6WukEoZich9qb1iwC4YhIOjBp0y55Ar51oUabQ9eNTruFxHFrflxEIjATNbq4WQMZfZi0Co4qM3Y-oEw6r_Xt56MYOUF10MjNazwfmoVr2VrXyLyWVhfQnJ7bebui4Qi21gZhwT6ryPxCdHOmghBonfbRKTpygYP5MuJ-uDNRC-477RRNure02SvC13aJa6OPJW55t-ABIz2LiUwSAqnuLGJSha6W3atFz2Ns8N61tzzba4jVUCErcZVYWjiN5MsVY9JU2VFnIeXRuV7BpEM2IKd7P_9lGhqcDHNiY0bON0IQ5K_xicdRUa_GEtOrgc-KQoErSfVi0YPArHGbw4FkHL1r_Ts_2LtNy35R9bJPb-gKEy6UOMQrM87_8ADJt31er-R4908LGjeFr2ZovgkL6lEQ7CFVL5QJLAmwGfACjN4u-gRwZbl_YbpsVCRfdPY0t4ZnZGupSH2fL64Tv7dumwjG92H9pZHtx0iteQxQbhsl7h_R_h5dgAo94bOEb3IyjUBqcP5jJ8ascE9SbYaTN-cQXhC-NOJzGHGfEhFYtKXXoCynOIUNkVNKDkGyf5knsLu5oW7QW9QjRafdaf-FblZLLGUyHELCSBAsm0VU3niTDB4S8KE96ou-B2jFHece5bOhVFrUtRyS7cjVh-gz7cfySBrbAciYTi-uMs.oiTi5nqQCd3bwiaeuzCGCA;"
hf_cookie = "token=tNdvqKWNwfXlIhemmKYNbDXDgXRshHvwXOFUzXWWVZjkNbWRfszhynxgDXSmfjZDZZuBwywspRCUZURCZBgTuchTYyqxmayZtumKiNfXtxajLQVZGKgvLSDIsWIMIbbq"
cookie = civitai_cookie + hf_cookie

model = "https://civitai.com/api/download/models/798204?type=Model&format=SafeTensor&size=full&fp=fp16"
pipeline = get_pipeline(model,download_kwargs={"headers":{"cookie":cookie}}).to("cuda")

lora ="https://civitai.com/api/download/models/997426?type=Model&format=SafeTensor"
set_lora(pipeline,lora,lora_name="hands",scale=0.4,download_kwargs={"headers":{"cookie":cookie},"directory":"lora"})

# 2
def generate(prompt,negative_prompt,num,seed=None,**kwargs):
    kwargs.update(get_embeds_from_pipe(pipeline,prompt,negative_prompt))
    seeds = []
    if seed:
        seeds.append(seed)
    else:
        for i in range(num):
            seeds.append(randint(-2 ** 63, 2 ** 64 - 1))
    for item in seeds:
        kwargs["generator"] = torch.manual_seed(item)
        image = pipeline(**kwargs).images[0]
        torch.cuda.empty_cache()
        threading.Thread(target=lambda: send_PIL_photo(image,file_name="Colab.PNG",file_type="PNG",caption=f"Prompt:{prompt}\n\nNegative Prompt:{negative_prompt}\n\nStep:{num_inference_steps},CFG:{guidance_scale},CLIP Skip:{clip_skip}\nSampler:{pipeline.scheduler.config._class_name}\nSeed:{item}")).start()


prompt = "detailed portrait Photo of a beautiful 20yo woman who is an instgram influencer, detailed rich background by Slim Aarons.She is holding a rose in her hands."
negative_prompt = "malformed limbs, malformed fingers, bad anatomy, fat fingers, ugly, unreal, cgi, airbrushed, watermark, low resolution"
num = 5

num_inference_steps = 30
guidance_scale = 1.5
clip_skip = 1
seed = -4100840620450523389

width = None
height = None

pipeline.set_adapters(["hands"], [0.4])

generate(prompt,negative_prompt,num=num,seed=seed,width=width,height=height,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,clip_skip=clip_skip)
