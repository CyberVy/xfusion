# 1
from google.colab import drive
drive.mount("/content/drive")
import sys
sys.path.append('/content/drive/MyDrive/utils')
from coffusers.components import get_pipe,get_vae,set_lora
from coffusers.download import download_file
from coffusers.message import send_PIL_photo
from random import randint
import torch,threading


civitai_cookie = "__Secure-civitai-token=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..o9E0O2vKpNnCWtB6.EOfQnHJr-gRewFTTErAboA38UymwyuDED35bicH12oBnFrFQ4_EMOZluxjMKi0g3Lm-mzOzzJbw0tBigTY8iAr6OGl81KPAAHIGtU-Y98eXT3P52cHFuo3Y-9FvwswxtsHCGtRm64MmTxGJKUivPBITpg7y89dzJ0YNp7OiPo8PN2qTVpHl2Jrl5Td591Z3ikMNIGHTEky8N_rX6QoGhVbDCrUW-MqcjUlE5enEMggUyYyCQxCYllMGUW4P1T4Ev5uXFRQWP221AHZI5gtOWQQMmaI98gYSL6x-q_ZnToa0UK-KQYIR_xIQFjvBrMk67ZFJLticbBs65q055teu2FbklMd38rrYe7NgHycrG2yaeQInIW20NBKaM0XGqu2IWCZAJpoUFR6ZKuwo-G_UDDFJJ7l09Rgk3Ww6_1g1KB88PpT_l1oN5OGyny_REG_eqWmNgeLO-vSP_ogwDiXxqgNCmlMfyXruK3lYO4ro1glHsgH2PwMUBQ2Rpjtao5rO7dzHdXOw5lVRbQvOk4ChuhmIQSXJRtQE8s55DNX6yQEPl0pLZsRewuQ-n5qJHOOJ4MB92q6i-6WukEoZich9qb1iwC4YhIOjBp0y55Ar51oUabQ9eNTruFxHFrflxEIjATNbq4WQMZfZi0Co4qM3Y-oEw6r_Xt56MYOUF10MjNazwfmoVr2VrXyLyWVhfQnJ7bebui4Qi21gZhwT6ryPxCdHOmghBonfbRKTpygYP5MuJ-uDNRC-477RRNure02SvC13aJa6OPJW55t-ABIz2LiUwSAqnuLGJSha6W3atFz2Ns8N61tzzba4jVUCErcZVYWjiN5MsVY9JU2VFnIeXRuV7BpEM2IKd7P_9lGhqcDHNiY0bON0IQ5K_xicdRUa_GEtOrgc-KQoErSfVi0YPArHGbw4FkHL1r_Ts_2LtNy35R9bJPb-gKEy6UOMQrM87_8ADJt31er-R4908LGjeFr2ZovgkL6lEQ7CFVL5QJLAmwGfACjN4u-gRwZbl_YbpsVCRfdPY0t4ZnZGupSH2fL64Tv7dumwjG92H9pZHtx0iteQxQbhsl7h_R_h5dgAo94bOEb3IyjUBqcP5jJ8ascE9SbYaTN-cQXhC-NOJzGHGfEhFYtKXXoCynOIUNkVNKDkGyf5knsLu5oW7QW9QjRafdaf-FblZLLGUyHELCSBAsm0VU3niTDB4S8KE96ou-B2jFHece5bOhVFrUtRyS7cjVh-gz7cfySBrbAciYTi-uMs.oiTi5nqQCd3bwiaeuzCGCA;"
hf_cookie = "token=tNdvqKWNwfXlIhemmKYNbDXDgXRshHvwXOFUzXWWVZjkNbWRfszhynxgDXSmfjZDZZuBwywspRCUZURCZBgTuchTYyqxmayZtumKiNfXtxajLQVZGKgvLSDIsWIMIbbq"
cookie = civitai_cookie + hf_cookie

model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipe = get_pipe(model,download_kwargs={"headers":{"cookie":cookie}}).to("cuda")

lora ="https://civitai.com/api/download/models/997426?type=Model&format=SafeTensor"
set_lora(pipe,lora,lora_name="hands",scale=0.8,download_kwargs={"headers":{"cookie":cookie},"directory":"lora"})

# 2
prompt = "detailed portrait Photo of a beautiful 20yo woman who is an instgram influencer, detailed rich background by Slim Aarons.She is holding a rose in her hands."
negative_prompt = "(malformed limbs:1.5), (malformed fingers:1.5), (bad anatomy:1.6), fat fingers, ugly, unreal, cgi, airbrushed, watermark, low resolution"
num_inference_steps = 30
guidance_scale = 1.5
clip_skip = 1


height = None
width = None
num = 5

pipe.set_adapters(["hands"], adapter_weights=[0.7])
def generate(prompt,negative_prompt,height,width,num,seed=None,**kwargs):
    for i in range(num):
        if seed is None:
            _seed = randint(-2 ** 63, 2 ** 64 - 1)
        else:
            _seed = seed
        kwargs["generator"] = torch.manual_seed(_seed)
        image = pipe(prompt,height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,clip_skip=clip_skip,negative_prompt=negative_prompt,num_images_per_prompt=1,**kwargs).images[0]
        torch.cuda.empty_cache()
        threading.Thread(target=lambda: send_PIL_photo(image,file_name="Colab.PNG",file_type="PNG",caption=f"Prompt:{prompt}\n\nNegative Prompt:{negative_prompt}\n\nStep:{num_inference_steps},CFG:{guidance_scale},CLIP Skip:{clip_skip}\nSeed:{_seed}")).start()

generate(prompt,negative_prompt,height,width,num)
