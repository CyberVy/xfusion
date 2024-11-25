# an extra module for users

from random import randint
import torch,threading,gc


def text_to_image_and_send_to_telegram(pipeline,prompt,negative_prompt,num,seed=None,**kwargs):
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
        caption = f"Prompt: {prompt[:384]}\n\nNegative Prompt: {negative_prompt[:384]}\n\nStep: {kwargs.get('num_inference_steps')}, CFG: {kwargs.get('guidance_scale')}, CLIP Skip: {kwargs.get('clip_skip')}\nSampler: {pipeline.scheduler.config._class_name}\nLoRa: {pipeline.lora_dict}\nSeed: {item}\n\nModel:{pipeline.model_name}"
        threading.Thread(target=lambda: pipeline.send_PIL_photo(image,file_name="Colab.PNG",file_type="PNG",caption=caption)).start()
        torch.cuda.empty_cache();gc.collect()

