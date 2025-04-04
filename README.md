# Hello Xfusion!
Xfusion is a Python library specializing in assembling open-source AI models, with a particular focus on models like Stable Diffusion, Flux, and others.

# Features
- Enhanced support for models from sources outside Hugging Face.
- Simple and efficient model loading using just a model URL.
- Extension for models.
- Multiple GPUs support.

# Supported Models
- Stable Diffusion 1.5/2/3/3.5/XL
- Flux

# Installation
```bash
pip install -q git+https://github.com/CyberVy/xfusion.git
```
# Code Example
**Use UI with a single GPU**
```python
from xfusion.enhancement import load_enhancer

telegram_kwargs = {"token":"","chat_id":""}
download_kwargs = {"directory":"./xfusion_models"}
pipeline = load_enhancer(None,model_version="xl", download_kwargs=download_kwargs, telegram_kwargs=telegram_kwargs)
server = pipeline.load_ui(globals())
```
**Use UI with multiple GPUs/A single GPU is also supported**

```python
from xfusion import load_enhancer
from xfusion import load_stable_diffusion_ui
from xfusion.const import GPU_COUNT

telegram_kwargs = {"token":"","chat_id":""}
download_kwargs = {"directory":"./xfusion_models"}
pipelines = [load_enhancer(None, model_version="xl", download_kwargs=download_kwargs, telegram_kwargs=telegram_kwargs) for i in range(GPU_COUNT)]
server = load_stable_diffusion_ui(pipelines, _globals=globals())
```

---
**Directly use a pipeline in the backend**
<details>
<summary>Code</summary>

```python
from xfusion.enhancement import load_enhancer
import torch

telegram_kwargs = {"token":"","chat_id":""}
download_kwargs = {"directory":"./xfusion_models"}

model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = load_enhancer(model,model_version="xl", download_kwargs=download_kwargs, telegram_kwargs=telegram_kwargs).to("cuda")

prompt = """
young white woman with dramatic makeup resembling a melted clown, deep black smokey eyes, smeared red lipstick, and white face paint streaks, wet hair falling over shoulders, dark and intense aesthetic, fashion editorial style, aged around 20 years, inspired by rick genest's zombie boy look, best quality
"""
negative_prompt = """
bad hands, malformed limbs, malformed fingers, bad anatomy, fat fingers, ugly, unreal, cgi, airbrushed, watermark, low resolution
"""

num_inference_steps = 30
guidance_scale = 2
clip_skip = 0

seed = 13743883683399229202

width = None
height = None

images = pipeline(prompt=prompt,negative_prompt=negative_prompt,generator=torch.Generator(pipeline.device).manual_seed(seed),width=width,height=height,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,clip_skip=clip_skip).images
```
</details>

---
# Acknowledgments
Xfusion leverages the Diffusers library and is inspired by the incredible work of the open-source community.
