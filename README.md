# Hello Xfusion!
Xfusion is a Python library built on top of [Diffusers](https://github.com/huggingface/diffusers). It specializes in assembling open-source AI models, with a particular focus on models like Stable Diffusion, Flux, and others.

# Features
- Enhanced support for models from sources outside Hugging Face.
- Simple and efficient model loading using just a model URL.
- Extension for models.

# Supported Models
- Stable Diffusion 1.5/2/3/3.5/XL
- Flux

# Installation
```bash
pip install -q git+https://github.com/CyberVy/xfusion.git
```
# Code Example
**Use with UI**
```python
from xfusion.enhancement import load_enhancer
import torch

model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = load_enhancer(model).to("cuda")
server = pipeline.load_ui(globals(),debug=True,inline=False)

```
**UI with multiple GPUs**
```python
from xfusion.enhancement import load_enhancer
from xfusion.ui import load_stable_diffusion_ui_for_multiple_pipelines
import torch

model = "https://civitai.com/api/download/models/577919?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = load_enhancer(model,download_kwargs={"directory":"/xfusion"}).to("cuda:0")
_pipeline = load_enhancer(model,download_kwargs={"directory":"/xfusion"}).to("cuda:1")
server = load_stable_diffusion_ui_for_multiple_pipelines([pipeline,_pipeline],_globals=globals())
server.launch(debug=True,inline=False,quiet=True)
```
**Use with the backend pipeline**
```python
from xfusion.enhancement import load_enhancer
import torch

model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = load_enhancer(model,"xl").to("cuda")

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

# Acknowledgments
Xfusion leverages the Diffusers library and is inspired by the incredible work of the open-source community.
