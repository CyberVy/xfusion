# Hello Xfusion!
Xfusion is a Python library built on top of [Diffusers](https://github.com/huggingface/diffusers). It specializes in assembling open-source AI models, with a particular focus on models like Stable Diffusion, Flux, and others.

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
from xfusion.enhancement import SDPipelineEnhancer
pipeline = SDPipelineEnhancer(None)
server = pipeline.load_ui(globals(),debug=True,inline=False)
```
**Use UI with multiple GPUs/single GPU is also supported**

```python
from xfusion import SDPipelineEnhancer
from xfusion import load_stable_diffusion_ui_for_multiple_pipelines
from xfusion.const import GPU_Count

pipelines = [SDPipelineEnhancer(None) for i in range(GPU_Count)]
server = load_stable_diffusion_ui_for_multiple_pipelines(pipelines,_globals=globals())
server.launch(debug=True,inline=False,quiet=True)
```
Or
```python
from xfusion import load_enhancer
from xfusion import load_stable_diffusion_ui_for_multiple_pipelines
from xfusion.const import GPU_Count

pipelines = [load_enhancer(None,download_kwargs={"directory":"./"}) for i in range(GPU_Count)]
server = load_stable_diffusion_ui_for_multiple_pipelines(pipelines,_globals=globals())
server.launch(debug=True,inline=False,quiet=True)
```
---
**Use a pipeline with a single GPU in UI**
```python
from xfusion.enhancement import load_enhancer

model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = load_enhancer(model).to("cuda")
server = pipeline.load_ui(globals(),debug=True,inline=False)
```

**Use pipelines with multiple GPUs in UI**
```python
from xfusion.enhancement import load_enhancer
from xfusion.ui import load_stable_diffusion_ui_for_multiple_pipelines

model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = load_enhancer(model,download_kwargs={"directory":"/xfusion"}).to("cuda:0")
_pipeline = load_enhancer(model,download_kwargs={"directory":"/xfusion"}).to("cuda:1")
server = load_stable_diffusion_ui_for_multiple_pipelines([pipeline,_pipeline],_globals=globals())
server.launch(debug=True,inline=False,quiet=True)
```
---
**Directly use a pipeline in the backend**
```python
from xfusion.enhancement import load_enhancer
import torch

model = "https://civitai.com/api/download/models/646523?type=Model&format=SafeTensor&size=pruned&fp=fp16"
pipeline = load_enhancer(model,model_version="xl").to("cuda")

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
---
# Acknowledgments
Xfusion leverages the Diffusers library and is inspired by the incredible work of the open-source community.
