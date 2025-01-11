import os
import torch

GPU_Count = torch.cuda.device_count()
GPU_Name = [torch.cuda.get_device_name(i) for i in range(GPU_Count)]


XFUSION_COOKIE = os.environ.get("XFUSION_COOKIE")
XFUSION_PROXY = os.environ.get("XFUSION_PROXY")

HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN") or "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"
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN") or "1dab3490177fd9b4985323655f917d0c"
