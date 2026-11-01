import os
import functools
from typing import List, Optional
import torch

_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)

GPU_COUNT: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
GPU_NAME: List[str] = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
if GPU_COUNT > 0:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

XFUSION_COOKIE: Optional[str] = os.environ.get("XFUSION_COOKIE")
XFUSION_PROXY: Optional[str] = os.environ.get("XFUSION_PROXY")

HF_HUB_TOKEN: Optional[str] = os.environ.get("HF_HUB_TOKEN")
CIVITAI_TOKEN: Optional[str] = os.environ.get("CIVITAI_TOKEN")

PROXY_URL_PREFIX: str = os.environ.get("PROXY_URL_PREFIX") or "https://us2.xsolutiontech.com"
DOWNLOAD_PATH: str = os.environ.get("DOWNLOAD_PATH", "/xfusion_models")
DELETE_AFTER_LOAD: bool = os.environ.get("DELETE_AFTER_LOAD", "false").lower() in ("true", "1", "yes")



@functools.lru_cache(maxsize=1)
def get_origin_country() -> Optional[str]:
    """Detect current geographic region for proxy routing (cached, non-blocking fallback)."""
    import re
    import requests
    try:
        text = requests.get("https://cloudflare.com/cdn-cgi/trace", timeout=1.5).text
        match = re.search(r"loc=(.*)", text)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=1)
def is_proxy_needed() -> bool:
    """Returns True if traffic should route through PROXY_URL_PREFIX (e.g. within mainland CN)."""
    explicit = os.environ.get("NEED_PROXY")
    if explicit is not None:
        return explicit.lower() in ("true", "1", "yes")
    country = get_origin_country()
    return country in ("CN", None)
