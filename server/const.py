import os
try:
    import torch
    GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
    GPU_NAME = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
except Exception:
    GPU_COUNT = 0
    GPU_NAME = []

XFUSION_COOKIE = os.environ.get("XFUSION_COOKIE")
XFUSION_PROXY = os.environ.get("XFUSION_PROXY")

HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN")
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN")

PROXY_URL_PREFIX = os.environ.get("PROXY_URL_PREFIX") or "https://us2.xsolutiontech.com"


def get_origin():
    import re
    import requests
    country = None
    try:
        text = requests.get("https://cloudflare.com/cdn-cgi/trace", timeout=1.5).text
        search = re.search("loc=(.*)", text)
        if search:
            country = search[1]
    except Exception:
        pass
    return country

LOCATION = get_origin()
NEED_PROXY_LOCATION_LIST = ["CN",None]
NEED_PROXY = LOCATION in NEED_PROXY_LOCATION_LIST
