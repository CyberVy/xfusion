import os
import torch

GPU_COUNT = torch.cuda.device_count()
GPU_NAME = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]

XFUSION_COOKIE = os.environ.get("XFUSION_COOKIE")
XFUSION_PROXY = os.environ.get("XFUSION_PROXY")

HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN") or "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"
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN") or "1dab3490177fd9b4985323655f917d0c"

PROXY_URL_PREFIX = os.environ.get("PROXY_URL_PREFIX") or "https://us2.xsolutiontech.com"


def get_origin():
    import re
    import requests
    country = None
    try:
        text = requests.get("http://104.16.0.0/cdn-cgi/trace",timeout=5).text
        search = re.search("loc=(.*)",text)
        if search:
            country = search[1]
    except IOError:
        pass
    return country

LOCATION = get_origin()
NEED_PROXY_LOCATION_LIST = ["CN",None]
# NEED_PROXY = LOCATION in NEED_PROXY_LOCATION_LIST
NEED_PROXY = True
