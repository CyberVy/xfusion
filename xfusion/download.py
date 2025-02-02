from .const import XFUSION_COOKIE,HF_HUB_TOKEN,CIVITAI_TOKEN
from .const import PROXY_URL_PREFIX,NEED_PROXY
import requests
from tqdm import tqdm
from urllib.parse import urlparse
import os
import re


def download_file(url,filename=None,directory=None,mute=False,**kwargs):

    directory = "./" if directory is None else directory
    url = urlparse(url)
    headers = kwargs.pop("headers",{})

    if headers.get("cookie") is None:
        headers.update(cookie=XFUSION_COOKIE)

    if url.hostname == "huggingface.co":
        if headers.get("authorization") is None:
            headers.update(authorization=f"Bearer {HF_HUB_TOKEN}")

        url = urlparse(f"{PROXY_URL_PREFIX}/{url.geturl()}") if NEED_PROXY and PROXY_URL_PREFIX else url

    elif url.hostname == "civitai.com":
        if headers.get("authorization") is None:
            headers.update(authorization=f"Bearer {CIVITAI_TOKEN}")

        url = urlparse(f"{PROXY_URL_PREFIX}/{url.geturl()}") if NEED_PROXY and PROXY_URL_PREFIX else url

    kwargs.update(headers=headers)

    if not directory.endswith("/"):
        directory += "/"

    if not os.path.exists(directory):
        os.mkdir(directory)

    response = requests.get(url.geturl(),stream=True,**kwargs)
    if filename is None:
        filename = urlparse(response.url).path.split("/")[-1]
        content_disposition = response.headers.get("content-disposition","")
        if content_disposition:
            match = re.match('.*filename="(.*)".*',content_disposition)
            if match:
                try:
                    filename = match.group(1).split("/")[-1]
                except IndexError:
                    ...

    if os.path.exists(f"{directory}{filename}"):
        print(f"{directory}{filename} exists.")
        return f"{directory}{filename}"

    total_size = int(response.headers.get('content-length', 0))

    if not mute:
        with open(f"{directory}unfinished.{filename}", 'wb') as file, tqdm(desc=filename,total=total_size, unit='B',unit_scale=True,unit_divisor=1024,) as progress_bar:
            for data in response.iter_content(8192 * 2):
                file.write(data)
                progress_bar.update(len(data))
    else:
        with open(f"{directory}unfinished.{filename}", 'wb') as file:
            for data in response.iter_content(8192 * 2):
                file.write(data)

    os.rename(f"{directory}unfinished.{filename}",f"{directory}{filename}")
    return f"{directory}{filename}"


class DownloadArgumentsMixin:
    overrides = ["download_kwargs","set_download_kwargs"]

    def __init__(self):
        self.download_kwargs = {}

    def set_download_kwargs(self,**download_kwargs):
        self.download_kwargs.update(**download_kwargs)
