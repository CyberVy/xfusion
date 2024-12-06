from .const import cookie
import requests
from tqdm import tqdm
from urllib.parse import urlparse
import os
import re


def download_file(url,filename=None,directory=None,**kwargs):

    directory = "./" if directory is None else directory

    headers = {}
    if kwargs.get("headers") is None:
        headers.update(cookie=cookie)
        headers.update({"user-agent":"coffusers"})
        kwargs.update(headers=headers)

    if not directory.endswith("/"):
        directory += "/"

    if not os.path.exists(directory):
        os.mkdir(directory)

    response = requests.get(url,stream=True,**kwargs)
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

    with open(f"{directory}unfinished.{filename}", 'wb') as file, tqdm(desc=filename,total=total_size, unit='B',unit_scale=True,unit_divisor=1024,) as progress_bar:
        for data in response.iter_content(8192 * 2):
            file.write(data)
            progress_bar.update(len(data))
    os.rename(f"{directory}unfinished.{filename}",f"{directory}{filename}")
    return f"{directory}{filename}"


class DownloadArgumentsMixin:
    overrides = ["download_kwargs","set_download_kwargs"]

    def __init__(self):
        self.download_kwargs = {}

    def set_download_kwargs(self,**download_kwargs):
        self.download_kwargs.update(**download_kwargs)

