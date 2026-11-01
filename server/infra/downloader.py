import os
import re
from urllib.parse import urlparse, unquote
from typing import Optional, Dict, Union
import requests
from tqdm import tqdm

from . import config


def download_file(
    url: str,
    filename: Optional[str] = None,
    directory: Optional[str] = None,
    mute: bool = False,
    **kwargs: Union[str, int, float, bool, None, Dict[str, str]]
) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("A valid URL is required.")

    target_dir = directory or "./"
    parsed_url = urlparse(url)
    headers_arg = kwargs.pop("headers", {})
    headers: Dict[str, str] = dict(headers_arg) if isinstance(headers_arg, dict) else {}

    if "cookie" not in headers and config.XFUSION_COOKIE:
        headers["cookie"] = config.XFUSION_COOKIE

    if parsed_url.hostname == "huggingface.co":
        if "authorization" not in headers and config.HF_HUB_TOKEN:
            headers["authorization"] = f"Bearer {config.HF_HUB_TOKEN}"
        if config.is_proxy_needed() and config.PROXY_URL_PREFIX:
            parsed_url = urlparse(f"{config.PROXY_URL_PREFIX}/{parsed_url.geturl()}")

    elif parsed_url.hostname == "civitai.com":
        if "authorization" not in headers and config.CIVITAI_TOKEN:
            headers["authorization"] = f"Bearer {config.CIVITAI_TOKEN}"
        if config.is_proxy_needed() and config.PROXY_URL_PREFIX:
            parsed_url = urlparse(f"{config.PROXY_URL_PREFIX}/{parsed_url.geturl()}")

    kwargs["headers"] = headers

    if not target_dir.endswith("/"):
        target_dir += "/"

    os.makedirs(target_dir, exist_ok=True)

    response = requests.get(parsed_url.geturl(), stream=True, **kwargs)
    if filename is None:
        filename = urlparse(response.url).path.split("/")[-1]
        content_disposition = unquote(response.headers.get("content-disposition", ""))
        if content_disposition:
            match = re.match(r'.*filename="([^"]+)".*', content_disposition)
            if match:
                try:
                    filename = match.group(1).split("/")[-1]
                except IndexError:
                    pass

    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path):
        return target_path

    total_size = int(response.headers.get("content-length", 0))
    unfinished_path = os.path.join(target_dir, f"unfinished.{filename}")

    if not mute:
        with open(unfinished_path, "wb") as f, tqdm(
            desc=target_path,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(8192 * 2):
                f.write(chunk)
                pbar.update(len(chunk))
    else:
        with open(unfinished_path, "wb") as f:
            for chunk in response.iter_content(8192 * 2):
                f.write(chunk)

    os.rename(unfinished_path, target_path)
    return target_path


class DownloadArgumentsMixin:
    """Mixin for managing model download arguments and caching directories."""

    def __init__(self):
        self.download_kwargs: Dict[str, str] = {"directory": config.DOWNLOAD_PATH}

    def set_download_kwargs(self, **kwargs: str) -> None:
        self.download_kwargs.update(kwargs)
