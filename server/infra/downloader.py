import os
import re
from urllib.parse import urlparse, unquote
from typing import Optional, Dict, Any, List, Union
import requests
from tqdm import tqdm

from . import config


def get_hf_repo_filename_url_dict(
    repo_id: str,
    subfolders: Optional[Union[str, List[str]]] = None,
    token: Optional[str] = None
) -> Dict[str, str]:
    headers = {}
    auth_token = token if token is not None else config.HF_HUB_TOKEN
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"

    hf_url = "https://huggingface.co"
    if config.is_proxy_needed() and config.PROXY_URL_PREFIX:
        hf_url = f"{config.PROXY_URL_PREFIX}/{hf_url}"

    resp = requests.get(f"{hf_url}/api/models/{repo_id}", headers=headers)
    json_info = resp.json()
    file_name_list = json_info.get("siblings")
    if file_name_list is None:
        print(f"The repo {repo_id} does not exist or is private.")
        return {}

    complete_dict: Dict[str, str] = {}
    for item in file_name_list:
        file_name = item["rfilename"]
        complete_dict[file_name] = f"{hf_url}/{repo_id}/resolve/main/{file_name}?download=true"

    if subfolders is None:
        return complete_dict

    subfolder_list = [subfolders] if isinstance(subfolders, str) else subfolders
    filtered_dict: Dict[str, str] = {}
    for file_name, file_url in complete_dict.items():
        for sub in subfolder_list:
            prefix = f"{sub}/" if not sub.endswith("/") else sub
            if file_name.startswith(prefix):
                filtered_dict[file_name] = file_url
    return filtered_dict


def download_file(
    url: str,
    filename: Optional[str] = None,
    directory: Optional[str] = None,
    mute: bool = False,
    **kwargs: Any
) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("A valid URL is required.")

    target_dir = directory or "./"
    parsed_url = urlparse(url)
    headers = kwargs.pop("headers", {})

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
        self.download_kwargs: Dict[str, Any] = {"directory": config.DOWNLOAD_PATH}

    def set_download_kwargs(self, **kwargs: Any) -> None:
        self.download_kwargs.update(kwargs)
