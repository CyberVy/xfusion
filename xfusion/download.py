from .const import XFUSION_COOKIE,HF_HUB_TOKEN,CIVITAI_TOKEN
from .const import PROXY_URL_PREFIX,NEED_PROXY
import requests
from tqdm import tqdm
from urllib.parse import urlparse,unquote
import os
import re


def get_hf_repo_filename_url_dict(repo_id:str,subfolders=None,token=None) -> dict:
    token = token if token is not None else HF_HUB_TOKEN
    headers = {"authorization":token}
    hf_url = "https://huggingface.co"
    hf_url = f"{PROXY_URL_PREFIX}/{hf_url}" if NEED_PROXY else hf_url
    print(f"{hf_url}/api/models/{repo_id}")
    print(requests.get(f"{hf_url}/api/models/{repo_id}", headers=headers).text)
    json_info = requests.get(f"{hf_url}/api/models/{repo_id}",headers=headers).json()
    file_name_list =  json_info.get("siblings")
    if file_name_list is None:
        print(f"The repo {repo_id} is not existing or the repo is private.")
        return {}
    complete_filename_url_dict = {}
    for item in file_name_list:
        file_name = item["rfilename"]
        complete_filename_url_dict[file_name] = f"{hf_url}/{repo_id}/resolve/main/{file_name}?download=true"

    if subfolders is None:
        return complete_filename_url_dict

    subfolders = [subfolders] if not isinstance(subfolders,list) else subfolders
    filename_url_dict = {}
    for file_name in complete_filename_url_dict.keys():
        for subfolder in subfolders:
            subfolder = f"{subfolder}/" if not subfolder.endswith("/") else subfolder
            if file_name.startswith(subfolder):
                filename_url_dict[file_name] = complete_filename_url_dict[file_name]

    return filename_url_dict

def download_file(url:str,filename=None,directory=None,mute=False,**kwargs):

    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("A valid URL is required.")

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
        os.makedirs(directory)

    response = requests.get(url.geturl(),stream=True,**kwargs)
    if filename is None:
        filename = urlparse(response.url).path.split("/")[-1]
        content_disposition = unquote(response.headers.get("content-disposition",""))
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
        with open(f"{directory}unfinished.{filename}", 'wb') as file, tqdm(desc=f"{directory}{filename}",total=total_size, unit='B',unit_scale=True,unit_divisor=1024,) as progress_bar:
            for data in response.iter_content(8192 * 2):
                file.write(data)
                progress_bar.update(len(data))
    else:
        with open(f"{directory}unfinished.{filename}", 'wb') as file:
            for data in response.iter_content(8192 * 2):
                file.write(data)

    os.rename(f"{directory}unfinished.{filename}",f"{directory}{filename}")
    return f"{directory}{filename}"

def download_hf_repo_files(repo_id,directory,*,subfolders=None,token=None):
    directory = directory + "/" if not directory.endswith("/") else directory
    filename_url_dict = get_hf_repo_filename_url_dict(repo_id, subfolders, token)
    r = []
    for filename,url in filename_url_dict.items():
        download_directory = directory
        download_directory = download_directory + "/".join(filename.split("/")[:-1])
        filename = filename.split("/")[-1]
        r.append(download_file(url,filename=filename,directory=download_directory))
    return r

class DownloadArgumentsMixin:
    overrides = ["download_kwargs","set_download_kwargs"]

    def __init__(self):
        self.download_kwargs = {}

    def set_download_kwargs(self,**download_kwargs):
        self.download_kwargs.update(**download_kwargs)
