token="HF_HUB_TOKEN"

from xfusion.download import download_hf_repo_files,download_file

def download_repo(repo_id,directory="./",token=None):
    return download_hf_repo_files(repo_id,directory=directory,token=token)


from huggingface_hub import HfApi
def upload_repo(repo_id,folder_path,path_in_repo="/"):
    api = HfApi()
    try:
        api.create_repo(repo_id,token=token)
    except:
        pass
    return api.upload_folder(repo_id=repo_id,folder_path=folder_path,path_in_repo=path_in_repo,token=token)


def upload_file_to_repo(repo_id,file_path,path_in_repo):
    api = HfApi()
    return api.upload_file(repo_id=repo_id,path_or_fileobj=file_path,path_in_repo=path_in_repo,token=token)
