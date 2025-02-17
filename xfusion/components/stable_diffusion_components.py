from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline,StableDiffusion3Pipeline
from diffusers import ControlNetModel,SD3ControlNetModel
from .component_const import default_stable_diffusion_model_url
from .component_const import SD_V1_CONFIG_PATH,SD_V2_CONFIG_PATH,SD_3_CONFIG_PATH,SD_XL_CONFIG_PATH
from .component_const import SD_V1_V2_CONTROLNET_CONFIG_PATH,SD_XL_CONTROLNET_CONFIG_PATH
from .component_utils import infer_model_version
from ..download import download_file
from ..const import HF_HUB_TOKEN
import requests
import torch
import os


def load_stable_diffusion_controlnet(controlnet_model,
                                     model_version=None,file_format="safetensors",download_kwargs=None,**kwargs):
    use_internet = True
    model_version = str(model_version).lower()
    if kwargs.get("token") is None:
        kwargs.update(token=HF_HUB_TOKEN)
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if download_kwargs is None:
        download_kwargs = {}

    if controlnet_model.startswith(".") or controlnet_model.startswith("/") or controlnet_model.startswith("~"):
        use_internet = False

    kwargs.update(cache_dir=download_kwargs.get("directory"))

    if use_internet:
        # from Hugging face
        if not (controlnet_model.startswith("http://") or controlnet_model.startswith("https://")):
            if model_version in ["1","1.5","2","xl","pony"]:
                return ControlNetModel.from_pretrained(controlnet_model,variant="fp16",**kwargs)
            elif model_version in ["3","3.5"]:
                return SD3ControlNetModel.from_pretrained(controlnet_model,**kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
        # from single file
        else:
            file_path = download_file(controlnet_model, **download_kwargs)
            # bad file name from the url, 'filename.tensor/bin/ckpt' is wanted, but get 'filename'

            if not (file_path.endswith(".safetensors") or file_path.endswith(".bin") or file_path.endswith(".ckpt")):
                print(f"Warning: Can't get the format of the downloaded file, guess it is .{file_format}")
                os.rename(file_path, f"{file_path}.{file_format}")
                file_path = f"{file_path}.{file_format}"

            if not model_version:
                inferred_model_version = infer_model_version(file_path)
                if "controlnet_xl_large" == inferred_model_version:
                    model_version = "xl"
                elif "controlnet" == inferred_model_version:
                    model_version = "1.5"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")
                print(f"Auto detect result: {model_version}. If not work, please pass in 'model_version' manually.")

            if model_version in ["1","1.5","2"]:
                return ControlNetModel.from_single_file(file_path,config=SD_V1_V2_CONTROLNET_CONFIG_PATH,**kwargs)
            elif model_version in ["xl", "pony"]:
                return ControlNetModel.from_single_file(file_path,config=SD_XL_CONTROLNET_CONFIG_PATH,**kwargs)
            elif model_version in ["3", "3.5"]:
                return SD3ControlNetModel.from_single_file(file_path,config=None,**kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
    else:
        # from local single file
        if controlnet_model.endswith(".safetensors") or controlnet_model.endswith(".bin") or controlnet_model.endswith(".ckpt"):

            if not model_version:
                inferred_model_version = infer_model_version(controlnet_model)
                if "controlnet_xl_large" == inferred_model_version:
                    model_version = "xl"
                elif "controlnet" == inferred_model_version:
                    model_version = "1.5"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")
                print(f"Auto detect result: {model_version}. If not work, please pass in 'model_version' manually.")

            if model_version in ["1", "1.5", "2"]:
                return ControlNetModel.from_single_file(controlnet_model, config=SD_V1_V2_CONTROLNET_CONFIG_PATH,**kwargs)
            elif model_version in ["xl", "pony"]:
                return ControlNetModel.from_single_file(controlnet_model, config=SD_XL_CONTROLNET_CONFIG_PATH, **kwargs)
            elif model_version in ["3", "3.5"]:
                return SD3ControlNetModel.from_single_file(controlnet_model, config=None, **kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
        # from local directory
        else:
            if model_version in ["1","1.5", "2", "xl", "pony"]:
                return ControlNetModel.from_pretrained(controlnet_model,**kwargs)
            elif model_version in ["3", "3.5"]:
                return SD3ControlNetModel.from_pretrained(controlnet_model, **kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")


def load_stable_diffusion_pipeline(model=None,
                                   model_version=None, file_format="safetensors", download_kwargs=None, **kwargs):
    """
    :param model: hugging face repo id or unet file URI
    :param model_version: base model version, make sure this parameter is "xl" if the model is based on XL
    :param file_format: only for model while it is unet file, when not able to get the file format,this parameter will be the file format.
    :param download_kwargs:
    :param kwargs: for .from_pretrained
    :return:
    """
    use_internet = True
    model = default_stable_diffusion_model_url if not model else model
    model_version = "" if model_version is None else model_version
    model_version = str(model_version).lower()
    if download_kwargs is None:
        download_kwargs = {}
    if kwargs.get("token") is None:
        kwargs.update(token=HF_HUB_TOKEN)
    if kwargs.get("torch_dtype") is None:
        kwargs.update({"torch_dtype": torch.float16})
    if model.startswith(".") or model.startswith("/") or model.startswith("~"):
        use_internet = False

    kwargs.update(cache_dir=download_kwargs.get("directory"))

    if use_internet:
        # from Hugging face
        if not (model.startswith("http://") or model.startswith("https://")):
            text = requests.get(f"https://huggingface.co/{model}/tree/main/vae?not-for-all-audiences=true").text
            is_fp32 = "diffusion_pytorch_model.safetensors" in text
            is_fp16 = "diffusion_pytorch_model.fp16.safetensors" in text
            if is_fp16:
                kwargs.update({"variant": "fp16"})
                print("model.fp16.safetensors is selected.")
            elif is_fp32:
                print("model.safetensors is selected.")
            else:
                raise Exception("Model is not supported.")
            return DiffusionPipeline.from_pretrained(model, **kwargs)
        # from single file
        else:
            file_path = download_file(model,**download_kwargs)

            # bad file name from the url, 'filename.tensor/bin/ckpt' is wanted, but get 'filename'
            if not (file_path.endswith(".safetensors") or file_path.endswith(".bin") or file_path.endswith(".ckpt")):
                print(f"Warning: Can't get the format of the downloaded file, guess it is .{file_format}")
                os.rename(file_path,f"{file_path}.{file_format}")
                file_path = f"{file_path}.{file_format}"
            print(f"Loading the model {model_version}...")

            if not model_version:
                inferred_model_version = infer_model_version(file_path)
                if "sd3" in inferred_model_version:
                    model_version = "3"
                elif "xl_base" == inferred_model_version:
                    model_version = "xl"
                elif "v1" == inferred_model_version:
                    model_version = "1"
                elif "v2" == inferred_model_version:
                    model_version = "2"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")
                print(f"Auto detect result: {model_version}. If not work, please pass in 'model_version' manually.")

            if model_version in ["xl","pony"]:
                return StableDiffusionXLPipeline.from_single_file(file_path,config=SD_XL_CONFIG_PATH,**kwargs)
            elif model_version == "3":
                return StableDiffusion3Pipeline.from_single_file(file_path,config=SD_3_CONFIG_PATH,**kwargs)
            elif model_version == "2":
                return StableDiffusionPipeline.from_single_file(file_path,config=SD_V2_CONFIG_PATH,**kwargs)
            elif model_version == "1":
                return StableDiffusionPipeline.from_single_file(file_path,config=SD_V1_CONFIG_PATH,**kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
    else:
        # from local single file
        if model.endswith(".safetensors") or model.endswith(".bin") or model.endswith(".ckpt"):
            if not model_version:
                inferred_model_version = infer_model_version(model)
                if "sd3" in inferred_model_version:
                    model_version = "3"
                elif "xl_base" == inferred_model_version:
                    model_version = "xl"
                elif "v1" == inferred_model_version:
                    model_version = "1"
                elif "v2" == inferred_model_version:
                    model_version = "2"
                else:
                    raise ValueError(f"{model_version} is not supported yet.")
                print(f"Auto detect result: {model_version}. If not work, please pass in 'model_version' manually.")

            if model_version in ["xl","pony"]:
                return StableDiffusionXLPipeline.from_single_file(model, config=SD_XL_CONFIG_PATH, **kwargs)
            elif model_version == "3":
                return StableDiffusion3Pipeline.from_single_file(model, config=SD_3_CONFIG_PATH, **kwargs)
            elif model_version == "2":
                return StableDiffusionPipeline.from_single_file(model, config=SD_V2_CONFIG_PATH, **kwargs)
            elif model_version == "1":
                return StableDiffusionPipeline.from_single_file(model, config=SD_V1_CONFIG_PATH, **kwargs)
            else:
                raise ValueError(f"{model_version} is not supported yet.")
        # from local directory
        else:
            return DiffusionPipeline.from_pretrained(model, **kwargs)
