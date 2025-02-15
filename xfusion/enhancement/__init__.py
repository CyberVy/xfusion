from .stable_diffusion_enhancement import SDPipelineEnhancer
from .flux_enhancement import FluxPipelineEnhancer


def load_enhancer(pipeline_or_model=None,**kwargs):
    model_version = kwargs.pop("model_version","")
    model_version = str(model_version).lower()
    if model_version in ["","1.5","2","3","3.5","xl","sdxl","pony"]:
        if isinstance(pipeline_or_model,str):
            if model_version in ["sdxl","pony"]:
                model_version = "xl"
            if model_version == "3.5":
                model_version = "3"
            enhancer = SDPipelineEnhancer.from_url(pipeline_or_model,model_version=model_version,**kwargs)
        else:
            pipeline = pipeline_or_model
            enhancer = SDPipelineEnhancer(pipeline)
        enhancer.model_name = pipeline_or_model
        download_kwargs = kwargs.get("download_kwargs")
        telegram_kwargs = kwargs.get("telegram_kwargs")
        if isinstance(download_kwargs, dict):
            enhancer.set_download_kwargs(**download_kwargs)
        if isinstance(telegram_kwargs,dict):
            enhancer.set_telegram_kwargs(**telegram_kwargs)
        return enhancer

    elif model_version in ["flux"]:
        if isinstance(pipeline_or_model, str):
            enhancer = FluxPipelineEnhancer.from_url(pipeline_or_model, **kwargs)
        else:
            pipeline = pipeline_or_model
            enhancer = FluxPipelineEnhancer(pipeline)
        enhancer.model_name = pipeline_or_model
        download_kwargs = kwargs.get("download_kwargs")
        telegram_kwargs = kwargs.get("telegram_kwargs")
        if isinstance(download_kwargs, dict):
            enhancer.set_download_kwargs(**download_kwargs)
        if isinstance(telegram_kwargs, dict):
            enhancer.set_telegram_kwargs(**telegram_kwargs)
        return enhancer

    else:
        raise ValueError(f"{model_version} is not supported yet.")
