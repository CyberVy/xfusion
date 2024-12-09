from .stable_diffusion_enhancement import SDPipelineEnhancer
from .flux_enhancement import FluxPipelineEnhancer
from ..components import load_flux_pipeline,load_stable_diffusion_pipeline


def load_enhancer(pipeline_or_model=None,**kwargs):
    model_version = kwargs.get("model_version")
    model_version = str(model_version).lower()
    if model_version in ["none","1.5","2","3","3.5","xl","pony"]:
        if isinstance(pipeline_or_model,str):
            pipeline = load_stable_diffusion_pipeline(pipeline_or_model,**kwargs)
        else:
            pipeline = pipeline_or_model
        enhancer = SDPipelineEnhancer(pipeline)
        enhancer.model_name = pipeline_or_model
        download_kwargs = kwargs.get("download_kwargs")
        if isinstance(download_kwargs, dict):
            enhancer.set_download_kwargs(**download_kwargs)
        return enhancer
    elif model_version in ["flux"]:
        # todo: implement flux enhancer
        kwargs.pop("model_version")
        if isinstance(pipeline_or_model, str):
            pipeline = load_flux_pipeline(pipeline_or_model, **kwargs)
        else:
            pipeline = pipeline_or_model
        enhancer = FluxPipelineEnhancer(pipeline)
        enhancer.model_name = pipeline_or_model
        return enhancer
    else:
        raise ValueError(f"{model_version} is not supported yet.")
