from compel import Compel,ReturnedEmbeddingsType
from .components import get_pipeline,get_clip_from_pipeline
from .const import cookie
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline


def get_embeds_from_pipeline(pipeline,prompt,negative_prompt):

    tokenizers,text_encoders = get_clip_from_pipeline(pipeline)
    tokenizers = [tokenizer for tokenizer in tokenizers]
    text_encoders = [text_encoder for text_encoder in text_encoders]

    if isinstance(pipeline,StableDiffusionPipeline):
        compel = Compel(tokenizers,text_encoders)
        return {"prompt_embeds":compel(prompt),"negative_prompt_embeds":compel(negative_prompt)}
    else:
        compel = Compel(tokenizers,text_encoders,returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,requires_pooled=[False,True])
        conditioning,pooled = compel(prompt)
        negative_conditioning, negative_pooled = compel(negative_prompt)
        return {"prompt_embeds":conditioning,"negative_prompt_embeds":negative_conditioning,
                "pooled_prompt_embeds":pooled,"negative_pooled_prompt_embeds":negative_pooled}

class CLIPEnhancerMixin:
    pipeline = None
    overrides = ["__class__","overrides","pipeline",
                 "get_embeds_from_pipeline","__call__"]
    def get_embeds_from_pipeline(self,prompt,negative_prompt):
        return get_embeds_from_pipeline(self.pipeline,prompt,negative_prompt)

    def __call__(self,**kwargs):
        kwargs.update(self.get_embeds_from_pipeline(kwargs.get("prompt"),kwargs.get("negative_prompt")))
        return self.pipeline.__call__(**kwargs)

class PipelineEnhancer(CLIPEnhancerMixin):
    overrides = ["to"]
    overrides.extend(CLIPEnhancerMixin.overrides)

    def __init__(self,pipeline):
        self.pipeline = pipeline

    def __getattribute__(self, item):
        if item in object.__getattribute__(self,"__class__").overrides:
            return object.__getattribute__(self,item)
        else:
            return object.__getattribute__(self, "pipeline").__getattribute__(item)

    def to(self,*args,**kwargs):
        return self.__class__(self.pipeline.to(*args,**kwargs))

def get_enhancer(pipeline_or_model,**kwargs):

    if kwargs.get("download_kwargs", None) is None:
        kwargs.update({"headers": {"cookie": cookie}})
    if isinstance(pipeline_or_model,str):
        pipeline = get_pipeline(pipeline_or_model,**kwargs)
    else:
        pipeline = pipeline_or_model

    return PipelineEnhancer(pipeline)
