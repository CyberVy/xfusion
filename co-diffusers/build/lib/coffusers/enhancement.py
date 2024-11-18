from compel import Compel,ReturnedEmbeddingsType
from .components import *
from .const import cookie
from .utils import inherit
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline


def get_embeds_from_pipe(pipe,prompt,negative_prompt):

    tokenizers,text_encoders = get_clip_from_pipe(pipe)
    tokenizers = [tokenizer for tokenizer in tokenizers]
    text_encoders = [text_encoder for text_encoder in text_encoders]

    if isinstance(pipe,StableDiffusionPipeline):
        compel = Compel(tokenizers,text_encoders)
        return {"prompt_embeds":compel(prompt),"negative_prompt_embeds":compel(negative_prompt)}
    else:
        compel = Compel(tokenizers,text_encoders,returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,requires_pooled=[False,True])
        conditioning,pooled = compel(prompt)
        negative_conditioning, negative_pooled = compel(negative_prompt)
        return {"prompt_embeds":conditioning,"negative_prompt_embeds":negative_conditioning,
                "pooled_prompt_embeds":pooled,"negative_pooled_prompt_embeds":negative_pooled}


def get_enhancer(pipeline_or_model="",**kwargs):

    if kwargs.get("download_kwargs", None) is None:
        kwargs.update({"headers": {"cookie": cookie}})
    if isinstance(pipeline_or_model,str):
        pipeline = get_pipe(pipeline_or_model,**kwargs)
    else:
        pipeline = pipeline_or_model

    class I(inherit(pipeline)):

        def __setattr__(self, key, value):
            return object.__setattr__(self,key,value)

        def __call__(self,**kwargs):
            kwargs.update(self.get_embeds_from_pipe(kwargs.get("prompt"),kwargs.get("negative_prompt")))
            kwargs.update({"prompt":None,"negative_prompt":None})
            return self.__oins__.__call__(**kwargs)

        def get_embeds_from_pipe(self,prompt,negative_prompt):
            return get_embeds_from_pipe(self.__oins__,prompt,negative_prompt)
    I.__name__ = pipeline.__class__.__name__
    I.__qualname__ = pipeline.__class__.__qualname__
    return I()

