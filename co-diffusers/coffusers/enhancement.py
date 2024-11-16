from compel import Compel,ReturnedEmbeddingsType
from .components import *
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline

def get_embeds_from_pipe(pipe,prompt,negative_prompt):
    tokenizers,text_encoders = get_clip_from_pipe(pipe)
    tokenizers = [tokenizer for tokenizer in tokenizers]
    text_encoders = [text_encoder for text_encoder in text_encoders]

    if isinstance(pipe,StableDiffusionPipeline):
        compel = Compel(tokenizers,text_encoders)
        conditioning = compel(prompt)
        negative_condition = compel(negative_prompt)
        return conditioning,negative_condition

    elif isinstance(pipe,StableDiffusionXLPipeline):
        compel = Compel(tokenizers,text_encoders,returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,requires_pooled=[False,True])
        conditioning,pooled = compel(prompt)
        negative_conditioning, negative_pooled = compel(negative_prompt)
        return conditioning,negative_conditioning,pooled,negative_pooled
