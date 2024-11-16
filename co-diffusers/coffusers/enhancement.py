from compel import Compel,ReturnedEmbeddingsType
from .components import *
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline

def get_embeds_from_pipe(pipe,prompt,negative_prompt):

    tokenizers,text_encoders = get_clip_from_pipe(pipe)
    tokenizers = [tokenizer for tokenizer in tokenizers]
    text_encoders = [text_encoder for text_encoder in text_encoders]

    if isinstance(pipe,StableDiffusionPipeline):
        compel = Compel(tokenizers,text_encoders)
        return {"prompt_embeds":compel(prompt),"negative_prompt_embeds":compel(negative_prompt)}

    elif isinstance(pipe,StableDiffusionXLPipeline):
        compel = Compel(tokenizers,text_encoders,returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,requires_pooled=[False,True])
        conditioning,pooled = compel(prompt)
        negative_conditioning, negative_pooled = compel(negative_prompt)
        return {"prompt_embeds":conditioning,"negative_prompt_embeds":negative_conditioning,
                "pooled_prompt_embeds":pooled,"negative_pooled_prompt_embeds":negative_pooled}
