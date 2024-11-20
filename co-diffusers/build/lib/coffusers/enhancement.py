from compel import Compel,ReturnedEmbeddingsType
from .components import get_pipeline,get_clip_from_pipeline
from .utils import EasyInitSubclass
from .download import download_file,DownloadArgumentsMixin
from .message import TGBotMixin
from .const import cookie,hf_token
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


def set_lora(pipeline,lora_uri,lora_name,weight=0.5,download_kwargs=None):

    use_internet = True
    if lora_uri.startswith(".") or lora_uri.startswith("/") or lora_uri.startswith("~"):
        use_internet = False
    if download_kwargs is None:
        download_kwargs = {}
    if use_internet:
        lora_path = download_file(lora_uri,**download_kwargs)
        pipeline.load_lora_weights(lora_path,adapter_name=lora_name)
    else:
        pipeline.load_lora_weights(lora_uri,adapter_name=lora_name)
    pipeline.set_adapters([lora_name],adapter_weights=[weight])


class LoraEnhancerMixin(DownloadArgumentsMixin):
    # __oins__ here is the pipeline instance to implement.
    __oins__ = None
    overrides = ["lora_dict","set_lora","set_lora_weight"]
    overrides.extend(DownloadArgumentsMixin.overrides)

    def __init__(self):
        self.lora_dict = {}
        DownloadArgumentsMixin.__init__(self)

    def set_lora(self,lora_uri="",lora_name="",weight=0.4):
        if not (lora_name or lora_name):
            return

        if lora_name not in self.lora_dict:
            set_lora(self.__oins__,lora_uri,lora_name,weight,self.download_kwargs)
            self.lora_dict.update({lora_name:weight})
        else:
            self.set_lora_weight(lora_name,weight)

    def set_lora_weight(self,lora_name,weight):
        self.__oins__.set_adapters([lora_name],adapter_weights=[weight])
        self.lora_dict.update({lora_name:weight})


class CLIPEnhancerMixin:
    # __oins__ here is the pipeline instance to implement.
    __oins__ = None
    overrides = ["get_embeds_from_pipeline","__call__"]

    def get_embeds_from_pipeline(self,prompt,negative_prompt):
        return get_embeds_from_pipeline(self.__oins__,prompt,negative_prompt)

    def __call__(self,**kwargs):
        kwargs.update(self.get_embeds_from_pipeline(kwargs.get("prompt"),kwargs.get("negative_prompt")))
        kwargs.update(prompt=None,negative_prompt=None)
        return self.__oins__.__call__(**kwargs)


class PipelineEnhancer(LoraEnhancerMixin,CLIPEnhancerMixin,TGBotMixin,EasyInitSubclass):
    overrides = ["to"]
    overrides.extend(LoraEnhancerMixin.overrides)
    overrides.extend(CLIPEnhancerMixin.overrides)
    overrides.extend(TGBotMixin.overrides)
    overrides.extend(EasyInitSubclass.overrides)

    def __init__(self,__oins__):
        LoraEnhancerMixin.__init__(self)
        TGBotMixin.__init__(self)
        EasyInitSubclass.__init__(self,__oins__)

    def to(self, *args, **kwargs):
        return PipelineEnhancer(self.__oins__.to(*args, **kwargs))

    def __call__(self,**kwargs):
        prompt = kwargs["prompt"]
        negative_prompt = kwargs["negative_prompt"]
        if isinstance(prompt,list) and isinstance(negative_prompt,list):
            prompt_type = list
        elif isinstance(prompt,str) and isinstance(negative_prompt,str):
            prompt_type = str
        else:
            raise ValueError("The type of prompt and negative_prompt need to be str or list.")

        prompt_str = f"{prompt} {negative_prompt}" if prompt_type == str else f"{' '.join(prompt)} {' '.join(negative_prompt)}"
        saved_lora_dict = self.lora_dict.copy()

        for lora,weight in self.lora_dict.items():
            if lora not in prompt_str:
                self.set_lora_weight(lora, 0)
                print(f"LoRA{lora}:{weight} is disable due to {lora} is not in prompts.")

        r = CLIPEnhancerMixin.__call__(self,**kwargs)
        for lora,weight in saved_lora_dict.items():
            self.set_lora_weight(lora,weight)
        return r


def get_enhancer(pipeline_or_model,**kwargs):

    if isinstance(pipeline_or_model,str):
        pipeline = get_pipeline(pipeline_or_model,**kwargs)
    else:
        pipeline = pipeline_or_model

    return PipelineEnhancer(pipeline)
