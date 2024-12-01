from compel import Compel,ReturnedEmbeddingsType
from .components import get_pipeline,get_clip_from_pipeline
from .utils import EasyInitSubclass
from .download import download_file,DownloadArgumentsMixin
from .message import TGBotMixin
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline
from random import randint
import torch,threading,gc


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


class LoraEnhancerMixin(DownloadArgumentsMixin,EasyInitSubclass):
    # __oins__ here is the pipeline instance to implement.
    __oins__ = None
    overrides = ["lora_dict","set_lora","set_lora_weight"]

    def __init__(self):
        DownloadArgumentsMixin.__init__(self)
        self.lora_dict = {}
        self.download_kwargs.update(directory="./lora")

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


def get_embeds_from_pipeline(pipeline,prompt,negative_prompt):
    """
    To overcome 77 tokens limit, and support high level syntax for prompts.
    limited support for sd1 sd2 sdxl

    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :return:
    """
    tokenizers,text_encoders = get_clip_from_pipeline(pipeline)
    tokenizers = [tokenizer for tokenizer in tokenizers if tokenizer]
    text_encoders = [text_encoder for text_encoder in text_encoders if text_encoder]

    if isinstance(pipeline,StableDiffusionPipeline):
        compel = Compel(tokenizers,text_encoders,truncate_long_prompts=False)
        conditioning = compel(prompt)
        negative_conditioning = compel(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        return {"prompt_embeds":conditioning,"negative_prompt_embeds":negative_conditioning}
    elif isinstance(pipeline,StableDiffusionXLPipeline):
        compel = Compel(tokenizers,text_encoders,returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,requires_pooled=[False,True],truncate_long_prompts=False)
        conditioning,pooled = compel(prompt)
        negative_conditioning, negative_pooled = compel(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        return {"prompt_embeds":conditioning,"negative_prompt_embeds":negative_conditioning,
                "pooled_prompt_embeds":pooled,"negative_pooled_prompt_embeds":negative_pooled}
    # compel only supports sd1 sd2 sdxl now
    else:
        return {}


class CLIPEnhancerMixin:
    # __oins__ here is the pipeline instance to implement.
    __oins__ = None
    overrides = ["tokenizers","text_encoders","skip_clip_layer","get_embeds_from_pipeline","__call__"]

    def __init__(self):
        self.tokenizers, self.text_encoders = get_clip_from_pipeline(self.__oins__)

    def skip_clip_layer(self,n):
        text_encoders = [text_encoder for text_encoder in self.text_encoders if text_encoder]
        if isinstance(n,int) and  n >= 1:
            for text_encoder in text_encoders:
                if not hasattr(text_encoder.text_model.encoder,"backup_layers"):
                    text_encoder.text_model.encoder.backup_layers = text_encoder.text_model.encoder.layers
                text_encoder.text_model.encoder.layers = text_encoder.text_model.encoder.backup_layers[:-n]
        elif n is None or n == 0:
            for text_encoder in text_encoders:
                if hasattr(text_encoder.text_model.encoder,"backup_layers"):
                    text_encoder.text_model.encoder.layers = text_encoder.text_model.encoder.backup_layers
        else:
            raise ValueError("An integer >=0 is required.")

    def get_embeds_from_pipeline(self,prompt,negative_prompt,clip_skip=None):
        self.skip_clip_layer(clip_skip)
        r =  get_embeds_from_pipeline(self.__oins__,prompt,negative_prompt)
        self.skip_clip_layer(0)
        return r

    def __call__(self,**kwargs):
        kwargs.update(self.get_embeds_from_pipeline(kwargs.get("prompt"),kwargs.get("negative_prompt"),kwargs.get("clip_skip")))
        if kwargs.get("prompt_embeds") is not None:
            kwargs.update(prompt=None,negative_prompt=None)
        return self.__oins__.__call__(**kwargs)


class GenerateAssistantMixin:
    def __init__(self):
        ...


def text_to_image_and_send_to_telegram(pipeline,prompt,negative_prompt,num,seed=None,use_enhancer=True,**kwargs):
    kwargs.update(prompt=prompt,negative_prompt=negative_prompt)
    seeds = []
    if seed:
        seeds.append(seed)
    else:
        for i in range(num):
            seeds.append(randint(-2 ** 63, 2 ** 64 - 1))
    for item in seeds:
        kwargs.update(generator=torch.Generator(pipeline.device).manual_seed(item))
        image = pipeline(**kwargs).images[0] if use_enhancer else pipeline.__oins__(**kwargs).images[0]
        caption = f"Prompt: {prompt[:384]}\n\nNegative Prompt: {negative_prompt[:384]}\n\nStep: {kwargs.get('num_inference_steps')}, CFG: {kwargs.get('guidance_scale')}, CLIP Skip: {kwargs.get('clip_skip')}\nSampler: {pipeline.scheduler.config._class_name}\nLoRa: {pipeline.lora_dict}\nSeed: {item}\n\nModel:{pipeline.model_name}"
        threading.Thread(target=lambda: pipeline.send_PIL_photo(image,file_name="Colab.PNG",file_type="PNG",caption=caption)).start()
        torch.cuda.empty_cache()


class PipelineEnhancer(LoraEnhancerMixin,CLIPEnhancerMixin,TGBotMixin,EasyInitSubclass):
    overrides = ["model_name","to","text_to_image_and_send_to_telegram"]

    def __init__(self,__oins__):
        EasyInitSubclass.__init__(self, __oins__)
        LoraEnhancerMixin.__init__(self)
        CLIPEnhancerMixin.__init__(self)
        TGBotMixin.__init__(self)
        self.model_name = __oins__.name_or_path

    def to(self, *args, **kwargs):
        self.__oins__ = self.__oins__.to(*args, **kwargs)
        return self

    def __call__(self,**kwargs):
        prompt = kwargs["prompt"]
        negative_prompt = kwargs["negative_prompt"]
        if isinstance(prompt,list) and isinstance(negative_prompt,list):
            prompt_type = list
        elif isinstance(prompt,list) and not isinstance(negative_prompt,list):
            negative_prompt = [negative_prompt for _ in range(len(prompt))]
            prompt_type = list
        elif not isinstance(prompt,list) and isinstance(negative_prompt,list):
            prompt = [prompt for _ in range(len(negative_prompt))]
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

    def text_to_image_and_send_to_telegram(self,prompt,negative_prompt,num,seed=None,**kwargs):
        return text_to_image_and_send_to_telegram(self,prompt,negative_prompt,num,seed=seed,**kwargs)


def get_enhancer(pipeline_or_model,**kwargs):

    if isinstance(pipeline_or_model,str):
        pipeline = get_pipeline(pipeline_or_model,**kwargs)
        enhancer = PipelineEnhancer(pipeline)
        enhancer.model_name = pipeline_or_model
    else:
        pipeline = pipeline_or_model
        enhancer = PipelineEnhancer(pipeline)

    download_kwargs = kwargs.get("download_kwargs")
    if isinstance(download_kwargs,dict):
        enhancer.set_download_kwargs(**download_kwargs)
    return enhancer
