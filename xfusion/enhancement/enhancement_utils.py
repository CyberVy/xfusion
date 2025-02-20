from ..utils import EasyInitSubclass,delete,free_memory_to_system
from ..ui.ui_utils import UIMixin
from ..download import DownloadArgumentsMixin,download_file
from ..message import TGBotMixin


class FromURLMixin:
    overrides = ["from_url"]

    @classmethod
    def from_url(cls,url,init_sub_pipelines=True,**kwargs):
        raise NotImplementedError(f"{cls} not implement 'from_url'")


def load_lora(pipeline,lora_uri,lora_name,download_kwargs=None,**kwargs):

    use_internet = True
    if lora_uri.startswith(".") or lora_uri.startswith("/") or lora_uri.startswith("~"):
        use_internet = False
    if download_kwargs is None:
        download_kwargs = {}
    if use_internet:
        lora_path = download_file(lora_uri,**download_kwargs)
        pipeline.load_lora_weights(lora_path,adapter_name=lora_name,**kwargs)
    else:
        pipeline.load_lora_weights(lora_uri,adapter_name=lora_name,**kwargs)
class LoraEnhancerMixin(DownloadArgumentsMixin,EasyInitSubclass):
    # __oins__ here is the pipeline instance to implement.
    __oins__ = None
    overrides = ["lora_dict","set_lora","set_lora_strength","delete_adapters"]

    def __init__(self):
        DownloadArgumentsMixin.__init__(self)
        self.lora_dict = {}

    def set_lora(self,lora_uri,lora_name,weight=0.4):
        if lora_name not in self.lora_dict:
            download_kwargs = self.download_kwargs.copy()
            directory = download_kwargs.get("directory", "./")
            if directory.endswith("/"):
                directory += "lora"
            else:
                directory += "/lora"
            download_kwargs.update(directory=directory)

            load_lora(self,lora_uri,lora_name,download_kwargs)

        self.set_lora_strength(lora_name,weight)

    def set_lora_strength(self,lora_name,weight):
        self.lora_dict.update({lora_name:weight})
        self.set_adapters(list(self.lora_dict.keys()),list(self.lora_dict.values()))

    def delete_adapters(self,adapter_names):
        self.__oins__.delete_adapters(adapter_names)
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for name in adapter_names:
            self.lora_dict.pop(name)
        free_memory_to_system()


class ControlnetEnhancerMixin:
    overrides = ["_controlnet",
                 "text_to_image_controlnet_pipeline","image_to_image_controlnet_pipeline","inpainting_controlnet_pipeline",
                 "load_controlnet","offload_controlnet","_check_controlnet_inference_kwargs"]
    def __init__(self):
        self._controlnet = None
        self.text_to_image_controlnet_pipeline = None
        self.image_to_image_controlnet_pipeline = None
        self.inpainting_controlnet_pipeline = None

    def _check_controlnet_inference_kwargs(self,kwargs):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'check_controlnet_inference_kwargs'")

    def load_controlnet(self,controlnet_model=None):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'load_controlnet'")

    def offload_controlnet(self):
        delete(self._controlnet)
        free_memory_to_system()


class PipelineEnhancerBase(ControlnetEnhancerMixin,LoraEnhancerMixin,TGBotMixin,FromURLMixin,UIMixin,EasyInitSubclass):
    pipeline_map = {}
    scheduler_map = {}

    overrides = ["pipeline_map","scheduler_map",
                 "enhancer_class","is_empty_pipeline","model_version","pipeline_type","pipeline_class",
                 "model_name","_scheduler","sub_pipelines",
                 "image_to_image_pipeline","inpainting_pipeline",
                 "sync_sub_pipelines_mixin_kwargs",
                 "check_original_pipeline","check_inference_kwargs",
                 "set_scheduler","reset_scheduler",
                 "to","clear","reload","load"]

    @property
    def is_empty_pipeline(self):
        return self.__oins__ is None

    def sync_sub_pipelines_mixin_kwargs(self):
        for pipeline in self.sub_pipelines.values():
            pipeline.telegram_kwargs = self.telegram_kwargs
            pipeline.download_kwargs = self.download_kwargs
            pipeline.lora_dict = self.lora_dict
            pipeline.model_name = self.model_name

    def __init__(self,__oins__,init_sub_pipelines=True):
        EasyInitSubclass.__init__(self,__oins__)
        TGBotMixin.__init__(self)
        LoraEnhancerMixin.__init__(self)
        ControlnetEnhancerMixin.__init__(self)
        self.enhancer_class:"PipelineEnhancerBase" = object.__getattribute__(self,"__class__")
        self.sub_pipelines = {}
        self.model_name = None

        # support empty pipeline
        if __oins__ is None:
            return

        # pipeline_type: 0,1,2 -> text_to_image, image_to_image, inpainting
        self.model_version,self.pipeline_type,self.pipeline_class = self.check_original_pipeline()
        self.model_name = self.name_or_path
        self._scheduler = self.scheduler
        components = self.components

        vae = components.get("vae")
        if vae:
            vae.enable_tiling()

        if "flux" in self.__oins__.__class__.__name__.lower():
            components.pop("image_encoder",None)
            components.pop("feature_extractor", None)

        if init_sub_pipelines:
            if self.pipeline_type != 0:
                self.text_to_image_pipeline = self.enhancer_class(self.pipeline_map[self.model_version][0](**components),
                                                                  init_sub_pipelines=False)
                self.sub_pipelines.update(text_to_image_pipeline=self.text_to_image_pipeline)
            else:
                self.text_to_image_pipeline = self

            if self.pipeline_type != 1:
                self.image_to_image_pipeline = self.enhancer_class(self.pipeline_map[self.model_version][1](**components),
                                                                   init_sub_pipelines=False)
                self.sub_pipelines.update(image_to_image_pipeline=self.image_to_image_pipeline)
            else:
                self.image_to_image_pipeline = self

            if self.pipeline_type != 2:
                self.inpainting_pipeline =  self.enhancer_class(self.pipeline_map[self.model_version][2](**components),
                                                                init_sub_pipelines=False)
                self.sub_pipelines.update(inpainting_pipeline=self.inpainting_pipeline)
            else:
                self.inpainting_pipeline = self

            self.sync_sub_pipelines_mixin_kwargs()

    def check_original_pipeline(self):
        for model_version, pipeline_class_tuple in self.pipeline_map.items():
            for pipeline_type,pipeline_class in enumerate(pipeline_class_tuple):
                if pipeline_class == self.__oinstype__:
                    return model_version,pipeline_type,pipeline_class
        raise TypeError(f"{self.__oinstype__} is not supported yet.")

    def check_inference_kwargs(self,kwargs):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'check_inference_kwargs'")

    def set_scheduler(self,scheduler_type,**kwargs):
        if not isinstance(scheduler_type,str):
            self.scheduler = scheduler_type.from_config(self.scheduler.config,**kwargs)
        else:
            if scheduler_type.upper() in self.scheduler_map:
                self.scheduler = self.scheduler_map[scheduler_type.upper()][0].from_config(
                    self.scheduler.config, **self.scheduler_map[scheduler_type.upper()][1])
            else:
                raise BaseException(f"{scheduler_type} is not supported yet.")

    def reset_scheduler(self):
        self.scheduler = self._scheduler

    def to(self, *args, **kwargs):
        self.__oins__.to(*args, **kwargs)
        return self

    def clear(self):
        if not hasattr(self,"components"):
            return

        for component in self.components.values():
            uncleared = delete(component)[-1]
            if uncleared:
                print(f"Warning: {uncleared}")

        uncleared = delete(self._controlnet)[-1]
        if uncleared:
            print(f"Warning: {uncleared}")

        delete(self.__oins__)
        free_memory_to_system()

    def reload(self,url,**kwargs):

        if self.is_empty_pipeline:
            self.load(url,**kwargs)
            return

        if "/" not in url:
            raise ValueError("A URL or Hugging Face Repo ID is required.")

        download_kwargs = self.download_kwargs
        telegram_kwargs = self.telegram_kwargs
        device = self.device
        self.clear()
        object.__getattribute__(self,"__init__")(
            self.from_url(url,init_sub_pipelines=False,download_kwargs=download_kwargs,**kwargs).__oins__)
        self.set_download_kwargs(**download_kwargs)
        self.set_telegram_kwargs(**telegram_kwargs)
        self.to(device)
        self.model_name = url
        for pipeline in self.sub_pipelines.values():
            pipeline.model_name = url

    def load(self,url,**kwargs):
        if not self.is_empty_pipeline:
            raise RuntimeError("This is not an empty pipeline, use 'reload' instead.")

        if "/" not in url:
            raise ValueError("A URL or Hugging Face Repo ID is required.")

        download_kwargs = self.download_kwargs
        telegram_kwargs = self.telegram_kwargs
        object.__getattribute__(self,"__init__")(
            self.from_url(url,init_sub_pipelines=False,download_kwargs=download_kwargs,**kwargs).__oins__)
        self.set_download_kwargs(**download_kwargs)
        self.set_telegram_kwargs(**telegram_kwargs)
        self.model_name = url
        for pipeline in self.sub_pipelines.values():
            pipeline.model_name = url
