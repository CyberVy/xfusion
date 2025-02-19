import os,shutil,gc
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
from PIL.Image import Image,Resampling,merge,fromarray
import PIL.Image
import PIL.ImageOps
import cv2
import torch
import numpy as np
from typing import Tuple,Union,Optional,Callable
from types import FunctionType,MethodType


executor = ThreadPoolExecutor()


class EasyInitSubclass:
    """
        A utility class for delegating most behavior to an internal object (`__oins__`),
        while allowing specific attributes and methods to be overridden in subclasses.

        Purpose:
            - Simplifies the extension of complex objects by delegating their behaviors to `__oins__`.
            - Allows subclasses to define their own attributes and methods by explicitly listing them in the `overrides` attribute.

        Important Notes:
            - Make sure to include all directly defined attributes and methods of your subclass in the `overrides` list.
            - Init EasyInitSubclass first when subclassed.
            - Do not put the variable names starting with "__" but not ending with "__" into attribute overrides (It,s a python feature called Name Mangling).
            - If an `AttributeError` occurs for an overridden attribute, verify that the attribute is correctly listed in `overrides`.

        Example:
            >>> class Extended(EasyInitSubclass):
            >>>    # Custom attributes and methods
            >>>    extended_value = []
            >>>    overrides = ["extended_method", "extended_value"]
            >>>
            >>>    def extended_method(self):
            >>>        return self.extended_value
            >>>
            >>>    def __init__(self, obj):
            >>>        EasyInitSubclass.__init__(self, obj)
            >>>        # Initialize custom attributes
            >>>        self.extended_value = []
            >>> obj = 1  # Original object
            >>> extended_obj = Extended(obj)
            >>> print(extended_obj.extended_method())

        Attributes:
            - `overrides` (list): A list of attribute and method names that should not be delegated to `__oins__`.

        """
    overrides = ["__oins__","__oinstype__"]

    def __init__(self,__oins__):
        self.__oins__ = __oins__
        self.__oinstype__ = __oins__.__class__

    def __init_subclass__(cls):
        for item in cls.__bases__:
            if hasattr(item, "overrides"):
                cls.overrides.extend(item.overrides)
        cls.overrides = list(set(cls.overrides))

    def __setattr__(self, key, value):
        if key in object.__getattribute__(self,"overrides"):
            return object.__setattr__(self,key,value)
        else:
            return self.__oins__.__setattr__(key,value)

    def __getattribute__(self, item):
        if item in object.__getattribute__(self,"overrides"):
            return object.__getattribute__(self,item)
        else:
            # when AttributeError: __oins__.__class__ object has no attribute '{item}' found
            # please check the if {item} is in overrides
            return object.__getattribute__(self, "__oins__").__getattribute__(item)

    def __call__(self, *args, **kwargs):
        return self.__oins__.__call__(*args,**kwargs)

    def __repr__(self):
        return self.__oins__.__repr__()

    def __str__(self):
        return self.__oins__.__str__()

    def __getitem__(self, item):
        return self.__oins__.__getitem__(item)

    def __len__(self):
        return self.__oins__.__len__()

    def __contains__(self, item):
        return self.__oins__.__contains__(item)

    def __iter__(self):
        return self.__oins__.__iter__()

    def __next__(self):
        return self.__oins__.__next__()

    def __enter__(self):
        return self.__oins__.__enter__(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.__oins__.__exit__(self,exc_type,exc_val,exc_tb)

    def __del__(self):
        # final step before self is completely deleted from RAM
        print(f"{object.__getattribute__(self,'__class__').__name__}:{id(self)} has been deleted from RAM.")

def delete(obj) -> Tuple[int,list,int,list]:
    """
        Example:
        >>> a = [1,2,3]
        >>> b = a
        >>> delete(a)
    """
    if obj is None:
        return 0, [], 0,[]
    i = 0
    _i = 0
    referrers = []
    _referrers = []
    for item in gc.get_referrers(obj):
        if hasattr(item, "__dict__"):
            # get the correct __dict__ by object.__getattribute__
            # item.__dict__ may not work when item.__getattribute__ is overridden
            __dict__ = object.__getattribute__(item,"__dict__")
        elif isinstance(item, dict):
            __dict__ = item
        elif isinstance(item, list):
            for index, _ in enumerate(item):
                if _ is obj:
                    item[index] = None
                    referrers.append(f"list.{index}")
                    i += 1
            continue
        else:
            _referrers.append(id(item))
            _i += 1
            continue

        target_keys = []
        for key, value in __dict__.items():
            if value is obj:
                target_keys.append(key)
                referrers.append(f"dict.{key}")
                i += 1
        for target_key in target_keys:
            __dict__.update({target_key: None})

    return i, referrers, _i, _referrers

def threads_execute(f:Union[FunctionType | MethodType],args:Union[list | tuple],_await:bool = True):
    threads = []
    if _await:
        for arg in args[1:]:
            thread = threading.Thread(target=f,args=(arg,))
            threads.append(thread)
            thread.start()
        f(args[0])
        for thread in threads:
            thread.join()
    else:
        for arg in args:
            thread = threading.Thread(target=f, args=(arg,))
            threads.append(thread)
            thread.start()
    return threads

def naive_async(f):
    @functools.wraps(f)
    def wrapper(*args,**kwargs):
        return executor.submit(f,*args,**kwargs)
    return wrapper

def allow_return_error(f:Union[FunctionType | MethodType]):
    @functools.wraps(f)
    def wrapper(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except Exception as e:
            return f"{e}"
    return wrapper

def load_image(
    image: Union[str, PIL.Image.Image], convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image

def normalize_image_size(image_size:Tuple[int,int],target_pixels:int,scale_divisor:int = 8) -> Tuple[int,int]:
    """
    upscale or downscale the image size with the same aspect ratio to the target pixels
    """
    width, height = image_size
    scale = (target_pixels / width / height) ** 0.5
    width = int(width * scale)
    height = int(height * scale)
    width = width - width % scale_divisor
    height = height - height % scale_divisor
    return width,height

def normalize_image(image:Image, target_pixels:int,scale_divisor:int = 8) -> Image:
    """
    upscale or downscale the image with the same aspect ratio to the target pixels
    """
    width,height = image.size
    width,height = normalize_image_size((width,height),target_pixels,scale_divisor)
    return image.resize((width,height),Resampling.LANCZOS)

def convert_mask_image_to_rgb(mask_image:Image) -> Image:
    """
    convert rgba mask image to rgb image
    """
    r, g, b, a = mask_image.split()
    return merge("L", [a]).convert("RGB")

def convert_image_to_canny(image:Image,low_threshold:Optional[int] = None,high_threshold:Optional[int] = None) -> Image:

    low_threshold = low_threshold if low_threshold is not None else 100
    high_threshold = high_threshold if high_threshold is not None else 200

    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = fromarray(image)
    return image

def dict_to_str(_dict:dict) -> str:
    r = ""
    for key,value in _dict.items():
        r += f"{key}: {value}\n\n"
    return r

def free_memory_to_system():
    gc.collect()
    torch.cuda.empty_cache()
