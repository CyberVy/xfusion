from __future__ import annotations
import io
import os
import base64
import requests
from typing import Tuple, Optional, Callable, Dict, Union
from PIL import Image, ImageOps
import cv2
import numpy as np


def pil_to_base64_data_uri(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to a base64 Data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def decode_image_data(image_str: str) -> Image.Image:
    """Decode base64 Data URI, raw base64 string, or download from HTTP/HTTPS URL."""
    if image_str.startswith("http://") or image_str.startswith("https://"):
        response = requests.get(image_str, timeout=30)
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    if "base64," in image_str:
        image_str = image_str.split("base64,")[1]

    image_bytes = base64.b64decode(image_str)
    return Image.open(io.BytesIO(image_bytes))


def load_image(
    image: Image.Image | str,
    convert_method: Optional[Callable[[Image.Image], Image.Image]] = None
) -> Image.Image:
    """Load image from path, URL or existing instance and apply orientation normalization."""
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            loaded = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            loaded = Image.open(image)
        else:
            raise ValueError(
                f"Invalid image path or URL: {image}"
            )
    elif isinstance(image, Image.Image):
        loaded = image
    else:
        raise ValueError("Image must be a URL, local file path, or PIL Image instance.")

    loaded = ImageOps.exif_transpose(loaded)
    if convert_method is not None:
        return convert_method(loaded)
    return loaded.convert("RGB")


def normalize_image_size(
    image_size: Tuple[int, int],
    target_pixels: int,
    scale_divisor: int = 8
) -> Tuple[int, int]:
    """Calculate scaled width/height maintaining aspect ratio and divisibility by scale_divisor."""
    width, height = image_size
    scale = (target_pixels / width / height) ** 0.5
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_width = new_width - (new_width % scale_divisor)
    new_height = new_height - (new_height % scale_divisor)
    return max(new_width, scale_divisor), max(new_height, scale_divisor)


def normalize_image(
    image: Image.Image,
    target_pixels: int,
    scale_divisor: int = 8
) -> Image.Image:
    """Resize image to target pixel budget while respecting scale_divisor alignment."""
    width, height = normalize_image_size(image.size, target_pixels, scale_divisor)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def convert_mask_image_to_rgb(mask_image: Image.Image) -> Image.Image:
    """Extract alpha channel of RGBA mask as grayscale RGB image for inpainting."""
    if mask_image.mode == "RGBA":
        _, _, _, alpha = mask_image.split()
        return Image.merge("L", [alpha]).convert("RGB")
    return mask_image.convert("RGB")


def convert_image_to_canny(
    image: Image.Image,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None
) -> Image.Image:
    """Compute Canny edge representation as a 3-channel RGB image for ControlNet conditioning."""
    low = 100 if low_threshold is None else low_threshold
    high = 200 if high_threshold is None else high_threshold

    img_np = np.array(image)
    edges = cv2.Canny(img_np, low, high)
    edges_3ch = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_3ch)


def dict_to_str(d: Dict[str, Union[str, int, float, bool, None]]) -> str:
    """Format dictionary entries into readable string representation."""
    return "".join(f"{k}: {v}\n\n" for k, v in d.items())
