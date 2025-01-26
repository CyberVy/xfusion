from .message_const import TELEGRAM_BOT_API_URL_PREFIX
import requests
from io import BytesIO
from PIL.Image import Image


def send_text(text,**kwargs):
    token = kwargs.get("token") or "8001790084:AAFNqWprWz7WUnco5fob6U0CMHwockkZY8M"
    chat_id = kwargs.get("chat_id") or "5143886367"
    parse_mode = kwargs.get("parse_mode") or ""
    r = requests.post(f"{TELEGRAM_BOT_API_URL_PREFIX}/bot{token}/sendMessage",
                      data={"chat_id":chat_id,"text":text,"parse_mode":parse_mode})
    return r

def send_pil_photo(image:Image, **kwargs):
    """
    :param image:
    :param kwargs:
        caption: str, description of the picture
        file_name: str, file name
        file_type: str, file suffix
        token: telegram bot token, get it from BotFather
        chat_id: telegram account id
    :return:
    """
    caption = kwargs.get("caption") or ""
    file_type = kwargs.get("file_type") or "PNG"
    file_name = kwargs.get("file_name") or f"file.{'JPG' if file_type.upper() == 'JPEG' else file_type.upper()}"
    token = kwargs.get("token") or "8001790084:AAFNqWprWz7WUnco5fob6U0CMHwockkZY8M"
    chat_id = kwargs.get("chat_id") or "5143886367"
    parse_mode = kwargs.get("parse_mode") or ""
    image_byte_array = BytesIO()
    image.save(image_byte_array, format=file_type)
    image_byte_array.seek(0)
    r = requests.post(f"{TELEGRAM_BOT_API_URL_PREFIX}/bot{token}/sendDocument",
                      data={"chat_id": chat_id, "caption": caption,"parse_mode":parse_mode},
                      files={"document": (file_name,image_byte_array,f"image/{file_type.lower()}")})
    print(r.json()["file_id"])
    return r


class TGBotMixin:
    overrides = ["telegram_kwargs","send_text","send_pil_photo","set_telegram_kwargs","generate_image_and_send_to_telegram"]

    def __init__(self):
        self.telegram_kwargs = {}

    def generate_image_and_send_to_telegram(self, **kwargs):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'generate_image_and_send_to_telegram'")

    def send_pil_photo(self, image, **kwargs):
        kw = self.telegram_kwargs.copy()
        kw.update(**kwargs)
        return send_pil_photo(image, **kw)

    def send_text(self,text):
        return send_text(text,**self.telegram_kwargs)

    def set_telegram_kwargs(self,**kwargs):
        self.telegram_kwargs.update(**kwargs)
