from .message_const import TELEGRAM_BOT_API_URL_PREFIX
import requests
from io import BytesIO
from PIL.Image import Image


def send_message_to_telegram(text,parse_mode="HTML",tg_token=None,chat_id=None):
    r = requests.post(f"{TELEGRAM_BOT_API_URL_PREFIX}/bot{tg_token}/sendMessage",
                      data={"method":"POST","chat_id":chat_id,"text":text,"parse_mode":parse_mode})
    return r

def send_PIL_photo(image:Image,**kwargs):
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
    file_type = kwargs.get("file_type") or "JPEG"
    file_name = kwargs.get("file_name") or f"file.{"JPG" if file_type.upper() == "JPEG" else file_type.upper()}"
    token = kwargs.get("token") or "8001790084:AAFNqWprWz7WUnco5fob6U0CMHwockkZY8M"
    chat_id = kwargs.get("chat_id") or "5143886367"
    image_byte_array = BytesIO()
    image.save(image_byte_array, format=file_type)
    image_byte_array.seek(0)
    r = requests.post(f"{TELEGRAM_BOT_API_URL_PREFIX}/bot{token}/sendDocument",
                      data={"chat_id": chat_id, "caption": caption},
                      files={"document": (file_name,image_byte_array,f"image/{file_type.lower()}")})
    return r


class TGBotMixin:
    overrides = ["telegram_kwargs","send_PIL_photo","set_telegram_kwargs","generate_image_and_send_to_telegram"]

    def __init__(self):
        self.telegram_kwargs = {}

    def generate_image_and_send_to_telegram(self, **kwargs):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'generate_image_and_send_to_telegram'")

    def send_PIL_photo(self,image,**kwargs):
        kw = self.telegram_kwargs.copy()
        kw.update(**kwargs)
        send_PIL_photo(image,**kw)

    def set_telegram_kwargs(self,**kwargs):
        self.telegram_kwargs.update(**kwargs)
