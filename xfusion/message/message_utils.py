from .message_const import Telegram_Bot_API_URL_Prefix
import requests
from io import BytesIO


def send_message_to_telegram(text,parse_mode="HTML",tg_token=None,chat_id=None):
    r = requests.post(f"{Telegram_Bot_API_URL_Prefix}/bot{tg_token}/sendMessage",
                      data={"method":"POST","chat_id":chat_id,"text":text,"parse_mode":parse_mode})
    return r

def send_PIL_photo(image,**kwargs):
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
    caption = ""
    file_name = "file.PNG"
    file_type = "PNG"
    token = "8001790084:AAFNqWprWz7WUnco5fob6U0CMHwockkZY8M"
    chat_id = "5143886367"
    if "caption" in kwargs.keys():
        caption = kwargs["caption"]
    if "file_name" in kwargs.keys():
        file_name = kwargs["file_name"]
    if "file_type" in kwargs.keys():
        file_type = kwargs["file_type"]
    if "token" in kwargs.keys():
        token = kwargs["token"]
    if "chat_id" in kwargs.keys():
        chat_id = kwargs["chat_id"]
    image_byte_array = BytesIO()
    image.save(image_byte_array, format=file_type)
    image_byte_array.seek(0)
    r = requests.post(f"{Telegram_Bot_API_URL_Prefix}/bot{token}/sendDocument",
                      data={"chat_id": chat_id, "caption": caption},
                      files={"document": (file_name,image_byte_array,"image/png")})
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
