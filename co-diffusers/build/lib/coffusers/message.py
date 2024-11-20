import requests
from io import BytesIO


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
        file_name = kwargs["token"]
    if "chat_id" in kwargs.keys():
        file_name = kwargs["chat_id"]
    image_byte_array = BytesIO()
    image.save(image_byte_array, format=file_type)
    image_byte_array.seek(0)
    r = requests.post(f"https://api.telegram.org/bot{token}/sendDocument",
                      data={"chat_id": chat_id, "caption": caption},
                      files={"document": (file_name,image_byte_array,"image/png")})


class TGBotMixin:
    overrides = ["send_PIL_photo","telegram_kwargs"]

    def __init__(self):
        self.telegram_kwargs = {}

    def send_PIL_photo(self,image):
        send_PIL_photo(image,**self.telegram_kwargs)

    def set_telegram_kwargs(self,**kwargs):
        self.telegram_kwargs.update(**kwargs)
