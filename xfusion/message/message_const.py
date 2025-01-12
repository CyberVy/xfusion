Telegram_Bot_API_URL_Prefix = "https://api.telegram.org"
Proxy_URL_Prefix = "https://us.xsolutiontech.com"


from ..const import LOCATION

if LOCATION in ["CN",None]:
    Telegram_Bot_API_URL_Prefix = f"{Proxy_URL_Prefix}/{Telegram_Bot_API_URL_Prefix}"
