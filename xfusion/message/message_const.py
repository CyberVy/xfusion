import requests


Telegram_Bot_API_URL_Prefix = "https://api.telegram.org"
Proxy_URL_Prefix = "https://us.xsolutiontech.com"

def get_origin():
    import re
    country = None
    try:
        text = requests.get("http://104.16.0.0/cdn-cgi/trace",timeout=5).text
        search = re.search("loc=(.*)",text)
        if search:
            country = search[1]
    except IOError:
        pass
    return country


if get_origin() in ["CN",None]:
    Telegram_Bot_API_URL_Prefix = f"{Proxy_URL_Prefix}/{Telegram_Bot_API_URL_Prefix}"
