from .. import const


TELEGRAM_BOT_API_URL_PREFIX = "https://api.telegram.org"

if const.NEED_PROXY and const.PROXY_URL_PREFIX:
    TELEGRAM_BOT_API_URL_PREFIX = f"{const.PROXY_URL_PREFIX}/{TELEGRAM_BOT_API_URL_PREFIX}"
