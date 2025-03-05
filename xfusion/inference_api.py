from . import const
import requests


def inference_api(prompt,model,token=None,**kwargs):
    """
    :param prompt:
    :param model: huggingface repo id
    :param token:
    :param kwargs: model params like
        num_inference_steps
        guidance_scale
        seed
        ...
    :return:
    """
    token = token if token is not None else const.HF_HUB_TOKEN
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "x-use-cache":"False"}
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs":prompt,"parameters":kwargs})
        return response
    except:
        print("Failed to get the response.")
        return
