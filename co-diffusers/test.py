import os
import threading

def cookie_test():
    import requests
    civitai_cookie = "__Secure-civitai-token=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..P8UHnrEnc9sswSnV.LRk9TjudMzG-nd5NRBJPrY3R2KlKE6q_wB8DH_IT_bSfEBMMTwYFcdN-ir8JnknFRBY8nQ3jbqTmIHBi92b8pvJCv-tjzT_TRjNJX29G_tanPi0nP_TX62CHZmVYsl2fRSosl6dwwbap4NYaMQFIiJaQFJeUPz1TZqockiclqpraHq0hcK_7g51-RbdI3-1talc6r6BkyRuiRp55sl5UBb4w-X4XkelHq7eN9DO7oIE29HE4hGjkX2o3qSmCUCX_pEdV26647zHcVctxa1Y42INbr4cc53q6Gyxzyazrc0bhOppUNUYVeQ4Lsn3VSeeq3EBU5LGcq697EeH3zEXXEYvSK5lBkvHGojBJWFSvvkOsNbjw66-5jKmSEZmgOvRZ8oEgLoK_4dIYWOtSWgozAQu24wusd1rIA3iGYCG717dsQiUiMMD5JgkHHK3DR0nCz9GWGqcJ86Dzq9ilY8z__ApT7XxoCIE8qHVmV8HqfB-WwBFpjzYL38V6lxF1vGfSMh4syXIcaNjMFtaXhpaZmWCMnOc1E2wLvwYSRQvUK9g2sNecUERB35HYS2aE_zPfwN-MBIA7dyHMQTGFWDw83urbleBda8I_k7pg7mKyvcpRmI02IJn2OV5y_5rlp-hgSru-aGVE1Y9yaQ88G0cdF8zXmblbRBxdfQ7fJSzZlVpcGh7wDdeDLrbpeg6Okv4MUxK2gzg0AEmholRUSs-uGLjWSe-AYOIBJRcNJTkEajCr-pnNxE5XIKqXJNBYEqjEX5OMs6Qgph01dpIuw_eWy2QC7KA03w5yx2txLDY_pCv6anU0kZ41ZA5eNbG-RYLf69cAsQNWEcgdysNkGnFRiiazqP77luMSj2aC9C_y3KjjM-DZeLLFDMOzLhJ9kgosbpqQmaOiYTi2Fkf3NS9r-XsYRrVFMW0U3e7V8FXW1B_BPjH0_rsRlRkh3_NV_JmYnnlevCRvgCyI_2JQAUV7EAfdBrxq4cacAM134wGbklWI20t-7oOcqHtiBLK_lBBtVHDvyWCRUQV0SeaE4GmpUdM2BvYdbHajGtWnOJdF941Eoi5X_kp0JLPzBtlX7oW5G5ySxHYW22nvKDWn9A-Cn_vRT_6PBjX-xL1z011b5SqIRq8JPblpebxgHMfaQ-Aisv6iPD_wF0AlfCwQqBNXa37_ivjFfmy4VNfFWbiWn5XyG03zjiZbKB6529QYtyHVa4ME9cmnDmvHEZkXnwtPt7RhQZ0MgHlwRFDVlp3ply0hIvkkB-LqNVkJnIYHvEOt21NXdZaN4u8p7-7bl9HbyQY.l29-hx9EZ6jpBJ4NYJjA7w;"
    hf_cookie = "token=tNdvqKWNwfXlIhemmKYNbDXDgXRshHvwXOFUzXWWVZjkNbWRfszhynxgDXSmfjZDZZuBwywspRCUZURCZBgTuchTYyqxmayZtumKiNfXtxajLQVZGKgvLSDIsWIMIbbq"
    cookie = civitai_cookie + hf_cookie

    civitai_model = "https://civitai.com/api/download/models/450105?type=Model&format=SafeTensor&size=full&fp=fp16"
    hf_model = "https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/resolve/main/Juggernaut-XI-byRunDiffusion.safetensors?download=true"
    r = requests.get(civitai_model, headers={"cookie": cookie}, stream=True)
    print(r.headers.get("content-disposition"))
    print(r.url)

def test_pipe(pipe,prompt=None):
    if prompt is None:
        prompt = "instagram photo, front view, portrait photo of a 24 y.o woman, wearing dress, beautiful face, cinematic shot"
    args = [[25,0],[30,0],[35,0],[25,1.5],[30,1.5],[35,1.5],[25,3],[30,3],[35,3],[25,4.5],[30,4.5],[35,4.5],[25,6],[30,6],[35,6]]
    r = {}
    for arg in args:
        print(f"Step:{arg[0]},CFG:{arg[1]}")
        r.update({arg.__str__():pipe(prompt,num_inference_steps=arg[0],guidance_scale=arg[1]).images[0]})
    return

def api(prompt,model,**kwargs):
    from coffusers.const import hf_token
    from coffusers.message import send_PIL_photo
    import requests,io
    from PIL import Image

    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}","x-use-cache":"False"}
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs":prompt,"parameters":kwargs})
        print(response.headers)
        if response.status_code != 200:
            return
    except:
        print("Failed to get the response.")
        return

    try:
        image = Image.open(io.BytesIO(response.content))
        send_PIL_photo(image,caption=f"Prompt:{prompt} \n\nStep: {kwargs.get('num_inference_steps')}, CFG: {kwargs.get('guidance_scale')}\nModel: {model}")
        return image
    except:
        print("Failed to send the photo.")


prompt = """
a razor-sharp portrait photo of a sexy-- gothic girl with very long black hair and dark red makeup, wearing a long- black dress, (holding her dress)+, looking at the camera, (with eyes focused on)++. The background is a graveyard.
"""

model = "stabilityai/stable-diffusion-3.5-large"
model = "black-forest-labs/FLUX.1-dev"
model = "XLabs-AI/flux-RealismLora"
# api(prompt,model=model,guidance_scale=2,seed=1)

