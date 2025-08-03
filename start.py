import os
from xfusion import load_enhancer
from xfusion import load_stable_diffusion_ui,load_flux_ui
from xfusion.const import GPU_COUNT

telegram_kwargs = {"token":os.environ.get("TG_TOKEN",""),"chat_id":os.environ.get("TG_ID","")}
download_kwargs = {"directory":os.environ.get("DOWNLOAD_PATH","/xfusion_models")}
pipelines = [load_enhancer(None,model_version=os.environ.get("MODEL_VERSION",""),download_kwargs=download_kwargs,telegram_kwargs=telegram_kwargs) for i in range(GPU_COUNT)]

if os.environ.get("MODEL_VERSION","") in ["", "1.5", "2", "3", "3.5", "xl", "sdxl", "pony"]:
    print("hello")
    server = load_stable_diffusion_ui(pipelines,_globals=globals(),pwa=True)
elif os.environ.get("MODEL_VERSION","") in ["flux"]:
    server = load_flux_ui(pipelines,_globals=globals(),pwa=True)
