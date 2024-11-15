import gradio as gr
from ..const import GPU_COUNT,GPU_NAME
from .ui_utils import safe_block,lock,lists_append
from ..utils import allow_return_error,threads_execute,free_memory_to_system
from ..utils import convert_mask_image_to_rgb,convert_image_to_canny
from ..download import download_file,download_hf_repo_files
import functools
import inspect
import sys,platform


def render_model_selection(fns):
    with gr.Accordion("Model Selection", open=True):
        gr.Markdown("# Model Selection")
        model_selection_inputs = []
        model_selection_outputs = []
        with gr.Row():
            with gr.Column():
                model_selection_inputs.append(
                    gr.Textbox(value="eramth/flux-4bit", placeholder="Give me a url of the model!",
                               label="Model"))
            with gr.Column():
                model_selection_outputs.append(gr.Textbox(label="Result"))
                model_selection_btn = gr.Button("Select")
                model_selection_btn.click(fn=fns["model_selection_fn"],inputs=model_selection_inputs,
                                          outputs=model_selection_outputs)

def render_lora(fns):
    with gr.Accordion("LoRA", open=False):
        gr.Markdown("# LoRA")
        set_lora_inputs = []
        lora_outputs = []
        with gr.Row():
            with gr.Column():
                set_lora_inputs.append(gr.Textbox(placeholder="Give me a url of LoRA!", label="LoRA"))
                with gr.Row():
                    set_lora_inputs.append(gr.Textbox(placeholder="Give the LoRA a name!", label="LoRA name"))
                    set_lora_inputs.append(gr.Slider(0, 1, 0.4, step=0.05, label="LoRA strength"))
                with gr.Row():
                    delete_lora_btn = gr.Button("Delete LoRA")
                    set_lora_btn = gr.Button("Set LoRA")
            with gr.Column():
                lora_outputs.append(gr.Textbox(label="Result"))
                delete_lora_btn.click(fn=fns["delete_lora_fn"], inputs=set_lora_inputs, outputs=lora_outputs)
                set_lora_btn.click(fn=fns["set_lora_fn"], inputs=set_lora_inputs, outputs=lora_outputs)
                with gr.Row():
                    show_lora_btn = gr.Button("Show all LoRA")
                    show_lora_btn.click(fn=fns["show_lora_fn"], outputs=lora_outputs)
                with gr.Row():
                    enable_lora_btn = gr.Button("Enable all LoRA")
                    enable_lora_btn.click(fn=fns["enable_lora_fn"], outputs=lora_outputs)
                    disable_lora_btn = gr.Button("Disable all LoRA")
                    disable_lora_btn.click(fn=fns["disable_lora_fn"], outputs=lora_outputs)

def render_text_to_image(fns):
    with gr.Accordion("Text To Image", open=True):
        gr.Markdown("# Text To Image")
        t2i_inputs = []
        t2i_outputs = []

        with gr.Row():
            with gr.Column():
                t2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
            with gr.Column():
                t2i_inputs.append(gr.Slider(0, 10, 3.5, step=0.1, label="Guidance Scale"))
                t2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                with gr.Row():
                    t2i_inputs.append(gr.Slider(512, 2048, 1024, step=16, label="Width"))
                    t2i_inputs.append(gr.Slider(512, 2048, 1024, step=16, label="Height"))
            with gr.Column():
                with gr.Row():
                    t2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    t2i_inputs.append(gr.Slider(1, 10, 1, step=1, label="Num"))
                with gr.Accordion("Code", open=False):
                    t2i_callback_args_name = ",".join([str(item).split("=")[0] for item in list(
                        inspect.signature(fns["text_to_image_fn"]).parameters.values())])
                    t2i_inputs.append(
                        gr.Code(
                            f"def preprocess({t2i_callback_args_name}):\n  kwargs['callback_on_step_end'] = None\n  return {t2i_callback_args_name.replace('*', '')}",
                            language="python", label="Python"))
                t2i_outputs.append(gr.Textbox(label="Result"))
                t2i_btn = gr.Button("Run")
                t2i_btn.click(fn=fns["text_to_image_fn"], inputs=t2i_inputs, outputs=t2i_outputs)

def render_image_to_image(fns):
    with gr.Accordion("Image To Image", open=False):
        gr.Markdown("# Image To Image")
        i2i_inputs = []
        i2i_outputs = []
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Image"):
                    i2i_inputs.append(gr.Image(type="pil", label="Image"))
                i2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
            with gr.Column():
                i2i_inputs.append(gr.Slider(0, 1, 0.4, step=0.1, label="Strength"))
                i2i_inputs.append(gr.Slider(0, 10, 3.5, step=0.1, label="Guidance Scale"))
                i2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                with gr.Row():
                    i2i_inputs.append(gr.Slider(512, 2048, 1024, step=16, label="Width"))
                    i2i_inputs.append(gr.Slider(512, 2048, 1024, step=16, label="Height"))
            with gr.Column():
                with gr.Row():
                    i2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    i2i_inputs.append(gr.Slider(1, 10, 1, step=1, label="Num"))
                with gr.Accordion("Code", open=False):
                    i2i_callback_args_name = ",".join([str(item).split("=")[0] for item in list(
                        inspect.signature(fns["image_to_image_fn"]).parameters.values())])
                    i2i_inputs.append(
                        gr.Code(
                            f"def preprocess({i2i_callback_args_name}):\n  kwargs['callback_on_step_end'] = None\n  return {i2i_callback_args_name.replace('*', '')}",
                            language="python", label="Python"))
                i2i_outputs.append(gr.Textbox(label="Result"))
                i2i_btn = gr.Button("Run")
                i2i_btn.click(fn=fns["image_to_image_fn"], inputs=i2i_inputs, outputs=i2i_outputs)

def render_inpainting(fns):
    with gr.Accordion("Inpainting", open=False):
        gr.Markdown("# Inpainting")
        inpainting_inputs = []
        inpainting_outputs = []
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Inpainting Image"):
                    inpainting_inputs.append(gr.ImageMask(type="pil", label="Inpainting Image"))
                inpainting_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
            with gr.Column():
                inpainting_inputs.append(gr.Slider(0, 1, 0.8, step=0.1, label="Strength"))
                inpainting_inputs.append(gr.Slider(0, 10, 3.5, step=0.1, label="Guidance Scale"))
                inpainting_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                with gr.Row():
                    inpainting_inputs.append(gr.Slider(512, 2048, 1024, step=16, label="Width"))
                    inpainting_inputs.append(gr.Slider(512, 2048, 1024, step=16, label="Height"))
            with gr.Column():
                with gr.Row():
                    inpainting_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    inpainting_inputs.append(gr.Slider(1, 10, 1, step=1, label="Num"))
                with gr.Accordion("Code", open=False):
                    inpainting_callback_args_name = ",".join([str(item).split("=")[0] for item in list(
                        inspect.signature(fns["inpainting_fn"]).parameters.values())])
                    inpainting_inputs.append(
                        gr.Code(
                            f"def preprocess({inpainting_callback_args_name}):\n  kwargs['callback_on_step_end'] = None\n  return {inpainting_callback_args_name.replace('*', '')}",
                            language="python", label="Python"))
                inpainting_outputs.append(gr.Textbox(label="Result"))
                inpainting_btn = gr.Button("Run")
                inpainting_btn.click(fn=fns["inpainting_fn"], inputs=inpainting_inputs, outputs=inpainting_outputs)

def render_controlnet_selection(fns):
    controlnet_inputs = []
    controlnet_outputs = []
    with gr.Accordion("Controlnet Selection", open=False):
        with gr.Row():
            with gr.Column():
                controlnet_inputs.append(gr.Textbox(placeholder="Give me a controlnet URL!", label="Controlnet Model"))
                controlnet_inputs[0].change(fn=fns["set_default_controlnet_for_auto_load_controlnet_fn"],
                                            inputs=controlnet_inputs[0], outputs=controlnet_outputs)
                with gr.Row():
                    load_controlnet_button = gr.Button("Load controlnet")
                    offload_controlnet_button = gr.Button("Offload controlnet")
            controlnet_outputs.append(gr.Textbox(label="Result"))
            load_controlnet_button.click(fn=fns["load_controlnet_fn"], outputs=controlnet_outputs)
            offload_controlnet_button.click(fn=fns["offload_controlnet_fn"], outputs=controlnet_outputs)

def render_controlnet_text_to_image(fns):
    with gr.Accordion("Controlnet Text To Image", open=False):
        gr.Markdown("# Controlnet Text To Image")
        controlnet_t2i_inputs = []
        controlnet_t2i_outputs = []
        controlnet_t2i_control_image_preview_inputs = []
        controlnet_t2i_control_image_preview_outputs = []
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Controlnet Image"):
                    lists_append(gr.Image(type="pil", label="Controlnet Image"),
                                 [controlnet_t2i_inputs, controlnet_t2i_control_image_preview_inputs])
                controlnet_t2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
            with gr.Column():
                controlnet_t2i_inputs.append(gr.Slider(0, 1, 0.5, step=0.05, label="Controlnet Scale"))
                controlnet_t2i_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                controlnet_t2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                with gr.Row():
                    controlnet_t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                    controlnet_t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
                with gr.Row():
                    lists_append((gr.Slider(0, 255, 100, step=5, label="Low Threshold")),
                                 [controlnet_t2i_inputs, controlnet_t2i_control_image_preview_inputs])
                    lists_append(gr.Slider(0, 255, 200, step=5, label="High Threshold"),
                                 [controlnet_t2i_inputs, controlnet_t2i_control_image_preview_inputs])
                controlnet_t2i_control_image_preview_outputs.append(gr.Image(label="Control Image Preview"))
                for component in controlnet_t2i_control_image_preview_inputs:
                    component.change(fn=fns["controlnet_preview_fn"],
                                     inputs=controlnet_t2i_control_image_preview_inputs,
                                     outputs=controlnet_t2i_control_image_preview_outputs)
            with gr.Column():
                with gr.Row():
                    controlnet_t2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    controlnet_t2i_inputs.append(gr.Slider(1, 10, 1, step=1, label="Num"))
                with gr.Accordion("Code", open=False):
                    controlnet_t2i_args_name = ",".join([str(item).split("=")[0] for item in list(
                        inspect.signature(fns["controlnet_text_to_image_fn"]).parameters.values())])
                    controlnet_t2i_inputs.append(
                        gr.Code(
                            f"def preprocess({controlnet_t2i_args_name}):\n  # kwargs['callback_on_step_end'] = cbk.SDXLControlnetCFGCutoffCallback(0.5)\n  return {controlnet_t2i_args_name.replace('*', '')}",
                            language="python", label="Python"))
                controlnet_t2i_outputs.append(gr.Textbox(label="Result"))
                controlnet_t2i_btn = gr.Button("Run")
                controlnet_t2i_btn.click(fn=fns["controlnet_text_to_image_fn"], inputs=controlnet_t2i_inputs,
                                         outputs=controlnet_t2i_outputs)

def render_controlnet_image_to_image(fns):
    with gr.Accordion("Controlnet Image To Image", open=False):
        gr.Markdown("# Controlnet Image To Image")
        controlnet_i2i_inputs = []
        controlnet_i2i_outputs = []
        controlnet_i2i_control_image_preview_inputs = []
        controlnet_i2i_control_image_preview_outputs = []
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Images"):
                    with gr.Row():
                        lists_append(gr.Image(type="pil", label="Controlnet Image"),
                                     [controlnet_i2i_inputs, controlnet_i2i_control_image_preview_inputs])
                        controlnet_i2i_inputs.append(gr.Image(type="pil", label="Image"))
                controlnet_i2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
            with gr.Column():
                controlnet_i2i_inputs.append(gr.Slider(0, 1, 0.5, step=0.05, label="Controlnet Scale"))
                controlnet_i2i_inputs.append(gr.Slider(0, 1, 0.8, step=0.1, label="Strength"))
                controlnet_i2i_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                controlnet_i2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                with gr.Row():
                    controlnet_i2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                    controlnet_i2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
                with gr.Row():
                    lists_append((gr.Slider(0, 255, 100, step=5, label="Low Threshold")),
                                 [controlnet_i2i_inputs, controlnet_i2i_control_image_preview_inputs])
                    lists_append(gr.Slider(0, 255, 200, step=5, label="High Threshold"),
                                 [controlnet_i2i_inputs, controlnet_i2i_control_image_preview_inputs])
                controlnet_i2i_control_image_preview_outputs.append(gr.Image(label="Control Image Preview"))
                for component in controlnet_i2i_control_image_preview_inputs:
                    component.change(fn=fns["controlnet_preview_fn"],
                                     inputs=controlnet_i2i_control_image_preview_inputs,
                                     outputs=controlnet_i2i_control_image_preview_outputs)

            with gr.Column():
                with gr.Row():
                    controlnet_i2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    controlnet_i2i_inputs.append(gr.Slider(1, 10, 1, step=1, label="Num"))
                with gr.Accordion("Code", open=False):
                    controlnet_i2i_args_name = ",".join([str(item).split("=")[0] for item in list(
                        inspect.signature(fns["controlnet_image_to_image_fn"]).parameters.values())])
                    controlnet_i2i_inputs.append(
                        gr.Code(
                            f"def preprocess({controlnet_i2i_args_name}):\n  kwargs['callback_on_step_end'] = None\n  return {controlnet_i2i_args_name.replace('*', '')}",
                            language="python", label="Python"))
                controlnet_i2i_outputs.append(gr.Textbox(label="Result"))
                controlnet_i2i_btn = gr.Button("Run")
                controlnet_i2i_btn.click(fn=fns["controlnet_image_to_image_fn"], inputs=controlnet_i2i_inputs,
                                         outputs=controlnet_i2i_outputs)

def render_controlnet_inpainting(fns):
    with gr.Accordion("Controlnet Inpainting", open=False):
        gr.Markdown("# Controlnet Inpainting")
        controlnet_inpainting_inputs = []
        controlnet_inpainting_outputs = []
        controlnet_inpainting_control_image_preview_inputs = []
        controlnet_inpainting_control_image_preview_outputs = []
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Images"):
                    with gr.Column():
                        lists_append(gr.Image(type="pil", label="Controlnet Image"),
                                     [controlnet_inpainting_inputs, controlnet_inpainting_control_image_preview_inputs])
                        controlnet_inpainting_inputs.append(gr.ImageMask(type="pil", label="Inpainting Image"))
                controlnet_inpainting_inputs.append(
                    gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
            with gr.Column():
                controlnet_inpainting_inputs.append(gr.Slider(0, 1, 0.5, step=0.05, label="Controlnet Scale"))
                controlnet_inpainting_inputs.append(gr.Slider(0, 1, 0.8, step=0.1, label="Strength"))
                controlnet_inpainting_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                controlnet_inpainting_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                with gr.Row():
                    controlnet_inpainting_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                    controlnet_inpainting_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
                with gr.Row():
                    lists_append(gr.Slider(0, 255, 100, step=5, label="Low Threshold"),
                                 [controlnet_inpainting_inputs, controlnet_inpainting_control_image_preview_inputs])
                    lists_append(gr.Slider(0, 255, 200, step=5, label="High Threshold"),
                                 [controlnet_inpainting_inputs, controlnet_inpainting_control_image_preview_inputs])
                controlnet_inpainting_control_image_preview_outputs.append(gr.Image(label="Control Image Preview"))
                for component in controlnet_inpainting_control_image_preview_inputs:
                    component.change(fn=fns["controlnet_preview_fn"],
                                     inputs=controlnet_inpainting_control_image_preview_inputs,
                                     outputs=controlnet_inpainting_control_image_preview_outputs)

            with gr.Column():
                with gr.Row():
                    controlnet_inpainting_inputs.append(
                        gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    controlnet_inpainting_inputs.append(gr.Slider(1, 10, 1, step=1, label="Num"))
                with gr.Accordion("Code", open=False):
                    controlnet_inpainting_args_name = ",".join([str(item).split("=")[0] for item in list(
                        inspect.signature(fns["controlnet_inpainting_fn"]).parameters.values())])
                    controlnet_inpainting_inputs.append(
                        gr.Code(
                            f"def preprocess({controlnet_inpainting_args_name}):\n  kwargs['callback_on_step_end'] = None\n  return {controlnet_inpainting_args_name.replace('*', '')}",
                            language="python", label="Python"))
                controlnet_inpainting_outputs.append(gr.Textbox(label="Result"))
                controlnet_inpainting_btn = gr.Button("Run")
                controlnet_inpainting_btn.click(fn=fns["controlnet_inpainting_fn"], inputs=controlnet_inpainting_inputs,
                                                outputs=controlnet_inpainting_outputs)

def render_download_file(fns):
    with gr.Accordion("Download File", open=False):
        gr.Markdown("# Download File")
        download_file_inputs = []
        download_file_outputs = []
        with gr.Row():
            with gr.Column():
                download_file_inputs.append(gr.Textbox(placeholder="Give me a url or HF repo id!",label="File URL"))
                download_file_inputs.append(gr.Textbox(placeholder="Where you want to store the file.",label="Directory"))
            with gr.Column():
                download_file_outputs.append(gr.Textbox(label="Result"))
                download_click = gr.Button("Download")
                download_click.click(fns["download_file_fn"],inputs=download_file_inputs,outputs=download_file_outputs,trigger_mode="multiple")

def render_code(fns):
    with gr.Accordion("Code", open=False):
        gr.Markdown("# Code")
        gr.Markdown(f"- GPUs: {GPU_NAME}")
        gr.Markdown(f"- Python: {sys.version}")
        gr.Markdown(f"- OS: {platform.platform()}")
        code_inputs = []
        code_outputs = []
        with gr.Row():
            with gr.Column():
                code_inputs.append(gr.Code(value="_cout = 'Hello world.'", language="python", lines=5, label="Python"))
            with gr.Column():
                code_outputs.append(gr.Textbox(label="Code Result"))
                code_btn = gr.Button("Run Code")
                code_btn.click(fn=fns["run_code_fn"], inputs=code_inputs, outputs=code_outputs)


def render_flux_ui(fns):
    theme = gr.themes.Ocean()

    with gr.Blocks(title=f"FLUX{GPU_NAME}", theme=theme) as server:

        render_model_selection(fns)
        render_lora(fns)
        render_text_to_image(fns)
        render_image_to_image(fns)
        render_inpainting(fns)

        with gr.Accordion("Controlnet",open=False):
            render_controlnet_selection(fns)
            render_controlnet_text_to_image(fns)
            render_controlnet_image_to_image(fns)
            render_controlnet_inpainting(fns)

        with gr.Accordion("Dev Tools", open=False):
            render_download_file(fns)
            render_code(fns)

    return server


def load_flux_ui(pipelines, _globals=None,**kwargs):
    pipelines = [pipelines] if not isinstance(pipelines, list) else pipelines
    pipelines: list = pipelines[:GPU_COUNT]
    if len(pipelines) == 0:
        print("Warning: No available GPU.")

    lock_state = [False, None]

    # the way Gradio pass the arguments to function is based on the position instead of the keyword
    # progress: args[-1], code: args[-2], num: args[-3], seed: args[-4]

    def allow_code_control(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            code = args[-2]
            exec_assets = locals()
            exec(code, _globals, exec_assets)
            preprocess = exec_assets.get("preprocess")
            if callable(preprocess):
                args_and_kwargs = preprocess(*args, **kwargs)
                return f(*args_and_kwargs[:-1], **args_and_kwargs[-1])
            else:
                return f(*args, **kwargs)

        return wrapper

    def auto_load_controlnet(f):
        @functools.wraps(f)
        def wrapper(*args,**kwargs):
            for pipeline in pipelines:
                if "controlnet" not in pipeline.get_list_adapters():
                    pipeline.load_controlnet()
            return f(*args,**kwargs)
        return wrapper

    def auto_offload_controlnet(f):
        @functools.wraps(f)
        def wrapper(*args,**kwargs):
            for pipeline in pipelines:
                if "controlnet" in pipeline.get_list_adapters():
                    pipeline.offload_controlnet()
            return f(*args,**kwargs)
        return wrapper

    def auto_gpu_distribute(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if int(args[-4]) != 0 or len(pipelines) == 1:
                r = f(*args, **kwargs)(pipelines[0])
                free_memory_to_system()
                return r
            else:
                threads_execute(f(*args, **kwargs), pipelines)
                free_memory_to_system()
                return f"{args[-3]} * {len(pipelines)}"

        return wrapper

    def auto_gpu_loop(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return [f(*args, **kwargs)(pipeline) for pipeline in pipelines][0]

        return wrapper

    @allow_return_error
    @lock(lock_state)
    def model_selection_fn(model, progress=gr.Progress(track_tqdm=True)):
        for pipeline in pipelines:
            pipeline.clear()
        for pipeline in pipelines:
            pipeline.reload(model)
            components_in_cpu = False
            for _,component in pipeline.components.items():
                if hasattr(component,"device"):
                    if str(component.device) == "cpu":
                        components_in_cpu = True
            if components_in_cpu:
                i = len(pipelines) - 1 - pipelines.index(pipeline)
                print(f"Loading the model into cuda:{i}...")
                pipeline.to(f"cuda:{i}")
        return f"{model}"

    @allow_return_error
    @lock(lock_state)
    @auto_gpu_loop
    def set_lora_fn(url, lora_name, strength, progress=gr.Progress(track_tqdm=True)):
        def f(pipeline):
            pipeline.set_lora(url, lora_name, strength)
            return f"{lora_name}, {strength}"

        return f

    @allow_return_error
    @lock(lock_state)
    @auto_gpu_loop
    def delete_lora_fn(_, lora_name, __):
        def f(pipeline):
            pipeline.delete_adapters(lora_name)
            return f"{lora_name} is deleted."

        return f

    @allow_return_error
    @auto_gpu_loop
    def show_lora_fn():
        def f(pipeline):
            return str(pipeline.lora_dict)

        return f

    @allow_return_error
    @lock(lock_state)
    @auto_gpu_loop
    def enable_lora_fn():
        def f(pipeline):
            pipeline.enable_lora()
            return f"LoRA Enabled."

        return f

    @allow_return_error
    @lock(lock_state)
    @auto_gpu_loop
    def disable_lora_fn():
        def f(pipeline):
            pipeline.disable_lora()
            return f"LoRA disabled."

        return f

    @allow_return_error
    @lock(lock_state)
    @allow_code_control
    @auto_gpu_distribute
    def text_to_image_fn(
            prompt,
            guidance_scale, num_inference_steps,
            width, height,
            seed, num,
            code,
            progress=gr.Progress(track_tqdm=True),**kwargs):

        def f(pipeline):
            return pipeline.text_to_image_pipeline.generate_image_and_send_to_telegram(
                prompt=prompt,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                width=width, height=height,
                seed=int(seed), num=int(num),**kwargs)
        return f

    @allow_return_error
    @lock(lock_state)
    @allow_code_control
    @auto_gpu_distribute
    def image_to_image_fn(
            image,
            prompt,
            strength,
            guidance_scale, num_inference_steps,
            width,height,
            seed, num,
            code,
            progress=gr.Progress(track_tqdm=True),**kwargs):

        if not image:
            raise ValueError("Please input an image.")

        def f(pipeline):
            return pipeline.image_to_image_pipeline.generate_image_and_send_to_telegram(
                image=image,
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                width=width,height=height,
                seed=int(seed), num=int(num),**kwargs)
        return f

    @allow_return_error
    @lock(lock_state)
    @allow_code_control
    @auto_gpu_distribute
    def inpainting_fn(
            image,
            prompt,
            strength,
            guidance_scale, num_inference_steps,
            width, height,
            seed, num,
            code,
            progress=gr.Progress(track_tqdm=True), **kwargs):

        _image = image["background"].convert("RGB")
        mask_image = convert_mask_image_to_rgb(image["layers"][0])

        def f(pipeline):
            return pipeline.inpainting_pipeline.generate_image_and_send_to_telegram(
                image=_image, mask_image=mask_image,
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                width=width, height=height,
                seed=int(seed), num=int(num), **kwargs)

        return f


    @allow_return_error
    @auto_gpu_loop
    def set_default_controlnet_for_auto_load_controlnet_fn(controlnet_model):
        def f(pipeline):
            pipeline.load_controlnet = functools.partial(pipeline.load_controlnet, controlnet_model=controlnet_model)
            if controlnet_model:
                return f"{controlnet_model} is set as the default controlnet model."
            else:
                return f"Using the default controlnet model."

        return f

    @allow_return_error
    @lock(lock_state)
    @auto_gpu_loop
    def load_controlnet_fn(progress=gr.Progress(track_tqdm=True)):
        def f(pipeline):
            pipeline.load_controlnet()
            return f"Controlnet is loaded."

        return f

    @allow_return_error
    @lock(lock_state)
    @auto_gpu_loop
    def offload_controlnet_fn():
        def f(pipeline):
            pipeline.offload_controlnet()
            return f"Controlnet is offloaded."

        return f

    @allow_return_error
    def controlnet_preview_fn(image, low_threshold, high_threshold):
        if image:
            return convert_image_to_canny(image, low_threshold, high_threshold)


    @allow_return_error
    @lock(lock_state)
    @allow_code_control
    @auto_load_controlnet
    @auto_gpu_distribute
    def controlnet_text_to_image_fn(
            image,
            prompt,
            controlnet_conditioning_scale, guidance_scale, num_inference_steps,
            width, height,
            low_threshold, high_threshold,
            seed, num,
            code,
            progress=gr.Progress(track_tqdm=True), **kwargs):

        if not image:
            raise ValueError("Please input an image.")

        image = convert_image_to_canny(image, low_threshold, high_threshold)

        def f(pipeline):
            pipeline.set_lora_strength("controlnet",controlnet_conditioning_scale)
            return pipeline.text_to_image_controlnet_pipeline.generate_image_and_send_to_telegram(
                control_image=image,
                prompt=prompt,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                width=width, height=height,
                seed=int(seed), num=int(num), **kwargs)

        return f

    @allow_return_error
    @lock(lock_state)
    @allow_code_control
    @auto_load_controlnet
    @auto_gpu_distribute
    def controlnet_image_to_image_fn(
            control_image, image,
            prompt,
            controlnet_conditioning_scale, strength,
            guidance_scale, num_inference_steps,
            width, height,
            low_threshold, high_threshold,
            seed, num,
            code,
            progress=gr.Progress(track_tqdm=True), **kwargs):

        if not image and not control_image:
            raise ValueError("Please input the images.")

        control_image = control_image if control_image else image
        image = image if image else control_image
        control_image = convert_image_to_canny(control_image, low_threshold, high_threshold)

        def f(pipeline):
            pipeline.set_lora_strength("controlnet",controlnet_conditioning_scale)
            return pipeline.image_to_image_controlnet_pipeline.generate_image_and_send_to_telegram(
                control_image=control_image, image=image,
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                width=width, height=height,
                seed=int(seed), num=int(num), **kwargs)
        return f

    @lock(lock_state)
    @allow_code_control
    @auto_load_controlnet
    @auto_gpu_distribute
    def controlnet_inpainting_fn(
            control_image, image,
            prompt,
            controlnet_conditioning_scale, strength,
            guidance_scale, num_inference_steps,
            width, height,
            low_threshold, high_threshold,
            seed, num,
            code,
            progress=gr.Progress(track_tqdm=True), **kwargs):

        _image = image["background"].convert("RGB")
        mask_image = convert_mask_image_to_rgb(image["layers"][0])

        control_image = control_image if control_image else _image
        control_image = convert_image_to_canny(control_image, low_threshold, high_threshold)

        def f(pipeline):
            pipeline.set_lora_strength("controlnet", controlnet_conditioning_scale)
            return pipeline.inpainting_controlnet_pipeline.generate_image_and_send_to_telegram(
                control_image=control_image, image=_image, mask_image=mask_image,
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                width=width, height=height,
                seed=int(seed), num=int(num), **kwargs)

        return f

    @allow_return_error
    def download_file_fn(url, directory, progress=gr.Progress(track_tqdm=True)):
        if url.startswith("http://") or url.startswith("https://"):
            return download_file(url, directory=directory)
        else:
            return download_hf_repo_files(url,directory=directory)

    @allow_return_error
    def run_code_fn(code, progress=gr.Progress(track_tqdm=True)):
        exec(code,_globals)
        if _globals:
            return _globals.pop("_cout", None)

    # import some important packages in ui
    exec("import os,sys,gc,torch\nimport xfusion.enhancement.callbacks as cbk", _globals)

    fns = locals()
    fns.pop("_globals")
    server = render_flux_ui(fns)

    if kwargs.get("inline") is None: kwargs.update(inline=False)
    if kwargs.get("quiet") is None: kwargs.update(quiet=True)
    block = kwargs.pop("debug", True)
    server.launch(**kwargs)

    if server.share_url:
        threads_execute(pipelines[0].send_text, (f"* Running on public URL: {server.share_url}",), _await=False)

    if block:
        def close():
            server.close()
            for pipeline in pipelines: pipeline.clear()

        safe_block(close)

    return server
