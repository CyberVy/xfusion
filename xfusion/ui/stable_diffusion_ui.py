import gradio as gr
from ..utils import allow_return_error,threads_execute
from ..utils import convert_mask_image_to_rgb,convert_image_to_canny
from ..const import GPU_Count,GPU_Name
from ..components.component_const import default_stable_diffusion_model_url
import sys,platform
import functools

scheduler_list = [
            "DPM++ 2M",
            "DPM++ 2M KARRAS",
            "DPM++ 2M SDE",
            "DPM++ 2M SDE KARRAS",
            "DPM++ 2S A",
            "DPM++ 2S A KARRAS",
            "DPM++ SDE",
            "DPM++ SDE KARRAS",
            "DPM2",
            "DPM2 KARRAS",
            "DPM2 A",
            "DPM2 A KARRAS",
            "EULER",
            "EULER A",
            "HEUN",
            "LMS",
            "LMS KARRAS",
            "DEIS",
            "UNIPC"
        ]

def stable_diffusion_ui_template(fns):
    theme = gr.themes.Ocean()

    with gr.Blocks(title=f"Xfusion{GPU_Name}", theme=theme) as server:
        with gr.Accordion("Model Selection", open=True):
            gr.Markdown("# Model Selection")
            model_selection_inputs = []
            model_selection_outputs = []
            with gr.Row():
                with gr.Column():
                    model_selection_inputs.append(gr.Textbox(value=default_stable_diffusion_model_url,placeholder="Give me a url of the model!", label="Model"))
                    model_selection_inputs.append(gr.Textbox(placeholder="Model version", label="Model Version"))
                with gr.Column():
                    model_selection_outputs.append(gr.Textbox(label="Result"))
                    model_selection_btn = gr.Button("Select")
                    model_selection_btn.click(fn=fns["model_selection_fn"], inputs=model_selection_inputs,
                                              outputs=model_selection_outputs)

        with gr.Accordion("LoRA",open=False):
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

        with gr.Accordion("Text To Image", open=True):
            gr.Markdown("# Text To Image")
            t2i_inputs = []
            t2i_outputs = []
            t2i_scheduler_inputs = []
            t2i_scheduler_outputs = []
            with gr.Row():
                with gr.Accordion("Scheduler", open=False):
                    t2i_scheduler_inputs.append(gr.Radio(scheduler_list, label="Scheduler"))
                    t2i_scheduler_outputs.append(gr.Textbox(label="Result"))
                    t2i_scheduler_inputs[0].change(
                        fn=fns["text_to_image_scheduler_fn"],inputs=t2i_scheduler_inputs,outputs=t2i_scheduler_outputs)
            with gr.Row():
                with gr.Column():
                    t2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
                    t2i_inputs.append(
                        gr.Textbox(placeholder="Give me a negative prompt!", label="Negative Prompt", lines=4))
                with gr.Column():
                    t2i_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                    t2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                    t2i_inputs.append(gr.Slider(0, 10, 0, step=1, label="CLIP Skip"))
                    with gr.Row():
                        t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                        t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
                with gr.Column():
                    with gr.Row():
                        t2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                        t2i_inputs.append(gr.Slider(1,10,1, step=1,label="Num"))
                    t2i_outputs.append(gr.Textbox(label="Result"))
                    t2i_btn = gr.Button("Run")
                    t2i_btn.click(fn=fns["text_to_image_fn"], inputs=t2i_inputs, outputs=t2i_outputs)

        with gr.Accordion("Image To Image", open=False):
            gr.Markdown("# Image To Image")
            i2i_inputs = []
            i2i_outputs = []
            i2i_scheduler_inputs = []
            i2i_scheduler_outputs = []
            with gr.Row():
                with gr.Accordion("Scheduler", open=False):
                    i2i_scheduler_inputs.append(gr.Radio(scheduler_list, label="Scheduler"))
                    i2i_scheduler_outputs.append(gr.Textbox(label="Result"))
                    i2i_scheduler_inputs[0].change(
                        fn=fns["image_to_image_scheduler_fn"], inputs=i2i_scheduler_inputs,outputs=i2i_scheduler_outputs)
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Image"):
                        i2i_inputs.append(gr.Image(type="pil", label="Image"))
                    i2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
                    i2i_inputs.append(
                        gr.Textbox(placeholder="Give me a negative prompt!", label="Negative Prompt", lines=4))
                with gr.Column():
                    i2i_inputs.append(gr.Slider(0, 1, 0.4, step=0.1, label="Strength"))
                    i2i_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                    i2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                    i2i_inputs.append(gr.Slider(0, 10, 0, step=1, label="CLIP Skip"))
                    with gr.Row():
                        i2i_inputs.append(gr.Slider(512,2048,1024,step=8,label="Width"))
                        i2i_inputs.append(gr.Slider(512,2048,1024,step=8,label="Height"))
                with gr.Column():
                    with gr.Row():
                        i2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                        i2i_inputs.append(gr.Slider(1,10,1, step=1,label="Num"))
                    i2i_outputs.append(gr.Textbox(label="Result"))
                    i2i_btn = gr.Button("Run")
                    i2i_btn.click(fn=fns["image_to_image_fn"], inputs=i2i_inputs, outputs=i2i_outputs)

        with gr.Accordion("Inpainting",open=False):
            gr.Markdown("# Inpainting")
            inpainting_inputs = []
            inpainting_outputs = []
            inpainting_scheduler_inputs = []
            inpainting_scheduler_outputs = []
            with gr.Row():
                with gr.Accordion("Scheduler", open=False):
                    inpainting_scheduler_inputs.append(gr.Radio(scheduler_list, label="Scheduler"))
                    inpainting_scheduler_outputs.append(gr.Textbox(label="Result"))
                    inpainting_scheduler_inputs[0].change(
                        fn=fns["inpainting_scheduler_fn"], inputs=inpainting_scheduler_inputs,outputs=inpainting_scheduler_outputs)
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Inpainting Image"):
                        inpainting_inputs.append(gr.ImageMask(type="pil", label="Inpainting Image"))
                    inpainting_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
                    inpainting_inputs.append(
                        gr.Textbox(placeholder="Give me a negative prompt!", label="Negative Prompt", lines=4))
                with gr.Column():
                    inpainting_inputs.append(gr.Slider(0, 1, 0.8, step=0.1, label="Strength"))
                    inpainting_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                    inpainting_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                    inpainting_inputs.append(gr.Slider(0, 10, 0, step=1, label="CLIP Skip"))
                    with gr.Row():
                        inpainting_inputs.append(gr.Slider(512,2048,1024,step=8,label="Width"))
                        inpainting_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
                with gr.Column():
                    with gr.Row():
                        inpainting_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                        inpainting_inputs.append(gr.Slider(1,10,1, step=1,label="Num"))
                    inpainting_outputs.append(gr.Textbox(label="Result"))
                    inpainting_btn = gr.Button("Run")
                    inpainting_btn.click(fn=fns["inpainting_fn"], inputs=inpainting_inputs, outputs=inpainting_outputs)

        with gr.Accordion("Controlnet",open=False):
            controlnet_inputs = []
            controlnet_outputs = []
            with gr.Accordion("Controlnet Selection",open=False):
                with gr.Row():
                    with gr.Column():
                        controlnet_inputs.append(gr.Textbox(placeholder="Give me a controlnet URL!",label="Controlnet Model"))
                        controlnet_inputs[0].change(fn=fns["set_default_controlnet_for_auto_load_controlnet_fn"],inputs=controlnet_inputs[0],outputs=controlnet_outputs)
                        with gr.Row():
                            load_controlnet_button = gr.Button("Load controlnet")
                            offload_controlnet_button = gr.Button("Offload controlnet")
                    controlnet_outputs.append(gr.Textbox(label="Result"))
                    load_controlnet_button.click(fn=fns["load_controlnet_fn"],inputs=controlnet_inputs,outputs=controlnet_outputs)
                    offload_controlnet_button.click(fn=fns["offload_controlnet_fn"],outputs=controlnet_outputs)
            with gr.Accordion("Controlnet Text To Image", open=False):
                gr.Markdown("# Controlnet Text To Image")
                controlnet_t2i_inputs = []
                controlnet_t2i_outputs = []
                controlnet_t2i_control_image_preview_inputs = []
                controlnet_t2i_control_image_preview_outputs = []
                controlnet_t2i_scheduler_inputs = []
                controlnet_t2i_scheduler_outputs = []
                with gr.Row():
                    with gr.Accordion("Scheduler", open=False):
                        controlnet_t2i_scheduler_inputs.append(gr.Radio(scheduler_list, label="Scheduler"))
                        controlnet_t2i_scheduler_outputs.append(gr.Textbox(label="Result"))
                        controlnet_t2i_scheduler_inputs[0].change(
                            fn=fns["controlnet_text_to_image_scheduler_fn"], inputs=controlnet_t2i_scheduler_inputs,outputs=controlnet_t2i_scheduler_outputs)
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Controlnet Image"):
                            controlnet_t2i_inputs.append(gr.Image(type="pil", label="Controlnet Image"))
                        controlnet_t2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
                        controlnet_t2i_inputs.append(
                            gr.Textbox(placeholder="Give me a negative prompt!", label="Negative Prompt", lines=4))
                    with gr.Column():
                        controlnet_t2i_inputs.append(gr.Slider(0, 1, 0.5, step=0.05, label="Controlnet Scale"))
                        controlnet_t2i_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                        controlnet_t2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                        controlnet_t2i_inputs.append(gr.Slider(0, 10, 0, step=1, label="CLIP Skip"))
                        with gr.Row():
                            controlnet_t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                            controlnet_t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
                        with gr.Row():
                            controlnet_t2i_control_image_preview_inputs.append(controlnet_t2i_inputs[0]) # the control image
                            controlnet_t2i_control_image_preview_inputs.append(gr.Slider(0, 255, 100, step=5, label="Low Threshold"))
                            controlnet_t2i_control_image_preview_inputs.append(gr.Slider(0, 255, 200, step=5, label="High Threshold"))
                        controlnet_t2i_control_image_preview_outputs.append(gr.Image(label="Control Image Preview"))
                        for component in controlnet_t2i_control_image_preview_inputs:
                            component.change(fn=fns["controlnet_preview_fn"],inputs=controlnet_t2i_control_image_preview_inputs,outputs=controlnet_t2i_control_image_preview_outputs)
                    with gr.Column():
                        with gr.Row():
                            controlnet_t2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                            controlnet_t2i_inputs.append(gr.Slider(1,10,1, step=1,label="Num"))
                        controlnet_t2i_outputs.append(gr.Textbox(label="Result"))
                        controlnet_t2i_btn = gr.Button("Run")
                        controlnet_t2i_btn.click(fn=fns["controlnet_text_to_image_fn"], inputs=controlnet_t2i_inputs, outputs=controlnet_t2i_outputs)
            with gr.Accordion("Controlnet Image To Image",open=False):
                gr.Markdown("# Controlnet Image To Image")
                controlnet_i2i_inputs = []
                controlnet_i2i_outputs = []
                controlnet_i2i_scheduler_inputs = []
                controlnet_i2i_scheduler_outputs = []
                with gr.Row():
                    with gr.Accordion("Scheduler", open=False):
                        controlnet_i2i_scheduler_inputs.append(gr.Radio(scheduler_list, label="Scheduler"))
                        controlnet_i2i_scheduler_outputs.append(gr.Textbox(label="Result"))
                        controlnet_i2i_scheduler_inputs[0].change(
                            fn=fns["controlnet_image_to_image_scheduler_fn"], inputs=controlnet_i2i_scheduler_inputs,
                            outputs=controlnet_i2i_scheduler_outputs)
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Images"):
                            with gr.Row():
                                controlnet_i2i_inputs.append(gr.Image(type="pil", label="Controlnet Image"))
                                controlnet_i2i_inputs.append(gr.Image(type="pil", label="Image"))
                        controlnet_i2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt", lines=5))
                        controlnet_i2i_inputs.append(
                            gr.Textbox(placeholder="Give me a negative prompt!", label="Negative Prompt", lines=4))
                    with gr.Column():
                        controlnet_i2i_inputs.append(gr.Slider(0, 1, 0.5, step=0.05, label="Controlnet Scale"))
                        controlnet_i2i_inputs.append(gr.Slider(0, 1, 0.4, step=0.1, label="Strength"))
                        controlnet_i2i_inputs.append(gr.Slider(0, 10, 2.5, step=0.1, label="Guidance Scale"))
                        controlnet_i2i_inputs.append(gr.Slider(0, 50, 20, step=1, label="Step"))
                        controlnet_i2i_inputs.append(gr.Slider(0, 10, 0, step=1, label="CLIP Skip"))
                        with gr.Row():
                            controlnet_i2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                            controlnet_i2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
                    with gr.Column():
                        with gr.Row():
                            controlnet_i2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                            controlnet_i2i_inputs.append(gr.Slider(1, 10, 1, step=1, label="Num"))
                        controlnet_i2i_outputs.append(gr.Textbox(label="Result"))
                        controlnet_i2i_btn = gr.Button("Run")
                        controlnet_i2i_btn.click(fn=fns["controlnet_image_to_image_fn"], inputs=controlnet_i2i_inputs, outputs=controlnet_i2i_outputs)

        with gr.Accordion("Code",open=False):
            gr.Markdown("# Code")
            gr.Markdown(f"- GPUs: {GPU_Name}")
            gr.Markdown(f"- Python: {sys.version}")
            gr.Markdown(f"- OS: {platform.platform()}")
            code_inputs = []
            code_outputs = []
            with gr.Row():
                with gr.Column():
                    code_inputs.append(gr.Code(value="import os,sys,gc,torch\n_cout = 'Hello world.'", language="python", lines=5, label="Python"))
                with gr.Column():
                    code_outputs.append(gr.Textbox(label="Code Result"))
                    code_btn = gr.Button("Run Code")
                    code_btn.click(fn=fns["run_code_fn"], inputs=code_inputs, outputs=code_outputs)
    return server

def load_stable_diffusion_ui(pipeline, _globals=None):

    def auto_load_controlnet(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if pipeline._controlnet is None:
                pipeline.load_controlnet()
            return f(*args, **kwargs)
        return wrapper

    def auto_offload_controlnet(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if pipeline._controlnet is not None:
                pipeline.offload_controlnet()
            return f(*args, **kwargs)

        return wrapper

    @allow_return_error
    def model_selection_fn(model,model_version):
        pipeline.reload(model, model_version=model_version)
        if str(pipeline.device) == "cpu":
            print("Loading the model into cuda...")
            pipeline.to("cuda")
        return f"{model}, {model_version}"

    @allow_return_error
    def set_lora_fn(url, lora_name, strength):
        pipeline.set_lora(url, lora_name, strength)
        return f"{lora_name}, {strength}"

    @allow_return_error
    def delete_lora_fn(_,lora_name,__):
        pipeline.delete_adapters(lora_name)
        return f"{lora_name} is deleted."

    @allow_return_error
    def show_lora_fn():
        return f"{pipeline.lora_dict}"

    @allow_return_error
    def enable_lora_fn():
        pipeline.enable_lora()
        return f"LoRA Enabled."

    @allow_return_error
    def disable_lora_fn():
        pipeline.disable_lora()
        return f"LoRA disabled."

    @allow_return_error
    def text_to_image_scheduler_fn(scheduler):
        pipeline.text_to_image_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for text to image pipeline."

    @allow_return_error
    @auto_offload_controlnet
    def text_to_image_fn(
            prompt, negative_prompt,
            guidance_scale, num_inference_steps, clip_skip,
            width, height,
            seed, num):

        return pipeline.text_to_image_pipeline.generate_image_and_send_to_telegram(
            prompt=prompt, negative_prompt=negative_prompt,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
            width=width, height=height,
            seed=int(seed), num=int(num))

    @allow_return_error
    def image_to_image_scheduler_fn(scheduler):
        pipeline.image_to_image_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for image to image pipeline."

    @allow_return_error
    @auto_offload_controlnet
    def image_to_image_fn(
            image,
            prompt, negative_prompt,
            strength,
            guidance_scale, num_inference_steps, clip_skip,
            width,height,
            seed, num):

        if not image:
            raise ValueError("Please input an image.")

        return pipeline.image_to_image_pipeline.generate_image_and_send_to_telegram(
            image=image,
            prompt=prompt, negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
            width=width,height=height,
            seed=int(seed), num=int(num))

    @allow_return_error
    def inpainting_scheduler_fn(scheduler):
        pipeline.inpainting_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for inpainting pipeline."

    @allow_return_error
    @auto_offload_controlnet
    def inpainting_fn(
            image,
            prompt, negative_prompt,
            strength,
            guidance_scale, num_inference_steps, clip_skip,
            width,height,
            seed, num):
        return pipeline.inpainting_pipeline.generate_image_and_send_to_telegram(
            image=image["background"].convert("RGB"),
            mask_image=convert_mask_image_to_rgb(image["layers"][0]),
            prompt=prompt, negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
            width=width,height=height,
            seed=int(seed), num=int(num))

    @allow_return_error
    def set_default_controlnet_for_auto_load_controlnet_fn(controlnet_model):
        pipeline.load_controlnet = functools.partial(pipeline.load_controlnet,controlnet_model=controlnet_model)
        if controlnet_model:
            return f"{controlnet_model} is set as the default controlnet model."
        else:
            return f"Using the default controlnet model."

    @allow_return_error
    def load_controlnet_fn(controlnet_model):
        pipeline.load_controlnet(controlnet_model=controlnet_model)
        return f"Controlnet is loaded."

    @allow_return_error
    def offload_controlnet_fn():
        pipeline.offload_controlnet()
        return f"Controlnet is offloaded."

    @allow_return_error
    def controlnet_preview_fn(image,low_threshold,high_threshold):
        return convert_image_to_canny(image,low_threshold,high_threshold)

    @allow_return_error
    def controlnet_text_to_image_scheduler_fn(scheduler):
        pipeline.text_to_image_controlnet_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for text to image controlnet pipeline."

    @allow_return_error
    @auto_load_controlnet
    def controlnet_text_to_image_fn(
            image,
            prompt, negative_prompt,
            controlnet_conditioning_scale,guidance_scale, num_inference_steps, clip_skip,
            width, height,
            seed, num):

        if not image:
            raise ValueError("Please input an image.")

        return pipeline.text_to_image_controlnet_pipeline.generate_image_and_send_to_telegram(
            image=convert_image_to_canny(image),
            prompt=prompt, negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
            width=width, height=height,
            seed=int(seed), num=int(num)
        )

    @allow_return_error
    def controlnet_image_to_image_scheduler_fn(scheduler):
        pipeline.image_to_image_controlnet_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for image to image controlnet pipeline."

    @allow_return_error
    @auto_load_controlnet
    def controlnet_image_to_image_fn(
            control_image,image,
            prompt, negative_prompt,
            controlnet_conditioning_scale,strength,
            guidance_scale, num_inference_steps, clip_skip,
            width, height,
            seed, num):

        if not image or not control_image:
            raise ValueError("Please input the images.")

        return pipeline.image_to_image_controlnet_pipeline.generate_image_and_send_to_telegram(
            control_image=convert_image_to_canny(control_image),image=image,
            prompt=prompt, negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,strength=strength,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
            width=width, height=height,
            seed=int(seed), num=int(num)
        )

    @allow_return_error
    def run_code_fn(code):
        exec(code,_globals)
        if _globals:
            return _globals.pop("_cout", None)

    fns = {"model_selection_fn": model_selection_fn,
           "set_lora_fn": set_lora_fn, "delete_lora_fn": delete_lora_fn,
           "show_lora_fn": show_lora_fn, "enable_lora_fn": enable_lora_fn,
           "disable_lora_fn": disable_lora_fn,
           "text_to_image_scheduler_fn": text_to_image_scheduler_fn,
           "text_to_image_fn": text_to_image_fn,
           "image_to_image_scheduler_fn": image_to_image_scheduler_fn,
           "image_to_image_fn": image_to_image_fn,
           "inpainting_scheduler_fn": inpainting_scheduler_fn,
           "inpainting_fn": inpainting_fn,
           "set_default_controlnet_for_auto_load_controlnet_fn": set_default_controlnet_for_auto_load_controlnet_fn,
           "load_controlnet_fn": load_controlnet_fn,
           "offload_controlnet_fn": offload_controlnet_fn,
           "controlnet_preview_fn":controlnet_preview_fn,
           "controlnet_text_to_image_scheduler_fn": controlnet_text_to_image_scheduler_fn,
           "controlnet_text_to_image_fn": controlnet_text_to_image_fn,
           "controlnet_image_to_image_scheduler_fn": controlnet_image_to_image_scheduler_fn,
           "controlnet_image_to_image_fn": controlnet_image_to_image_fn,
           "run_code_fn": run_code_fn}

    return stable_diffusion_ui_template(fns)

def load_stable_diffusion_ui_for_multiple_pipelines(pipelines, _globals=None):
    """
    load pipelines to multiple GPUs for acceleration
    """
    pipelines = pipelines[:GPU_Count]

    def auto_load_controlnet(f):
        @functools.wraps(f)
        def wrapper(pipeline):
            if pipeline._controlnet is None:
                pipeline.load_controlnet()
            return f(pipeline)
        return wrapper

    def auto_offload_controlnet(f):
        @functools.wraps(f)
        def wrapper(pipeline):
            if pipeline._controlnet is not None:
                pipeline.offload_controlnet()
            return f(pipeline)
        return wrapper

    @allow_return_error
    def model_selection_fn(model,model_version):

        for i,pipeline in enumerate(pipelines):
            pipeline.reload(model,model_version=model_version)
            if str(pipeline.device) == "cpu":
                print(f"Loading the model into cuda:{i}...")
                pipeline.to(f"cuda:{i}")

        return f"{model}, {model_version}"

    @allow_return_error
    def set_lora_fn(url, lora_name, strength):
        for pipeline in pipelines:
            pipeline.set_lora(url,lora_name,strength)
        return f"{lora_name}, {strength}"

    @allow_return_error
    def delete_lora_fn(_,lora_name,__):
        for pipeline in pipelines:
            pipeline.delete_adapters(lora_name)
        return f"{lora_name} is deleted."

    @allow_return_error
    def show_lora_fn():
        r = ""
        for pipeline in pipelines:
            r += str(pipeline.lora_dict) + " "
        return f"{r}"

    @allow_return_error
    def enable_lora_fn():
        for pipeline in pipelines:
            pipeline.enable_lora()
        return f"LoRA Enabled."

    @allow_return_error
    def disable_lora_fn():
        for pipeline in pipelines:
            pipeline.disable_lora()
        return f"LoRA disabled."

    @allow_return_error
    def text_to_image_scheduler_fn(scheduler):
        for pipeline in pipelines:
            pipeline.text_to_image_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for text to image pipeline."

    @allow_return_error
    def text_to_image_fn(
            prompt, negative_prompt,
            guidance_scale, num_inference_steps, clip_skip,
            width, height,
            seed, num):
        @auto_offload_controlnet
        def f(pipeline):
            return pipeline.text_to_image_pipeline.generate_image_and_send_to_telegram(
                prompt=prompt, negative_prompt=negative_prompt,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
                width=width, height=height,
                seed=int(seed), num=int(num))
        if int(seed) != 0:
            return f(pipelines[0])
        else:
            threads_execute(f, pipelines)
            return f"{num} * {len(pipelines)}"

    @allow_return_error
    def image_to_image_scheduler_fn(scheduler):
        for pipeline in pipelines:
            pipeline.image_to_image_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for image to image pipeline."

    @allow_return_error
    def image_to_image_fn(
            image,
            prompt, negative_prompt,
            strength,
            guidance_scale, num_inference_steps, clip_skip,
            width,height,
            seed, num):

        if not image:
            raise ValueError("Please input an image.")

        @auto_offload_controlnet
        def f(pipeline):
            return pipeline.image_to_image_pipeline.generate_image_and_send_to_telegram(
                image=image,
                prompt=prompt, negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
                width=width,height=height,
                seed=int(seed), num=int(num))
        if int(seed) != 0:
            return f(pipelines[0])
        else:
            threads_execute(f, pipelines)
            return f"{num} * {len(pipelines)}"

    @allow_return_error
    def inpainting_scheduler_fn(scheduler):
        for pipeline in pipelines:
            pipeline.inpainting_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for inpainting pipeline."

    @allow_return_error
    def inpainting_fn(
            image,
            prompt, negative_prompt,
            strength,
            guidance_scale, num_inference_steps, clip_skip,
            width, height,
            seed, num):
        @auto_offload_controlnet
        def f(pipeline):
            return pipeline.inpainting_pipeline.generate_image_and_send_to_telegram(
                image=image["background"].convert("RGB"),
                mask_image=convert_mask_image_to_rgb(image["layers"][0]),
                prompt=prompt, negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
                width=width, height=height,
                seed=int(seed), num=int(num))
        if int(seed) != 0:
            return f(pipelines[0])
        else:
            threads_execute(f, pipelines)
            return f"{num} * {len(pipelines)}"

    @allow_return_error
    def set_default_controlnet_for_auto_load_controlnet_fn(controlnet_model):
        for pipeline in pipelines:
            pipeline.load_controlnet = functools.partial(pipeline.load_controlnet, controlnet_model=controlnet_model)
        if controlnet_model:
            return f"{controlnet_model} is set as the default controlnet model."
        else:
            return f"Using the default controlnet model."

    @allow_return_error
    def load_controlnet_fn(controlnet_model):
        for pipeline in pipelines:
            pipeline.load_controlnet(controlnet_model)
        return f"Controlnet is loaded."

    @allow_return_error
    def offload_controlnet_fn():
        for pipeline in pipelines:
            pipeline.offload_controlnet()
        return f"Controlnet is offloaded."

    @allow_return_error
    def controlnet_preview_fn(image, low_threshold, high_threshold):
        return convert_image_to_canny(image, low_threshold, high_threshold)

    @allow_return_error
    def controlnet_text_to_image_scheduler_fn(scheduler):
        for pipeline in pipelines:
            pipeline.text_to_image_controlnet_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for text to image controlnet pipeline."

    @allow_return_error
    def controlnet_text_to_image_fn(
            image,
            prompt, negative_prompt,
            controlnet_conditioning_scale,guidance_scale, num_inference_steps, clip_skip,
            width, height,
            seed, num):

        if not image:
            raise ValueError("Please input an image.")

        @auto_load_controlnet
        def f(pipeline):
            return pipeline.text_to_image_controlnet_pipeline.generate_image_and_send_to_telegram(
                image=convert_image_to_canny(image),
                prompt=prompt, negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
                width=width, height=height,
                seed=int(seed), num=int(num))
        if int(seed) != 0:
            return f(pipelines[0])
        else:
            threads_execute(f, pipelines)
            return f"{num} * {len(pipelines)}"

    @allow_return_error
    def controlnet_image_to_image_scheduler_fn(scheduler):
        for pipeline in pipelines:
            pipeline.image_to_image_controlnet_pipeline.set_scheduler(scheduler)
        return f"{scheduler} is set for image to image controlnet pipeline."

    @allow_return_error
    def controlnet_image_to_image_fn(
            control_image,image,
            prompt, negative_prompt,
            controlnet_conditioning_scale, strength,
            guidance_scale, num_inference_steps, clip_skip,
            width, height,
            seed, num):

        if not image or not control_image:
            raise ValueError("Please input the images.")

        @auto_load_controlnet
        def f(pipeline):
            return pipeline.image_to_image_controlnet_pipeline.generate_image_and_send_to_telegram(
                control_image=convert_image_to_canny(control_image),image=image,
                prompt=prompt, negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,strength=strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
                width=width, height=height,
                seed=int(seed), num=int(num))
        if int(seed) != 0:
            return f(pipelines[0])
        else:
            threads_execute(f, pipelines)
            return f"{num} * {len(pipelines)}"


    @allow_return_error
    def run_code_fn(code):
        exec(code,_globals)
        if _globals:
            return _globals.pop("_cout", None)

    fns = {"model_selection_fn": model_selection_fn,
           "set_lora_fn": set_lora_fn, "delete_lora_fn": delete_lora_fn,
           "show_lora_fn": show_lora_fn, "enable_lora_fn": enable_lora_fn,
           "disable_lora_fn": disable_lora_fn,
           "text_to_image_scheduler_fn": text_to_image_scheduler_fn,
           "text_to_image_fn": text_to_image_fn,
           "image_to_image_scheduler_fn": image_to_image_scheduler_fn,
           "image_to_image_fn": image_to_image_fn,
           "inpainting_scheduler_fn": inpainting_scheduler_fn,
           "inpainting_fn": inpainting_fn,
           "set_default_controlnet_for_auto_load_controlnet_fn": set_default_controlnet_for_auto_load_controlnet_fn,
           "load_controlnet_fn":load_controlnet_fn,
           "offload_controlnet_fn":offload_controlnet_fn,
           "controlnet_preview_fn":controlnet_preview_fn,
           "controlnet_text_to_image_scheduler_fn":controlnet_text_to_image_scheduler_fn,
           "controlnet_text_to_image_fn":controlnet_text_to_image_fn,
           "controlnet_image_to_image_scheduler_fn":controlnet_image_to_image_scheduler_fn,
           "controlnet_image_to_image_fn":controlnet_image_to_image_fn,
           "run_code_fn": run_code_fn}

    return stable_diffusion_ui_template(fns)
