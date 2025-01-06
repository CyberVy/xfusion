import gradio as gr
from ..utils import allow_return_error,threads_execute
from ..utils import convert_mask_image_to_rgb
from ..const import GPU_Count
from ..components.component_const import default_stable_diffusion_model_url


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

    with gr.Blocks(title="Xfusion", theme=theme) as server:
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
                with gr.Column():
                    t2i_scheduler_outputs.append(gr.Textbox(label="Result"))
                    t2i_scheduler_btn = gr.Button("Set Scheduler")
                    t2i_scheduler_btn.click(fn=fns["text_to_image_scheduler_fn"], inputs=t2i_scheduler_inputs,
                                            outputs=t2i_scheduler_outputs)
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
                with gr.Column():
                    i2i_scheduler_outputs.append(gr.Textbox(label="Result"))
                    i2i_scheduler_btn = gr.Button("Set Scheduler")
                    i2i_scheduler_btn.click(fn=fns["image_to_image_scheduler_fn"], inputs=i2i_scheduler_inputs,
                                            outputs=i2i_scheduler_outputs)
            with gr.Row():
                with gr.Column():
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
                with gr.Column():
                    inpainting_scheduler_outputs.append(gr.Textbox(label="Result"))
                    inpainting_scheduler_btn = gr.Button("Set Scheduler")
                    inpainting_scheduler_btn.click(fn=fns["inpainting_scheduler_fn"], inputs=inpainting_scheduler_inputs,
                                                   outputs=inpainting_scheduler_outputs)
            with gr.Row():
                with gr.Column():
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

        with gr.Accordion("Code",open=False):
            gr.Markdown("# Code")
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
    def run_code_fn(code):
        exec(code,_globals)
        if _globals:
            return _globals.pop("_cout", None)

    fns = {"model_selection_fn":model_selection_fn,
           "set_lora_fn":set_lora_fn,"delete_lora_fn":delete_lora_fn,
           "show_lora_fn":show_lora_fn,"enable_lora_fn":enable_lora_fn,
           "disable_lora_fn":disable_lora_fn,"text_to_image_scheduler_fn":text_to_image_scheduler_fn,
           "text_to_image_fn":text_to_image_fn,"image_to_image_scheduler_fn":image_to_image_scheduler_fn,
           "image_to_image_fn":image_to_image_fn,"inpainting_scheduler_fn":inpainting_scheduler_fn,
           "inpainting_fn":inpainting_fn,"run_code_fn":run_code_fn}

    return stable_diffusion_ui_template(fns)

def load_stable_diffusion_ui_for_multiple_pipelines(pipelines, _globals=None):
    """
    load pipelines to multiple GPUs for acceleration
    """
    pipelines = pipelines[:GPU_Count]
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
    def run_code_fn(code):
        exec(code,_globals)
        if _globals:
            return _globals.pop("_cout", None)

    fns = {"model_selection_fn": model_selection_fn,
           "set_lora_fn": set_lora_fn, "delete_lora_fn": delete_lora_fn,
           "show_lora_fn": show_lora_fn, "enable_lora_fn": enable_lora_fn,
           "disable_lora_fn": disable_lora_fn, "text_to_image_scheduler_fn": text_to_image_scheduler_fn,
           "text_to_image_fn": text_to_image_fn, "image_to_image_scheduler_fn": image_to_image_scheduler_fn,
           "image_to_image_fn": image_to_image_fn, "inpainting_scheduler_fn": inpainting_scheduler_fn,
           "inpainting_fn": inpainting_fn, "run_code_fn": run_code_fn}

    return stable_diffusion_ui_template(fns)
