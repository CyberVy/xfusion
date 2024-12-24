import gradio as gr
from ..utils import allow_return_error
from PIL import Image


def load_stable_diffusion_ui(pipeline, _globals=None):

    @allow_return_error
    def model_selection_fn(model,model_version):
        pipeline.reload(model, model_version=model_version)
        return f"{model}, {model_version}"

    @allow_return_error
    def lora_fn(url, lora_name, strength):
        pipeline.set_lora(url, lora_name, strength)
        return f"{lora_name}, {strength}"

    @allow_return_error
    def text_to_image_fn(
            prompt, negative_prompt="",
            guidance_scale=2, num_inference_steps=28, clip_skip=0,
            width=None, height=None,
            seed=None, num=1):
        return pipeline.text_to_image_pipeline.generate_image_and_send_to_telegram(
            prompt=prompt, negative_prompt=negative_prompt,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
            width=width, height=height,
            seed=int(seed), num=int(num))

    @allow_return_error
    def image_to_image_fn(
            image,
            prompt, negative_prompt="",
            strength=0.3,
            guidance_scale=2, num_inference_steps=28, clip_skip=0,
            seed=None, num=1):
        image = Image.fromarray(image)
        return pipeline.image_to_image_pipeline.generate_image_and_send_to_telegram(
            image=image,
            prompt=prompt, negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, clip_skip=clip_skip,
            seed=int(seed), num=int(num))


    @allow_return_error
    def run_code_fn(code):
        exec(code,_globals)
        if _globals:
            return _globals.pop("_cout", None)

    with gr.Blocks(title="Xfusion",theme=gr.themes.Ocean()) as server:

        gr.Markdown("# Model Selection")
        with gr.Row():
            model_selection_inputs = []
            model_selection_outputs = []
            with gr.Column():
                model_selection_inputs.append(gr.Textbox(placeholder="Give me a url of the model!",label="Model"))
                model_selection_inputs.append(gr.Textbox(placeholder="Model version", label="Model Version"))
            with gr.Column():
                model_selection_outputs.append(gr.Textbox(label="Result"))
                model_selection_btn = gr.Button("Select")
                model_selection_btn.click(fn=model_selection_fn,inputs=model_selection_inputs,outputs=model_selection_outputs)
        gr.Markdown("---")

        gr.Markdown("# LoRA")
        with gr.Row():
            lora_inputs = []
            lora_outputs = []
            with gr.Column():
                lora_inputs.append(gr.Textbox(placeholder="Give me a url of LoRA!",label="LoRA"))
                with gr.Row():
                    lora_inputs.append(gr.Textbox(placeholder="Give the LoRA a name!",label="LoRA name"))
                    lora_inputs.append(gr.Slider(0,1,0.4,step=0.05,label="LoRA strength"))
            with gr.Column():
                lora_outputs.append(gr.Textbox(label="Result"))
                lora_btn = gr.Button("Set")
                lora_btn.click(fn=lora_fn,inputs=lora_inputs,outputs=lora_outputs)
        gr.Markdown("---")

        gr.Markdown("# Text To Image")
        with gr.Row():
            t2i_inputs = []
            t2i_outputs = []
            with gr.Column():
                t2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!",label="Prompt",lines=5))
                t2i_inputs.append(gr.Textbox(placeholder="Give me a negative prompt!",label="Negative Prompt",lines=4))
            with gr.Column():
                t2i_inputs.append(gr.Slider(0,10,2.5,step=0.1,label="Guidance Scale"))
                t2i_inputs.append(gr.Slider(0,50,28,step=1,label="Step"))
                t2i_inputs.append(gr.Slider(0, 10, 0, step=1, label="CLIP Skip"))
                with gr.Row():
                    t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                    t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
            with gr.Column():
                with gr.Row():
                    t2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    t2i_inputs.append(gr.Textbox(value="1", placeholder="Amount of the pictures.", label="Num"))
                t2i_outputs.append(gr.Textbox(label="Result"))
                t2i_btn = gr.Button("Run")
                t2i_btn.click(fn=text_to_image_fn, inputs=t2i_inputs, outputs=t2i_outputs)
        gr.Markdown("---")

        gr.Markdown("# Image To Image")
        with gr.Row():
            i2i_inputs = []
            i2i_outputs = []
            with gr.Column():
                i2i_inputs.append(gr.Image())
                i2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt",lines=5))
                i2i_inputs.append(gr.Textbox(placeholder="Give me a negative prompt!",label="Negative Prompt",lines=4))
            with gr.Column():
                i2i_inputs.append(gr.Slider(0, 1, 0.3, step=0.1, label="Strength"))
                i2i_inputs.append(gr.Slider(0, 10, 3, step=0.1, label="Guidance Scale"))
                i2i_inputs.append(gr.Slider(0, 50, 28, step=1, label="Step"))
                i2i_inputs.append(gr.Slider(0, 10, 0, step=1, label="CLIP Skip"))
            with gr.Column():
                with gr.Row():
                    i2i_inputs.append(gr.Textbox(value="0", placeholder="Give me an integer.", label="Seed"))
                    i2i_inputs.append(gr.Textbox(value="1", placeholder="Amount of the pictures.", label="Num"))
                i2i_outputs.append(gr.Textbox(label="Result"))
                i2i_btn = gr.Button("Run")
                i2i_btn.click(fn=image_to_image_fn, inputs=i2i_inputs, outputs=i2i_outputs)
        gr.Markdown("---")

        gr.Markdown("# Code")
        with gr.Row():
            code_inputs = []
            code_outputs = []
            with gr.Column():
                code_inputs.append(gr.Code(value="_cout = 'Hello world.'",language="python",lines=5,label="Python"))
            with gr.Column():
                code_outputs.append(gr.Textbox(label="Code Result"))
                code_btn = gr.Button("Run Code")
                code_btn.click(fn=run_code_fn,inputs=code_inputs,outputs=code_outputs)
    return server
