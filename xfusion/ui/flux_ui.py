import gradio as gr
from ..utils import allow_return_error


def load_flux_ui(fns,_globals=None):
    text_to_image_fn = fns["text_to_image"]
    image_to_image_fn = fns["image_to_image"]

    @allow_return_error
    def run_code_fn(code):
        exec(code,_globals)
        if _globals:
            return _globals.pop("_cout", None)

    with gr.Blocks(title="Xfusion",theme=gr.themes.Ocean()) as server:

        gr.Markdown("# Text To Image ")
        with gr.Row():
            t2i_inputs = []
            t2i_outputs = []
            with gr.Column():
                t2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!",label="Prompt"))
            with gr.Column():
                t2i_inputs.append(gr.Slider(0,10,2.5,step=0.1,label="Guidance Scale"))
                t2i_inputs.append(gr.Slider(0,50,28,step=1,label="Step"))
                t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
            with gr.Column():
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
                i2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt"))
            with gr.Column():
                i2i_inputs.append(gr.Slider(0, 1, 0.3, step=0.1, label="Strength"))
                i2i_inputs.append(gr.Slider(0, 10, 3, step=0.1, label="Guidance Scale"))
                i2i_inputs.append(gr.Slider(0, 50, 28, step=1, label="Step"))
            with gr.Column():
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
                code_inputs.append(
                    gr.Code(value="_cout = 'Hello world.'", language="python", lines=5, label="Python"))
            with gr.Column():
                code_outputs.append(gr.Textbox(label="Code Result"))
                code_btn = gr.Button("Run Code")
                code_btn.click(fn=run_code_fn, inputs=code_inputs, outputs=code_outputs)

    return server
