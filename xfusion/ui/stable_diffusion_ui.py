import gradio as gr


def load_stable_diffusion_ui(fns):
    t2i_fn = fns["t2i"]
    i2i_fn = fns["i2i"]

    with gr.Blocks(title="Xfusion",theme=gr.themes.Ocean()) as server:
        gr.Markdown("**Text To Image**")
        with gr.Row():
            t2i_inputs = []
            t2i_outputs = []
            with gr.Column():
                t2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!",label="Prompt"))
                t2i_inputs.append(gr.Textbox(placeholder="Give me a negative prompt!",label="Negative Prompt"))
                t2i_inputs.append(gr.Slider(0,10,2,step=0.1,label="Guidance Scale"))
                t2i_inputs.append(gr.Slider(0,50,28,step=1,label="Step"))
                t2i_inputs.append(gr.Slider(0, 0, 10, step=1, label="CLIP"))
                t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                t2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
            with gr.Column():
                t2i_outputs.append(gr.Textbox())
                t2i_btn = gr.Button("Run")
                t2i_btn.click(fn=t2i_fn, inputs=t2i_inputs, outputs=t2i_outputs)

        gr.Markdown("**Image To Image**")
        i2i_inputs = []
        i2i_outputs = []
        with gr.Row():
            with gr.Column():
                i2i_inputs.append(gr.Image())
                i2i_inputs.append(gr.Textbox(placeholder="Give me a prompt!", label="Prompt"))
                i2i_inputs.append(gr.Textbox(placeholder="Give me a negative prompt!",label="Negative Prompt"))
                i2i_inputs.append(gr.Slider(0, 1, 0.3, step=0.1, label="Strength"))
                i2i_inputs.append(gr.Slider(0, 10, 2, step=0.1, label="Guidance Scale"))
                i2i_inputs.append(gr.Slider(0, 50, 28, step=1, label="Step"))
                i2i_inputs.append(gr.Slider(0, 0, 10, step=1, label="CLIP"))
                i2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Width"))
                i2i_inputs.append(gr.Slider(512, 2048, 1024, step=8, label="Height"))
            with gr.Column():
                i2i_outputs.append(gr.Textbox())
                i2i_btn = gr.Button("Run")
                i2i_btn.click(fn=i2i_fn, inputs=i2i_inputs, outputs=i2i_outputs)

    return server
