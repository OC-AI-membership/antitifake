from SeqDeepFake.src.seqdeepfake_ import model_predict
from GroundingDINO.src.groundingdino_ import dino_predict
from TruFor.src.trufor_ import trufor
import gradio as gr


def make_result(image_path):
    results = model_predict(image_path)
    image_output = dino_predict(image_path, results[1])

    return results[0], results[2], image_output


with gr.Blocks() as demo:
    gr.Markdown("DeepFake Detection Demo - OC AI Membership")

    with gr.Tab("SeqDeepfake"):
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                image_path = gr.Image(type='filepath', value='./.test/seqdeepfake.png')
                btn = gr.Button(value='predict')
                result1_output = gr.Textbox(label='is it deepfake?')
                result2_output = gr.Textbox(label='manipulated')
            with gr.Column(scale=2, min_width=400):
                result3_output = gr.Image(type='filepath', width=400)
        btn.click(make_result, inputs=[image_path], outputs=[result1_output, result2_output, result3_output])

    with gr.Tab("Tru For"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type='filepath', value='./.test/trufor.png')
                btn = gr.Button(value='Predict')
                result1_output = gr.Textbox(label='is DeepFake?')
            with gr.Column():
                result2_output = gr.Image(type='pil', width=500)

        btn.click(trufor, inputs=[image_input], outputs=[result1_output, result2_output])


if __name__ == '__main__':
    demo.launch(share=True)
