from seqdeepfake_ import attribute_predict, component_predict
from groundingdino_ import dino_predict
import gradio as gr


def attribute_make_result(image_path):
    # SeqDeepFake - attribute model
    results = attribute_predict(image_path)
    # GroundingDINO
    image_output = dino_predict(image_path, results[1])

    return results[0], results[2], image_output


def component_make_result(image_path):
    # SeqDeepFake - component model
    results = component_predict(image_path)
    # GroundingDINO
    image_output = dino_predict(image_path, results[1])

    return results[0], results[2], image_output


with gr.Blocks() as demo:
    gr.Markdown("DeepFake Detection Demo - OC AI Membership")

    with gr.Tab("Facial Attributes"):
        with gr.Row():
            with gr.Column(scale=1, min_width=700):
                image_path = gr.Image(type='filepath')
                btn = gr.Button(value='Predict')
                result1_output = gr.Textbox(label='DeepFake')
                result2_output = gr.Textbox(label='Attribute')

            with gr.Column(scale=2, min_width=500):
                result3_output = gr.Image(type='filepath', width=500)
                btn.click(attribute_make_result, inputs=[image_path], outputs=[result1_output, result2_output, result3_output])

    with gr.Tab("Facial Components"):
        with gr.Row():
            with gr.Column(scale=1, min_width=700):
                image_path = gr.Image(type='filepath')
                btn = gr.Button(value='Predict')
                result1_output = gr.Textbox(label='DeepFake')
                result2_output = gr.Textbox(label='Component')

            with gr.Column(scale=2, min_width=500):
                result3_output = gr.Image(type='filepath', width=500)
                btn.click(component_make_result, inputs=[image_path], outputs=[result1_output, result2_output, result3_output])


if __name__ == '__main__':
    demo.launch(share=True)
