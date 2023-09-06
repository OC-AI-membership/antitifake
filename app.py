from SeqDeepFake.src.seqdeepfake_ import attribute_predict, component_predict
from GroundingDINO.src.groundingdino_ import dino_predict
from TruFor.src.trufor_ import trufor
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
    gr.Markdown('DeepFake Detection Demo - OC AI Membership')

    with gr.Tab('Facial Attributes'):
        gr.Markdown('detect - bangs, eyeglasses, beard, smiling, young')
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                image_path = gr.Image(type='filepath', value='./.asset/attribute.png')
                btn = gr.Button(value='Predict')
                result1_output = gr.Textbox(label='DeepFake')
                result2_output = gr.Textbox(label='Attributes')
            with gr.Column(scale=2, min_width=400):
                result3_output = gr.Image(type='filepath', width=400)


            def predicts(image_path):
                # Trufor
                trufor_results = trufor(image_path)

                if trufor_results[0] > threshold:
                    # SeqDeepFake
                    seq_results = attribute_make_result(image_path)
                    result1_output = True
                    result2_output = seq_results[1]
                    result3_output = seq_results[2]

                else:
                    result1_output = False
                    result2_output = None
                    result3_output = None

                return result1_output, result2_output, result3_output

            btn.click(predicts, inputs=[image_path], outputs=[result1_output, result2_output, result3_output])

    with gr.Tab('Facial Components'):
        gr.Markdown('detect - nose, eye, eyebrow, lip, hair')

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                image_path = gr.Image(type='filepath', value='./.asset/component.png')
                btn = gr.Button(value='Predict')
                result1_output = gr.Textbox(label='DeepFake')
                result2_output = gr.Textbox(label='Components')
            with gr.Column(scale=2, min_width=400):
                result3_output = gr.Image(type='filepath', width=400)


            def predicts(image_path):
                # Trufor
                trufor_results = trufor(image_path)

                if trufor_results[0] > threshold:
                    # SeqDeepFake
                    seq_results = component_make_result(image_path)
                    result1_output = True
                    result2_output = seq_results[1]
                    result3_output = seq_results[2]

                else:
                    result1_output = False
                    result2_output = None
                    result3_output = None

                return result1_output, result2_output, result3_output

            btn.click(predicts, inputs=[image_path], outputs=[result1_output, result2_output, result3_output])


if __name__ == '__main__':
    threshold = 0.5
    demo.launch(share=True)
