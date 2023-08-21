import gradio as gr



def Trufor_predict(image_path):
    image_path
    pass
    return 


def attribute_make_result(image_path):
    return 


def component_make_result(image_path):
    return 

with gr.Blocks() as demo:
    gr.Markdown("DeepFake Detection Demo - OC AI Membership")

    with gr.Tab("SeqDeepfake"):
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

    with gr.Tab("Tru For"):

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type='filepath')
                btn = gr.Button(value='Predict')
            with gr.Column():
                result1_output = gr.Textbox(label='is DeepFake?')
                result2_output = gr.Textbox(label='result')

        btn.click(Trufor_predict, inputs=[image_input], outputs=[result1_output, result2_output])

if __name__ == '__main__':
    demo.launch(share=True)