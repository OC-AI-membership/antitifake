import gradio as gr
from PIL import Image, ImageDraw

title = "OC AI membership"

def detect_fake(input_):
    print(input_)
    img = Image.open(input_).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle((100,100,300,300), outline=(0,255,0), width = 3)

    return img

with gr.Blocks() as block:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        with gr.Row():
            image_init = gr.Image(source="upload", type="filepath",label="Input Image")
            detect_bttn = gr.Button("Detect")

        with gr.Column(elem_id="col-container"):

            image_output = gr.Image(label="Detected image")            

    detect_bttn.click(fn=detect_fake, inputs=[image_init], outputs=[image_output])
    
block.queue(max_size=12).launch(show_api=False, share=True)
