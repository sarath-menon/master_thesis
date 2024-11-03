import gradio as gr
from clicking.emulator_interface import *
import numpy as np
import json
from clicking.vision_model.utils import pil_to_base64
from PIL import ImageDraw, Image
from clicking.common.data_structures import ClickPoint
from pydantic import BaseModel
import yaml
import os
from clicking.common.image_utils import HostedModelClientBase


CONFIG_PATH = "projects/chat_gui/hosted_models.yaml"    

class Endpoint(BaseModel):
    model_name: str
    url: str

def load_config():
    CONFIG = yaml.safe_load(open(CONFIG_PATH))
    return CONFIG

config = load_config()
endpoints = []
for model in config["runpod"]:
    endpoint = Endpoint(**model)
    endpoints.append(endpoint)

endpoint_names = [endpoint.model_name for endpoint in endpoints]

client = None


def execute_btn_callback(chat_input):
    response = chat_input[-1][-1]
    response_json = json.loads(response)
    print(response_json["action"], response_json["direction/target"])

def add_message(history, message):
    if message.get('files'):
        for file in message['files']:
            history.append({"role": "user", "content": {"path": file}})
    if message.get('text'):
        history.append({"role": "user", "content": message['text']})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

async def bot(history: list):
    last_message = history[-1]["content"]

    response = "This is a test response"
    history.append({"role": "assistant", "content": response})
    yield history

CSS ="""
#chatbot { flex-grow: 1; overflow: auto; height: 50vh !important;}
"""

def draw_clickpoint(img, clickpoint, radius=10, color="yellow", outline="black", line_width=5, line_spacing=20):
    draw = ImageDraw.Draw(img)
    x = int(clickpoint.x / 100 * img.width)
    y = int(clickpoint.y / 100 * img.height)

    # Draw horizontal dotted line
    for i in range(0, img.width, line_spacing):
        draw.line([(i, y), (i+line_width, y)], fill=outline, width=line_width)
    
    # Draw vertical dotted line  
    for i in range(0, img.height, line_spacing):
        draw.line([(x, i), (x, i+line_width)], fill=outline, width=line_width)
        
    # # Draw circle at intersection
    # draw.ellipse(
    #     [(x - radius, y - radius), (x + radius, y + radius)],
    #     fill=color,
    #     outline=outline
    # )
    return img

def img_click_callback(img, evt: gr.SelectData):
    x, y = evt.index

    img_pil = Image.fromarray(img)

    # convert to percentage
    x_percent = (x / img_pil.width) * 100
    y_percent = (y / img_pil.height) * 100

    print(f"Clickpoint: {x_percent:.1f}, {y_percent:.1f}")

    if client is None:
        text_output = "No model selected"
    else:
        messages = []
        text_input = f"""
        The image is a game screenshot. Describe the object at the coordinates:
        {{       
        "x": {x_percent:.1f},
        "y": {y_percent:.1f}
        }}. Make sure that you describe the exact object at this point.
        """

        # text_input = f"""
        # The image is a game screenshot. Describe the object at the coordinates: <point x="{x_percent:.1f}" y="{y_percent:.1f}"> </point>. Make sure that you describe the exact object at this point.
        # """
        text_output = client.get_image_response(img_pil, text_input, messages)
    

    img_ann = draw_clickpoint(img_pil.copy(), ClickPoint(x=x_percent, y=y_percent))

    return img_ann, text_output

def set_model(model):
    global client

    for endpoint in endpoints:
        print(endpoint.model_name, model)
        if endpoint.model_name != model:
            continue

        client = HostedModelClientBase(model=endpoint.model_name, api_base=endpoint.url, api_key=os.environ["RUNPOD_API_KEY"])
        print("Model set to:", model)
        return

    print(f"Could not find model: {model} in config file")


with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# Hosted Model Chat")

    with gr.Tab("Clickpoint input"):
        with gr.Column():
            with gr.Row():
                input_img = gr.Image(label="Input")
                output_img = gr.Image(label="Selected Segment")

            text_output = gr.Textbox(
                value="",
                interactive=False,
                label="Output",
            )

            input_img.select(img_click_callback, [input_img], [output_img, text_output])

    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot(
            [], 
            elem_id="chatbot",
            type='messages',
            bubble_full_width=False,
            autoscroll=True,
        )

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Enter message or upload file...",
            show_label=False,
            autoscroll=True,
            autofocus=True,
        )

        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(bot, chatbot, [chatbot], api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        def print_like_dislike(x: gr.LikeData):
            print(x.index, x.value, x.liked)

        chatbot.like(print_like_dislike, None, None, like_user_message=True)

    with gr.Row():
        with gr.Column():

            model_dropdown = gr.Dropdown(
                [endpoint.model_name for endpoint in endpoints], label="Model selector"
            )
            # submit_btn = gr.Button("Set model")
            # connection_status = gr.Textbox(
            #     value="Disconnected",
            #     label="Connection Status",
            #     interactive=False
            # )

            set_model(model_dropdown.value)
            model_dropdown.change(fn=set_model, inputs=[model_dropdown])
            # submit_btn.click(fn=set_model, inputs=[model_dropdown])
        
        # with gr.Row():
        #     with gr.Column():
        #     # connect_emulator_btn.click(fn=connect_wrapper, outputs=[connection_status])
        #     # disconnect_emulator_btn.click(fn=disconnect_wrapper, outputs=[connection_status])
        # connect_emulator_btn = gr.Button("Connect emulator")
        # disconnect_emulator_btn = gr.Button("Disconnect emulator")

if __name__ == "__main__":
    demo.launch()
