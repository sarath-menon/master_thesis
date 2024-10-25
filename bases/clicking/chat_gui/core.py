import numpy as np
import gradio as gr
from PIL import Image
from io import BytesIO
from clicking.emulator_interface import RyujinxInterface, IphoneMirrorInterface
import base64
import datetime
import os
import json
from clicking.pipelines.molmo_direct import MolmoDirectPipelineWrapper
import yaml
import math
from PIL import ImageDraw
import io
from clicking.vision_model.utils import pil_to_base64

RYUJINX_URL = "http://localhost:8086/screenshot"
# gc = RyujinxInterface()
gc = IphoneMirrorInterface()

async def save_image_and_response(image_array, response):
    # Create a directory with the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_path = os.path.join("logs", current_time)
    os.makedirs(directory_path, exist_ok=True)
    
    # Save the image
    img = Image.fromarray(image_array)
    img = img.convert('RGB')
    img.save(os.path.join(directory_path, "screenshot.jpeg"), "JPEG")
    
    # Save the response
    response_path = os.path.join(directory_path, "response.txt")
    with open(response_path, "w") as file:
        file.write(response)
    
    print(f"Saved image and response in {directory_path}")



async def chatbox_callback(message, history):
    img = gc.get_screenshot()
    
    # Process the image using the pipeline wrapper
    clickpoint = await pipeline_wrapper.process_image(img, message['text'])

    if clickpoint.validity.status == 'invalid':
        return f"Invalid clickpoint: {clickpoint.validity.reason}"
        
    # click on the screen
    gc.click(x=clickpoint.x, y=clickpoint.y)
    
    # save the image
    img.save(os.path.join('./development/iphone_click', "screenshot.png"), "PNG")


    # Overlay a circle on the image at the clickpoint coordinates
    draw = ImageDraw.Draw(img)
    x = int(clickpoint.x / 100 * img.width)
    y = int(clickpoint.y / 100 * img.height)
    circle_radius = 10
    draw.ellipse(
        [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
        fill="yellow",
        outline="black"
    )    
    
    # Convert PIL image to bytes
    img_base64 = pil_to_base64(img)

    # Create and return MultimodalMessage
    # text_msg = f"Clickpoint is x: {x}, y: {y}"
    img_msg = f"<img src='data:image/webp;base64,{img_base64}' style='width: 500px; max-width:none; max-height:none'></img>"

    return img_msg

def execute_btn_callback(chat_input):
    response = chat_input[-1][-1]
    response_json = json.loads(response)
    print(response_json["action"], response_json["direction/target"])



def set_emulator(emulator):
    global gc
    if emulator == "Ryujinx":
        gc = RyujinxInterface()
    elif emulator == "Iphone Mirror":
        gc = IphoneMirrorInterface()

# Load the configuration file
CONFIG_PATH = "./development/pipelines/game_object_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Initialize the pipeline wrapper
pipeline_wrapper = MolmoDirectPipelineWrapper(config)

CSS ="""
#chatbot { flex-grow: 1; overflow: auto; height: 60vh !important;}
"""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# Game Screenshot and Response")

    with gr.Column():
        with gr.Tab("Chatbot"):
            chatbot = gr.Chatbot(
                [], 
                elem_id="chatbot",
                type='messages',
                bubble_full_width=False,
            )

            chat_input = gr.ChatInterface(
                fn=chatbox_callback,
                examples=[{"text": "start button"}, {"text": "back button"}, {"text": "button named"}, {"text": "game object that the instructions in the textbox are asking you to click on"}],
                example_labels=["Start button", "Back button", "Button named ...", "Follow instructions"],
                type='messages',
                chatbot=chatbot,
                multimodal=True,
                autofocus=True,
            )

        with gr.Tab("Manual Action"):
            with gr.Column():
                gr.Markdown("## Select action manually")
                action_select = gr.Radio(["move_player", "orbit_camera", "throw_hat", "jump"], label="Select action")

                direction_select = gr.Radio(["forward", "backward", "left", "right"], label="Select direction")

                action_button = gr.Button("Do action")
                action_select.change(fn=update_direction_options, inputs=[action_select], outputs=[direction_select])

                action_button.click(fn=do_action, inputs=[action_select, direction_select])    
        
        with gr.Row():
            pause_button = gr.Button("Pause game")
            resume_button = gr.Button("Resume game")
            
            pause_button.click(fn=gc.pause_emulator)
            resume_button.click(fn=gc.resume_emulator)

        with gr.Row():
            emulator_dropdown = gr.Dropdown(
                ["Iphone Mirror", "Ryujinx"], label="Emulator selector"
            )
            connect_emulator_btn = gr.Button("Connect emulator")
            disconnect_emulator_btn = gr.Button("Disconnect emulator")

            emulator_dropdown.change(fn=set_emulator, inputs=[emulator_dropdown])
            connect_emulator_btn.click(fn=gc.connect_emulator)
            disconnect_emulator_btn.click(fn=gc.disconnect_emulator)

if __name__ == "__main__":
    demo.launch()
