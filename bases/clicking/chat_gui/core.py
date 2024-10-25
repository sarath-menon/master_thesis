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
from development.pipelines.loop_executor import LoopExecutor

RYUJINX_URL = "http://localhost:8086/screenshot"
gc = RyujinxInterface()
# gc = IphoneMirrorInterface()

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


def draw_clickpoint(img, clickpoint):
    draw = ImageDraw.Draw(img)
    x = int(clickpoint.x / 100 * img.width)
    y = int(clickpoint.y / 100 * img.height)
    circle_radius = 10
    draw.ellipse(
        [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
        fill="yellow",
        outline="black"
    )
    return img

async def chatbox_callback(message, history):

    if message.get('files') and message['files'][0].endswith('.yaml'):
        yaml_file = next(f for f in message['files'] if f.endswith('.yaml'))
        
        if not os.path.exists(yaml_file):
            yield "Error: YAML file not found"
            return
            
        # Execute the sequence using the loop executor
        async for img, clickpoint, text_input in LoopExecutor(CONFIG_PATH).execute_sequence_async(
            sequence_file=yaml_file,
            get_image_func=gc.get_screenshot,
            delay=1.0
        ):
            gc.click(x=clickpoint.x, y=clickpoint.y)
            draw_clickpoint(img, clickpoint)

            img_base64 = pil_to_base64(img)
            img_msg = f"{text_input}\n<img src='data:image/webp;base64,{img_base64}' style='width: 500px; max-width:none; max-height:none'></img>"

            yield img_msg


    # Process the single screenshot using the pipeline wrapper
    else:
        img = gc.get_screenshot()
        clickpoint = await pipeline_wrapper.process_image(img, message['text'])

        if clickpoint.validity.status == 'invalid':
            yield f"Invalid clickpoint: {clickpoint.validity.reason}"
            return
            
        # click on the screen
        gc.click(x=clickpoint.x, y=clickpoint.y)

        # Draw the clickpoint on the image
        img = draw_clickpoint(img, clickpoint)
    
        # Create response message
        text_msg = f"Clicked on ({clickpoint.x}, {clickpoint.y})" 
        img_base64 = pil_to_base64(img)
        img_msg = f"{text_msg}\n<img src='data:image/webp;base64,{img_base64}' style='width: 500px; max-width:none; max-height:none'></img>"

        print(message)
        yield img_msg


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

# Initialize the loop executor
loop_executor = LoopExecutor(CONFIG_PATH)

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

        with gr.Row():
            pause_button = gr.Button("Pause game")
            resume_button = gr.Button("Resume game")
            
            pause_button.click(fn=gc.pause_emulator)
            resume_button.click(fn=gc.resume_emulator)

        with gr.Row():
            emulator_dropdown = gr.Dropdown(
                [ "Ryujinx", "Iphone Mirror"], label="Emulator selector"
            )
            connect_emulator_btn = gr.Button("Connect emulator")
            disconnect_emulator_btn = gr.Button("Disconnect emulator")

            emulator_dropdown.change(fn=set_emulator, inputs=[emulator_dropdown])
            connect_emulator_btn.click(fn=gc.connect_emulator)
            disconnect_emulator_btn.click(fn=gc.disconnect_emulator)

if __name__ == "__main__":
    demo.launch()
