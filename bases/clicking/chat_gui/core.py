import numpy as np
import gradio as gr
from PIL import Image
import requests
from io import BytesIO
from clicking.ryujinx_interface.core import RyujinxInterface
import base64
import asyncio
import datetime
import os
import json
import time
import cv2
from clicking.pipelines.molmo_direct import MolmoDirectPipelineWrapper
import yaml
import math
from PIL import ImageDraw
from clicking.chat_gui.macos_clicking import WindowCapture

URL = "http://localhost:8086/screenshot"

gc = RyujinxInterface()

from dataclasses import dataclass

@dataclass
class GameData:
    selected_model: str = "gpt-4o-img"
    is_auto_execute: bool = False
config = GameData()

images_list = [{"files": ["/Users/sarathmenon/Documents/master_thesis/datasets/game_dataset/raw/fortnite/1.jpg"], "text": "Please pay attention to the movement of the object from the first image to the second image, then write a HTML code to show this movement."}]

def image_to_base64(pil_image):
    # Convert PIL Image to bytes directly
    buffered = BytesIO()
    pil_image.save(buffered, format="WEBP")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


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


async def do_action(action, direction=None):
    if action == "move_player":
        gc.move_player(direction)
    elif action == "orbit_camera":
        gc.orbit_camera(direction)
    elif action == "throw_hat":
        gc.special_action(action)
    elif action == "jump":
        gc.special_action(action)
    else:
        print(f"Invalid action: {action}")
        return
    print(f"Doing action: {action}, direction: {direction}")

def update_direction_options(action):
    if action == "move_player":
        return gr.update(choices=["forward", "backward", "left", "right"])
    elif action == "orbit_camera":
        return gr.update(choices=["up", "down", "left", "right"])
    elif action == "collect_treasure":
        return gr.update(choices=["forward", "backward", "left", "right"])
    return gr.update(choices=[])

# Image should be PIL image
async def call_model(text_input, image=None):
    pass

# from gradio.data_classes import FileData, BaseModel
# from typing import Optional, List, Tuple

# class FileMessage(BaseModel):
#     file: FileData
#     alt_text: Optional[str] = None

# class MultimodalMessage(BaseModel):
#     text: Optional[str] = None
#     files: Optional[List[FileMessage]] = None

# class ChatbotData(BaseModel):
#     root: List[Tuple[Optional[MultimodalMessage], Optional[MultimodalMessage]]]

import io

async def chatbox_callback(message, history):
    img = gc.get_screenshot()
    
    # Process the image using the pipeline wrapper
    clickpoint = await pipeline_wrapper.process_image(img, message['text'])

    # click on the screen
    window_capture = WindowCapture()
    window_capture.click(x=clickpoint.x, y=clickpoint.y)
    
    # Overlay a star icon on the image at the clickpoint coordinates
    draw = ImageDraw.Draw(img)
    x = int(clickpoint.x / 100 * img.width)
    y = int(clickpoint.y / 100 * img.height)
    star_size = 10
    star_points = generate_star_points((x, y), size=star_size)
    draw.polygon(star_points, fill="yellow", outline="black")
    
    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Create and return MultimodalMessage
    # text_msg = f"Clickpoint is x: {x}, y: {y}"
    img_base64 = image_to_base64(img)

    img_msg = f"<img src='data:image/webp;base64,{img_base64}' style='width: 600px; max-width:none; max-height:none'></img>"

    return img_msg

def execute_btn_callback(chat_input):
    response = chat_input[-1][-1]
    response_json = json.loads(response)
    print(response_json["action"], response_json["direction/target"])

def generate_star_points(centroid, size=20):
    x, y = centroid
    points = []
    # There are 10 points in a 5-point star
    for i in range(10):
        angle = math.pi / 2 + (i * 2 * math.pi / 10)
        if i % 2 == 0:
            # Outer point
            x = x + size * math.cos(angle)
            y = y - size * math.sin(angle)
        else:
            # Inner point (halfway toward the center)
            x = x + (size / 2) * math.cos(angle)
            y = y - (size / 2) * math.sin(angle)
        points.append((x, y))
    return points
    
async def clicking_pipeline_callback(model, text_input):
    img = gc.get_screenshot()
    
    # Process the image using the pipeline wrapper
    clickpoint = await pipeline_wrapper.process_image(img, text_input)

    # click on the screen
    window_capture = WindowCapture()
    window_capture.click(x=clickpoint.x, y=clickpoint.y)
    
    # Overlay a star icon on the image at the clickpoint coordinates
    draw = ImageDraw.Draw(img)
    x = int(clickpoint.x / 100 * img.width)
    y = int(clickpoint.y / 100 * img.height)
    star_size = 10
    star_points = generate_star_points((x, y), size=star_size)
    draw.polygon(star_points, fill="yellow", outline="black")
    
    # Return the overlayed image
    return img, f"x: {x}, y: {y}"

# Load the configuration file
CONFIG_PATH = "./development/pipelines/game_object_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Initialize the pipeline wrapper
pipeline_wrapper = MolmoDirectPipelineWrapper(config)


with gr.Blocks(css=".message-wrap.svelte-1lcyrx4>div.svelte-1lcyrx4  img {min-width: 200px}") as demo:
    gr.Markdown("# Game Screenshot and Response")

    with gr.Column():
        with gr.Tab("Chatbot"):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
            )

            chat_input = gr.ChatInterface(
                fn=chatbox_callback,
                examples=[{"text": "start button"}, {"text": "back button in the bottom left"}, {"text": "pink diamond in the top left"}],
                chatbot=chatbot,
                retry_btn=None,
                undo_btn=None,
                multimodal=True,
                clear_btn="Clear",
                autofocus=True,
            )

        # object detection output
        with gr.Tab("Object Detection"):
            with gr.Column():
                vlm_input = gr.Image(show_label=False)

            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Text input", placeholder="Enter a text input")
                
                    obj_detection_dropdown = gr.Dropdown(value="florence_2", choices=["grounding_dino", "florence_2"], label="Select model")
                    submit_button = gr.Button("Submit")

                with gr.Column():
                    obj_detection_output = gr.Textbox(label="Model output")

                submit_button.click(fn=clicking_pipeline_callback, inputs=[obj_detection_dropdown, text_input], outputs=[vlm_input, obj_detection_output])

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
            connect_emulator_btn = gr.Button("Connect emulator")
            disconnect_emulator_btn = gr.Button("Disconnect emulator")

            pause_button.click(fn=gc.pause_game)
            resume_button.click(fn=gc.resume_game)
            connect_emulator_btn.click(fn=gc.connect_websockets)
            disconnect_emulator_btn.click(fn=gc.close_websockets)

if __name__ == "__main__":
    demo.launch()
