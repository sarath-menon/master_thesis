import numpy as np
import gradio as gr
from PIL import Image
from io import BytesIO
from clicking.emulator_interface import *
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
import time

RYUJINX_URL = "http://localhost:8086/screenshot"

gc = CustomEmulator()

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
    elif emulator == "Appium":
        gc = AppiumInterface()  
        time.sleep(2)
    else:
        print("No emulator selected")

    print("Emulator set to:", gc.__class__.__name__)

def connect_wrapper():
    success = gc.connect_emulator()
    return "Connected" if success else "Connection failed"

def disconnect_wrapper():
    success = gc.disconnect_emulator()
    return "Disconnected" if success else "Disconnection failed"

# Load the configuration file
CONFIG_PATH = "./development/pipelines/game_object_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Initialize the pipeline wrapper
pipeline_wrapper = MolmoDirectPipelineWrapper(config)

# Initialize the loop executor
loop_executor = LoopExecutor(CONFIG_PATH)


def add_message(history, message):
    if message.get('files'):
        for file in message['files']:
            history.append({"role": "user", "content": {"path": file}})
    if message.get('text'):
        history.append({"role": "user", "content": message['text']})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

async def bot(history: list):
    last_message = history[-1]["content"]

    # Handle file uploads (YAML files)
    if last_message[0].endswith('.yaml'):
        yaml_file = last_message[0]
        if not os.path.exists(yaml_file):
            response = "Error: YAML file not found"
            history.append({"role": "assistant", "content": response})
            yield history, ""  # Add empty string for text_output
            return


        async for img, clickpoint, text_input in LoopExecutor(CONFIG_PATH).execute_sequence_async(
            sequence_file=yaml_file,
            get_image_func=gc.get_screenshot,
            delay=1.0
        ):
            if clickpoint.validity.status == 'invalid':
                response = f"No valid clickpoint for object: {text_input}"
            else:
                gc.click(x=clickpoint.x, y=clickpoint.y)
                draw_clickpoint(img, clickpoint)
                img_base64 = pil_to_base64(img)
                text_msg = f"{text_input}: ({clickpoint.x}, {clickpoint.y})"
                response = f"{text_msg}\n<img src='data:image/webp;base64,{img_base64}' style='width: 500px; max-width:none; max-height:none'></img>"

            history.append({"role": "assistant", "content": response})
            yield history, text_input  # Yield both history and text_input

    # Handle text messages
    else:
        img = gc.get_screenshot()
        clickpoint = await pipeline_wrapper.process_image(img, last_message)

        if clickpoint.validity.status == 'invalid':
            response = f"Invalid clickpoint: {clickpoint.validity.reason}"
        else:
            gc.click(x=clickpoint.x, y=clickpoint.y)
            img = draw_clickpoint(img, clickpoint)
            text_msg = f"Clicked on ({clickpoint.x}, {clickpoint.y})"
            img_base64 = pil_to_base64(img)
            response = f"{text_msg}\n<img src='data:image/webp;base64,{img_base64}' style='width: 500px; max-width:none; max-height:none'></img>"

        history.append({"role": "assistant", "content": response})
        yield history, last_message  # Yield both history and last_message
CSS ="""
#chatbot { flex-grow: 1; overflow: auto; height: 60vh !important;}
"""

# js = """
# (function() {
#     function scrollChatToBottom() {
#         const chatbot = document.querySelector('#chatbot');
#         if (chatbot) {
#             chatbot.scrollTop = chatbot.scrollHeight;
#         }
#     }

#     if (document.readyState === 'loading') {
#         document.addEventListener('DOMContentLoaded', initObserver);
#     } else {
#         initObserver();
#     }

#     function initObserver() {
#         const chatbot = document.querySelector('#chatbot');
#         if (chatbot) {
#             const observer = new MutationObserver(scrollChatToBottom);
#             observer.observe(chatbot, {
#                 childList: true,
#                 subtree: true
#             });
#             scrollChatToBottom();
#         }
#     }
# })();
# """
with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# Game Screenshot and Response")

    with gr.Column():
        with gr.Tab("Chatbot"):

            # output text
            text_output = gr.Textbox(
                value="",
                interactive=False,
                label="Current instruction",
            )

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
            bot_msg = chat_msg.then(bot, chatbot, [chatbot, text_output], api_name="bot_response")
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            def print_like_dislike(x: gr.LikeData):
                print(x.index, x.value, x.liked)

            chatbot.like(print_like_dislike, None, None, like_user_message=True)

        # with gr.Row():
        #     pause_button = gr.Button("Pause game")
        #     resume_button = gr.Button("Resume game")
            
        #     pause_button.click(fn=gc.pause_emulator)
        #     resume_button.click(fn=gc.resume_emulator)

        with gr.Row():
            with gr.Column():
                emulator_dropdown = gr.Dropdown(
                    [ "None", "Ryujinx", "Appium", "Iphone Mirror"], label="Emulator selector"
                )
                connection_status = gr.Textbox(
                    value="Disconnected",
                    label="Connection Status",
                    interactive=False
                )

            connect_emulator_btn = gr.Button("Connect emulator")
            disconnect_emulator_btn = gr.Button("Disconnect emulator")
            
            emulator_dropdown.change(fn=set_emulator, inputs=[emulator_dropdown])
            connect_emulator_btn.click(fn=connect_wrapper, outputs=[connection_status])
            disconnect_emulator_btn.click(fn=disconnect_wrapper, outputs=[connection_status])

if __name__ == "__main__":
    demo.launch()
