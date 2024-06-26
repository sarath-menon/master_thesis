import numpy as np
import gradio as gr
from PIL import Image
import requests
from io import BytesIO
from game_control import GameController
from models import GPT4OModel, Florence2Model
import base64
import asyncio
import datetime
import os
import json
import time

URL = "http://localhost:8086/screenshot"

gc = GameController()
model = GPT4OModel('prompts/mario-odessey.md')
bbox_model = Florence2Model()

from dataclasses import dataclass

@dataclass
class GameData:
    selected_model: str = "gpt-4o"
    is_auto_execute: bool = False

config = GameData()


def post_screenshot_callback(image, content):
    print("Post screenshot callback executed")
    # Add any additional logic you want to execute after the screenshot is taken
    return image, content

async def get_bboxes(image_array, text_input):

    print("text_input:", text_input)

    text_input = "locate the ladder"
    image = Image.fromarray(image_array)

    # phrase grounded detection
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    results = bbox_model.run_example(image, task_prompt, text_input=text_input)
    bbox_image = bbox_model.get_bbox_image(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
    return bbox_image

async def single_game_screenshot(dummy_call=True):
    game_screenshot = gc.get_screenshot()
    print("Got game screenshot")
    img = Image.open(BytesIO(game_screenshot))

    if dummy_call:
        content = {"action": "move_player", "direction": "forward", "reason": "testing"}

    else:
        # call model
        base64_image = base64.b64encode(game_screenshot).decode('utf-8')
        response = await model.generate_response(base64_image)
        content = response.choices[0].message.content
        content = json.loads(content)

    await do_action(content["action"], content["direction"])
    content = f"Action: {content['action']}, Direction: {content['direction']}, Reason: {content['reason']}"

    return np.array(img), content

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

async def stream_game_screenshot():
    while True:
        game_screenshot = gc.get_screenshot()
        img = Image.open(BytesIO(game_screenshot))

        # call model
        base64_image = base64.b64encode(game_screenshot).decode('utf-8')
        response = await model.generate_response(base64_image)
        content = response.choices[0].message.content

        # content = "Hello"
        await asyncio.sleep(0.01) #ms
        yield np.array(img), content

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


async def call_model(text_input, image=None):
    if config.selected_model == "gpt-4o":
        base64_image = base64.b64encode(image).decode('utf-8')
        stream = await model.generate_response_async(base64_image)
        return stream
    elif config.selected_model == "gpt-3.5":
        return await model.generate_waste_async(text_input)
    else:
        print("Model not supported")
        return None

async def chatbox_callback(message, history, dummy_call=True):

    # pause game
    gc.pause_game()
    game_screenshot = gc.get_screenshot()

    # call model
    stream = await call_model(message, game_screenshot )
    
    # print response
    response = ""
    async for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
        yield response

    response_json = json.loads(response)

    # take action if auto execute is true
    if config.is_auto_execute:
        await do_action(response_json["action"], response_json["direction"])

def execute_btn_callback(chat_input):
    response = chat_input[-1][-1]
    response_json = json.loads(response)
    print(response_json["action"], response_json["direction/target"])

def object_detection_callback(name, selv):
    img = gc.get_screenshot()
    img = Image.open(BytesIO(img))
    
    return img, "Hello " + name + "!"
        
with gr.Blocks() as demo:
    gr.Markdown("# Game Screenshot and Response")
    # with gr.Row():
    #     with gr.Tab("VLM Input"):
    #         vlm_input = gr.Image(show_label=False)
    #     with gr.Tab("Bounding Boxes"):
    #         bbox_output = gr.Image(show_label=False)

    with gr.Column():
        with gr.Tab("Chatbot"):
            chatbot = gr.Chatbot(render=False)            
            chat_input = gr.ChatInterface(
                fn=chatbox_callback,
                examples=["what do yo see", {"text": "hola"}, {"text": "merhaba"}],
                chatbot=chatbot,
                retry_btn=None,
                undo_btn=None,
                # clear_btn=None,
            )


            with gr.Row():
                model_select = gr.Dropdown(value="gpt-4o", choices=["gpt-4o", "gpt4-vision",'llava-1.6',"gpt-3.5"], label="Select model")

                with gr.Column():
                    execute_btn = gr.Button("Execute")
                    auto_execute_checkbox = gr.Checkbox(label="Auto execute")
                    
                
                model_select.change(fn=lambda x: setattr(config, 'selected_model', x), inputs=[model_select], outputs=[])

                auto_execute_checkbox.change(fn=lambda x: setattr(config, 'is_auto_execute', x), inputs=[auto_execute_checkbox], outputs=[])

                execute_btn.click(fn=execute_btn_callback, inputs=[chatbot], outputs=[])

        # object detection output
        with gr.Tab("Object Detection"):
            with gr.Column():
                vlm_input = gr.Image(show_label=False)

            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Text input", placeholder="Enter a text input")
                
                    dropdown = gr.Dropdown(value="florence_2", choices=["grounding_dino", "florence_2"], label="Select model")
                    submit_button = gr.Button("Submit")

                with gr.Column():
                    text_output = gr.Textbox(label="Model output")

            submit_button.click(fn=object_detection_callback, inputs=[text_input, dropdown], outputs=[vlm_input, text_output])

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

                    pause_button.click(fn=gc.pause_game)
                    resume_button.click(fn=gc.resume_game)

    # .then(
    #     fn=save_image_and_response, inputs=[vlm_input, code_output]
    # )

if __name__ == "__main__":
    demo.launch()