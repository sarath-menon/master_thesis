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

async def do_action(action, direction):
    if action == "move_player":
        gc.move_player(direction)
    elif action == "orbit_camera":
        gc.orbit_camera(direction)
    elif action == "collect_treasure":
        gc.collect_treasure(direction)
    print(f"Doing action: {action}, direction: {direction}")

def update_direction_options(action):
    if action == "move_player":
        return gr.update(choices=["forward", "backward", "left", "right"])
    elif action == "orbit_camera":
        return gr.update(choices=["up", "down", "left", "right"])
    elif action == "collect_treasure":
        return gr.update(choices=["forward", "backward", "left", "right"])
    return gr.update(choices=[])

# async def slow_echo(message, history):
#     # pause game
#     gc.pause_game()

#     # # get model output and game screenshot
#     # (img, content) = await single_game_screenshot()

#     chunk = await model.generate_waste_async(message)
#     print(chunk)

#     # async for chunk in stream:
#     #     yield chunk.choices[0].delta.content or ""

async def slow_echo(message, history, dummy_call=True):

    # pause game
    gc.pause_game()

    game_screenshot = gc.get_screenshot()
    print("Got game screenshot")
    img = Image.open(BytesIO(game_screenshot))

    # call model
    base64_image = base64.b64encode(game_screenshot).decode('utf-8')
    stream = await model.generate_response_async(base64_image)
    # content = response.choices[0].message.content
    # content = json.loads(content)

    # stream = await model.generate_waste_async(message)

    # print response
    response = ""
    async for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
        yield response

    content = json.loads(response)
    print(content)
    
    # do action
    await do_action(content["action"], content["direction"])
        
with gr.Blocks() as demo:
    gr.Markdown("# Game Screenshot and Response")
    # with gr.Row():
    #     with gr.Tab("VLM Input"):
    #         vlm_input = gr.Image(show_label=False)
    #     with gr.Tab("Bounding Boxes"):
    #         bbox_output = gr.Image(show_label=False)

    with gr.Column(scale=2):
        chatbot = gr.Chatbot(render=False)

        gr.ChatInterface(
        fn=slow_echo,
        examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}],
            chatbot=chatbot
        )

    with gr.Row():
    #     with gr.Column():
    #         gr.Markdown("## Model output")
    #         # text_input = gr.Textbox(container=False, lines=6)
    #         # code_output = gr.Textbox(container=False, lines=6)
    #         submit_button = gr.Button("Submit")
    #         # debug_button = gr.Button("Print Model Prompts")
    #         # debug_button.click(fn=lambda: print(model.prompts_dict))

        with gr.Column():
            gr.Markdown("## Select action manually")
            action_select = gr.Radio(["move_player", "orbit_camera", "collect_treasure"], label="Select action")

            direction_select = gr.Radio(["forward", "backward", "left", "right"], label="Select direction")

            action_button = gr.Button("Do action")
            action_button.click(fn=do_action, inputs=[action_select, direction_select])
            action_select.change(fn=update_direction_options, inputs=[action_select], outputs=[direction_select])

            with gr.Row():
                pause_button = gr.Button("Pause game")
                resume_button = gr.Button("Resume game")

                pause_button.click(fn=gc.pause_game)
                resume_button.click(fn=gc.resume_game)


            # submit_button.click(fn=single_game_screenshot, inputs=[], outputs=[vlm_input, code_output]).then(
            #     fn=get_bboxes, inputs=[vlm_input, code_output], outputs=[bbox_output]
            # )

    # .then(
    #     fn=save_image_and_response, inputs=[vlm_input, code_output]
    # )

if __name__ == "__main__":
    demo.launch()