import numpy as np
import gradio as gr
from PIL import Image
import requests
from io import BytesIO
from game_control import GameController
from models import GPT4OModel
import base64
import asyncio

URL = "http://localhost:8086/screenshot"

gc = GameController()
model = GPT4OModel()

async def single_game_screenshot(text):
    game_screenshot = gc.get_screenshot()
    print("Got game screenshot")
    img = Image.open(BytesIO(game_screenshot))

    # base64_image = base64.b64encode(game_screenshot).decode('utf-8')
    # response = await model.generate_response(base64_image)
    # content = response.choices[0].message.content

    content = "Hello"

    return np.array(img), content

async def stream_game_screenshot(text):
    while True:
        game_screenshot = gc.get_screenshot()
        img = Image.open(BytesIO(game_screenshot))

        base64_image = base64.b64encode(game_screenshot).decode('utf-8')

        response = await model.generate_response(base64_image)
        content = response.choices[0].message.content
        print("Got response")

        # content = "Hello"
        await asyncio.sleep(0.01)  # Sleep for 0.1 seconds (10 times a second)
        yield np.array(img), content

with gr.Blocks() as demo:
    gr.Markdown("# Game Screenshot and Response")
    with gr.Row():
        image_output = gr.Image(label="Game Screenshot", scale=2)
        code_output = gr.Code(label="Response")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Prompt", lines=6)
            submit_button = gr.Button("Submit")

    submit_button.click(fn=single_game_screenshot, inputs=text_input, outputs=[image_output, code_output])

if __name__ == "__main__":
    demo.launch()