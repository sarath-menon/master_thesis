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

async def sepia():
    while True:
        game_screenshot = gc.get_screenshot()
        img = Image.open(BytesIO(game_screenshot))

        base64_image = base64.b64encode(game_screenshot).decode('utf-8')

        # response = await model.generate_response(base64_image)
        # content = response.choices[0].message.content
        # print("Got response")

        content = "Hello"


        await asyncio.sleep(0.1)  # Sleep for 0.1 seconds (10 times a second)
        yield np.array(img), content

demo = gr.Interface(fn=sepia, inputs=[], outputs=[
    gr.Image(),
    gr.Code()]
    )

if __name__ == "__main__":
    demo.launch()