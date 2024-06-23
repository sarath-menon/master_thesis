import numpy as np
import gradio as gr
from PIL import Image
import requests
from io import BytesIO
from game_control import GameController

URL = "http://localhost:8086/screenshot"

gc = GameController()

def sepia():
    game_screenshot = gc.get_screenshot()
    img = Image.open(BytesIO(game_screenshot))
    return np.array(img), "Hello"

demo = gr.Interface(fn=sepia, inputs=[], outputs=[
    gr.Image(),
    gr.Code()]
    )

if __name__ == "__main__":
    demo.launch()