import numpy as np
import gradio as gr
from PIL import Image
import requests
from io import BytesIO

URL = "http://localhost:8086/screenshot"

def sepia():
    response = requests.get(URL)
    img = Image.open(BytesIO(response.content))
    return np.array(img), "Hello"

demo = gr.Interface(fn=sepia, inputs=[], outputs=[
    gr.Image(),
    gr.Code()]
    )

if __name__ == "__main__":
    demo.launch()