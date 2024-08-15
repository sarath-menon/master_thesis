import gradio as gr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
from clicking.visualization.core import show_localization_prediction, show_segmentation_prediction
from clicking.pipeline.core import Clicker
import matplotlib.pyplot as plt

clicker = Clicker()

section_labels = [
    "apple",
    "banana",
    "carrot",
    "donut",
    "eggplant",
    "fish",
    "grapes",
    "hamburger",
    "ice cream",
    "juice",
]

css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""
DESCRIPTION = "# [Florence-2 Demo](https://huggingface.co/microsoft/Florence-2-large)"

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Florence-2 Image Captioning"):
        with gr.Row():
            img_input = gr.Image()
            img_output = gr.AnnotatedImage(
                color_map={"banana": "#a89a00", "carrot": "#ffae00"}
            )

        section_btn = gr.Button("Identify Sections")

        with gr.Column():
            selected_section = gr.Textbox(label="Selected Section")
            prompt_input = gr.Textbox(label="Text prompt")

    def section(image, prompt_input):
        sections = []
        print(prompt_input)
        
        # convert numpy array to PIL image
        image = Image.fromarray(image)

        response = clicker.get_localization_prediction(image, prompt_input)

        sections = []
        for i, bbox in enumerate(response['bboxes']):
            x1,y1,w,h = map(int, bbox)
            sections.append(((x1,y1,w,h), section_labels[i]))


        #     mask = np.zeros(img.shape[:2])
        #     for i in range(img.shape[0]):
        #         for j in range(img.shape[1]):
        #             dist_square = (i - y) ** 2 + (j - x) ** 2
        #             if dist_square < r**2:
        #                 mask[i, j] = round((r**2 - dist_square) / r**2 * 4) / 4
        #     sections.append((mask, section_labels[b + num_boxes]))
    

        return (image, sections)

    section_btn.click(section, [img_input, prompt_input], img_output)

    def select_section(evt: gr.SelectData):
        return section_labels[evt.index]

    img_output.select(select_section, None, selected_section)

if __name__ == "__main__":
    demo.launch()