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

def select_section(evt: gr.SelectData):
        return section_labels[evt.index]

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

label_creation_models = {
    "GPT-4": "",
    "GPT-4o": "",
}

localization_models = {
    "Florence-2-large": "",
    "Florence-2-base": ""
}

segmentation_models = {
    "Sam-2-tiny": "",
    "Sam-2-large": "",
}

pipelines = {
    "localization + segmentation":"",
    "localization + geometric center":"",
}

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Florence-2 Image Captioning"):
        with gr.Row():
            img_input = gr.Image()
            img_output = gr.AnnotatedImage(
                color_map={"banana": "#a89a00", "carrot": "#ffae00"}
            )

        with gr.Row():
            with gr.Column():
                section_btn = gr.Button("Identify Sections")

                prompt_input = gr.Textbox(label="Text prompt")

                pipeline_selector = gr.Dropdown(choices=list(pipelines.keys()), label="Select pipeline")

                label_creation_mode = gr.Dropdown(choices=list(label_creation_models .keys()), label="Text prompt to class label")

                localization_mode = gr.Dropdown(choices=list(localization_models.keys()), label="Localization model")
                        
                segmentation_mode = gr.Dropdown(choices=list(segmentation_models.keys()), label="Segmentation model")

            with gr.Column():
                selected_section = gr.Textbox(label="Selected Section")
    
    section_btn.click(section, [img_input, prompt_input], img_output)
    img_output.select(select_section, None, selected_section)

if __name__ == "__main__":
    demo.launch()