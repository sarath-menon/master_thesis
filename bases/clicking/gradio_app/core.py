import gradio as gr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from clicking_client import Client
from clicking_client.models import PredictionReq, PredictionResp
from clicking_client.api.default import get_available_localization_models, get_localization_prediction, get_available_segmentation_models
import io
import base64


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

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def section(image, text_input):
        sections = []

        # convert numpy array to PIL image
        image = Image.fromarray(image)

        request = PredictionReq(image=image_to_base64(image), text_input=text_input, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>')

        response = get_localization_prediction.sync(client=client, body=request)

        sections = []
        for i, bbox in enumerate(response.bboxes):
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

pipelines = {
    "localization + segmentation":"",
    "localization + geometric center":"",
}

client = Client(base_url="http://localhost:8082")
res = get_available_localization_models.sync(client=client)
localization_models = res.models

res = get_available_segmentation_models.sync(client=client)
segmentation_models = res.models

def localization_type_change(model_name):
    selected_model = next((m for m in localization_models if m.name == model_name), None)
    if not selected_model:
        return [gr.update(choices=[], value=None)] * 2
    
    return [
        gr.update(choices=selected_model.variants, value=selected_model.variants[0] if selected_model.variants else None),
        gr.update(choices=selected_model.tasks, value=selected_model.tasks[0] if selected_model.tasks else None)
    ]

def segmentation_type_change(model_name):
    selected_model = next((m for m in segmentation_models if m.name == model_name), None)
    if not selected_model:
        return [gr.update(choices=[], value=None)] * 2
    
    return [
        gr.update(choices=selected_model.variants, value=selected_model.variants[0] if selected_model.variants else None),
        gr.update(choices=selected_model.tasks, value=selected_model.tasks[0] if selected_model.tasks else None)
    ]

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

                # Localization 
                with gr.Row():
                    model = gr.Dropdown(
                        choices=[m.name for m in localization_models],
                        label="Localization model",
                        value=localization_models[0].name if localization_models else None
                    )

                    variant = gr.Dropdown(
                        choices=localization_models[0].variants if localization_models else [],
                        label="Model variant",
                        interactive=True,
                        value=localization_models[0].variants[0] if localization_models and localization_models[0].variants else None
                    )

                    mode = gr.Dropdown(
                        choices=localization_models[0].tasks if localization_models else [],
                        label="Mode",
                        interactive=True,
                        value=localization_models[0].tasks[0] if localization_models and localization_models[0].tasks else None
                    )

                    model.change(fn=localization_type_change, inputs=[model], outputs=[variant, mode])

                # Segmentation
                with gr.Row():
                    model = gr.Dropdown(
                        choices=[m.name for m in segmentation_models],
                        label="Segmentation model",
                        value=segmentation_models[0].name if segmentation_models else None
                    )

                    variant = gr.Dropdown(
                        choices=segmentation_models[0].variants if segmentation_models else [],
                        label="Model variant",
                        interactive=True,
                        value=segmentation_models[0].variants[0] if segmentation_models and segmentation_models[0].variants else None
                    )

                    mode = gr.Dropdown(
                        choices=segmentation_models[0].tasks if segmentation_models else [],
                        label="Mode",
                        interactive=True,
                        value=segmentation_models[0].tasks[0] if segmentation_models and segmentation_models[0].tasks else None
                    )

                    model.change(fn=segmentation_type_change, inputs=[model], outputs=[variant, mode])
               
            with gr.Column():
                selected_section = gr.Textbox(label="Selected Section")
    
    section_btn.click(section, [img_input, prompt_input], img_output)
    img_output.select(select_section, None, selected_section)

if __name__ == "__main__":
    demo.launch()