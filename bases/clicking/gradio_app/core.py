import gradio as gr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from clicking_client import Client
from clicking_client.models import PredictionReq, PredictionResp
from clicking_client.api.default import get_available_localization_models, get_localization_prediction, get_available_segmentation_models,set_localization_model, set_segmentation_model, get_segmentation_prediction
from clicking.segmentation.utils import get_mask_centroid
import io
import base64
import sys
from gradio_log import Log
from clicking_client.models  import SetModelReq
from clicking_client.models import BodyGetSegmentationPrediction
from clicking_client.types import File
import io
import json
from pycocotools import mask as mask_utils
from dataclasses import dataclass, field
import math

css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""
DESCRIPTION = "# [Florence-2 Demo](https://huggingface.co/microsoft/Florence-2-large)"


@dataclass
class PipelineResult:
    input_instruction: str = ""
    object_name: str = ""
    bounding_boxes: list = field(default_factory=list)  # Changed to use default_factory
    segmentation_masks: list = field(default_factory=list)  # Changed to use default_factory
    click_point: tuple = ()
pipeline_result = PipelineResult()

def test(x):
    print("This is a test")
    print(f"Your function is running with input {x}...")
    return x

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

def select_section(evt: gr.SelectData):
        return pipeline_result.object_name[evt.index]

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

tasks = {
    "input instruction => class labels":"",
    "class labels => bounding boxes":"",
    "bounding boxes => segmentation masks":"",
    "segmentation masks => click point":"",
}

import math

def generate_star_points(centroid, size=20):
    x, y = centroid
    points = []
    # There are 10 points in a 5-point star
    for i in range(10):
        angle = math.pi / 2 + (i * 2 * math.pi / 10)
        if i % 2 == 0:
            # Outer point
            x = x + size * math.cos(angle)
            y = y - size * math.sin(angle)
        else:
            # Inner point (halfway toward the center)
            x = x + (size / 2) * math.cos(angle)
            y = y - (size / 2) * math.sin(angle)
        points.append((x, y))
    return points

def pipeline(image_np, input_instruction, object_name, pipeline_checkboxes):
        if image_np is None:
            raise ValueError("Set input image")
        elif input_instruction=='' and object_name=='':
            raise ValueError("Set input text or class labels")
        elif input_instruction!='' and object_name!='':
            raise ValueError("You can't set both input text and class labels")

        # convert numpy array to PIL image
        image = Image.fromarray(image_np)

        print(pipeline_checkboxes)

        pipeline_result.input_instruction = input_instruction
        
        # convert input instruction to class labels
        if "input instruction => class labels" in pipeline_checkboxes:
            pass

        pipeline_result.object_name = object_name
        # get localization prediction
        if "class labels => bounding boxes" in pipeline_checkboxes:
            request = PredictionReq(image=image_to_base64(image), text_input=object_name, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>')

            localization_response = get_localization_prediction.sync(client=client, body=request)

        pipeline_result.bounding_boxes = localization_response.bboxes
        
        # get segmentation prediction
        if "bounding boxes => segmentation masks" in pipeline_checkboxes:
            image_byte_arr = io.BytesIO()
            image.save(image_byte_arr, format='JPEG')
            image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")

            # Create the request object
            request = BodyGetSegmentationPrediction(
                image=image_file,
                task_prompt='bbox',
                input_boxes=json.dumps(localization_response.bboxes)  # Convert bboxes to JSON string
            )

            segmentation_response = get_segmentation_prediction.sync(client=client, body=request)

            pipeline_result.segmentation_masks = segmentation_response.masks
            # get click point prediction from segmentation masks
            if "segmentation masks => click point" in pipeline_checkboxes:
                pass

            sections = []

            for i, bbox in enumerate(localization_response.bboxes):
                x1,y1,w,h = map(int, bbox)
                sections.append(((x1,y1,w,h), object_name + "_bbox"))

            for (i, mask) in enumerate(segmentation_response.masks):
                mask = mask_utils.decode(mask)
                sections.append((mask, object_name + "_mask"   ))

        # get and overlay click point
        if "segmentation masks => click point" in pipeline_checkboxes:
            decoded_masks = [mask_utils.decode(mask) for mask in segmentation_response.masks]
            for i, mask in enumerate(decoded_masks):
                # Create a blank image with the same dimensions
                mask_image = Image.new('L', (image.width, image.height), 0)
                draw_mask = ImageDraw.Draw(mask_image)

                centroid = get_mask_centroid(mask)
                star_point = generate_star_points(centroid, size=10)

                # create a mask for the click point
                draw_mask.polygon(star_point, fill=1)
                mask_np = np.array(mask_image)
                sections.append((mask_np, "click point"))

        return (image, sections)

label_creation_models = {
    "GPT-4": "",
    "GPT-4o": "",
}


client = Client(base_url="http://localhost:8082")
res = get_available_localization_models.sync(client=client)
localization_models = res.models

res = get_available_segmentation_models.sync(client=client)
segmentation_models = res.models

# Redirect stdout to a log file
log_file = "./output.log"
# sys.stdout = open(log_file, "a")

def model_type_change(model_name, model_type):
    models = localization_models if model_type == "localization" else segmentation_models
    selected_model = next((m for m in models if m.name == model_name), None)
    if not selected_model:
        return [gr.update(choices=[], value=None)] * 2
    
    return [
        gr.update(choices=selected_model.variants, value=selected_model.variants[0] if selected_model.variants else None),
        gr.update(choices=selected_model.tasks, value=selected_model.tasks[0] if selected_model.tasks else None)
    ]

def localization_model_variant_change(model, variant, mode):
    print(model, variant, mode)
    request = SetModelReq(name=model, variant=variant)
    set_localization_model.sync(client=client, body=request)

def segmentation_model_variant_change(model, variant, mode):
    print(model, variant, mode)
    request = SetModelReq(name=model, variant=variant)
    set_segmentation_model.sync(client=client, body=request)

# gradio state variables
model_type_localization = gr.State(value="localization")
model_type_segmentation = gr.State(value="segmentation")

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Florence-2 Image Captioning"):
        with gr.Row():
            img_input = gr.Image()
            img_output = gr.AnnotatedImage(
                color_map={"click point": "#a89a00"}
            )

        with gr.Row():
            pipeline_checkboxes = gr.CheckboxGroup(
                choices=list(tasks.keys()),
                label="Pipeline",
                value=list(tasks.keys()), interactive=True
            )

        with gr.Row():
            with gr.Column():
                section_btn = gr.Button("Identify Sections")

                input_instruction = gr.Textbox(label="Input instruction", interactive=True)
                object_name = gr.Textbox(label="Class labels", interactive=True)
            
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

                    model.change(fn=model_type_change, inputs=[model, model_type_localization], outputs=[variant, mode])
                    variant.change(fn=localization_model_variant_change, inputs=[model, variant, mode], outputs=[])

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

                    model.change(fn=model_type_change, inputs=[model, model_type_segmentation], outputs=[variant, mode])
                    variant.change(fn=segmentation_model_variant_change, inputs=[model, variant, mode], outputs=[])
               
            with gr.Column():
                selected_section = gr.Textbox(label="Selected Section")
    
        # Log(log_file, dark=True, xterm_font_size=12)
    
    section_btn.click(pipeline, [img_input, input_instruction, object_name, pipeline_checkboxes], img_output)
    img_output.select(select_section, None, pipeline_result.object_name)

if __name__ == "__main__":
    demo.launch()