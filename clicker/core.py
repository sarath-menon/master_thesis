
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont 
import requests
import copy
import torchvision
from torchvision import transforms
from matplotlib.path import Path
import torch
from shapely.geometry import Point, Polygon


class Clicker:
    def __init__(self, class_labels):
        self.class_labels = class_labels
        self.model, self.processor = self.load_model('microsoft/Florence-2-base-ft')

    def get(self, image, annotations):
        text_input = self.create_text_input(annotations)
        print(f"text input: {text_input}")

        to_pil = transforms.ToPILImage()
        image = to_pil(image.mul(255).byte())

        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = self.run_example(image, task_prompt, text_input=text_input)

        click_points = self.get_click_point(results['<CAPTION_TO_PHRASE_GROUNDING>'])
        click_points = self.check_click_points(annotations, click_points)
        return click_points

    def run_example(self, image, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer


    def show(self, image, annotations, click_point=None):

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0)  # Transpose the image tensor

        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()
        
        for annotation in annotations:
            # plot class label
            class_label = self.class_labels[annotation['category_id']]

            class_label_x = annotation['segmentation'][0][0]
            class_label_y = annotation['segmentation'][0][1] 
            plt.text(class_label_x, class_label_y, class_label, fontsize=14, color='yellow')

            # plot segmentation map
            for segmentation in annotation['segmentation']:
                poly = np.array(segmentation).reshape((len(segmentation) // 2, 2))
                poly_path = Path(poly)
                patch = patches.PathPatch(poly_path, facecolor='blue', edgecolor='red', linewidth=2, alpha=0.6)
                ax.add_patch(patch)

        if click_point:
            for point in click_point:
                color = 'yellow' if point['valid'] else 'red'
                shape = '*' if point['valid'] else 'x'
                plt.plot(point['x'], point['y'], shape, color=color, markersize=10)
        
        plt.axis('off')
        plt.grid(False)
        plt.show()


    def load_model(self, model_id):
        from transformers.dynamic_module_utils import get_imports
        from unittest.mock import patch

        def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
            if not str(filename).endswith("/modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports.remove("flash_attn")
            return imports

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor


    # create text input from labels
    def create_text_input(self, annotations):
        labels = [self.class_labels[annotation['category_id']] for annotation in annotations]
        text_input = ""
        text_input = ". ".join(labels) + "." if labels else ""
        return text_input

    def check_click_points(self, annotations, click_points):

        for annotation in annotations:
            class_label = self.class_labels[annotation['category_id']]
            
            # Filter click points matching the current class label
            matching_click_points = [cp for cp in click_points if cp['label'] == class_label]
            
            # Check if the click point is within the polygon of the annotation
            for click_point in matching_click_points:
                if self.check_point_in_polygon(annotation, click_point):
                    # valid_click_points.append(click_point)
                    click_point['valid'] = True
                else:
                    print(f"Point lies outside the polygon for label: {class_label}")
                    click_point['valid'] = False

            if len(matching_click_points) == 0:
                print(f"No click points found for label: {class_label}")

        return click_points
        
    def check_point_in_polygon(self, annotation, click_point):
        point = Point(click_point['x'], click_point['y'])
        
        for segmentation in annotation['segmentation']:
            poly = np.array(segmentation).reshape((len(segmentation) // 2, 2))
            if Polygon(poly).contains(point):
                return True
        return False

    def get_click_point(self, data):
        click_points = []
        for bbox, label in zip(data['bboxes'], data['labels']):
            # Unpack the bounding box coordinates
            x1, y1, x2, y2 = bbox

            # get geometric center of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            click_points.append({'x': center_x, 'y': center_y, 'label': label})

        return click_points