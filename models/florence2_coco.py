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

DATA_DIR = './datasets/label_studio_gen/coco_dataset/images'
ANNOTATIONS_FILE = './datasets/label_studio_gen/coco_dataset/result.json'

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def run_example(image, model, processor, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


def load_dataset(data_dir, annFile):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    coco_dataset = torchvision.datasets.CocoDetection(root=data_dir, annFile=annFile, transform=transform)
    class_labels = [cat['name'] for cat in coco_dataset.coco.cats.values()]
    print(f"Dataset size: {len(coco_dataset)}")
    return coco_dataset, class_labels


def show(image, annotations, class_labels, click_point=None):
    """
    Plots an image from the COCO dataset along with its segmentation map.

    Args:
    image (PIL Image): The image to plot.
    annotations (list):z A list of annotations, where each annotation is a dictionary containing 'segmentation' and other keys.
    """

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0)  # Transpose the image tensor

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    

    for annotation in annotations:
        # plot class label
        class_label = class_labels[annotation['category_id']]

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


def load_model(model_id):
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
def create_text_input(annotations, class_labels):
    labels = [class_labels[annotation['category_id']] for annotation in annotations]
    text_input = ""
    text_input = ". ".join(labels) + "." if labels else ""
    return text_input

def check_click_points(annotations, click_points, class_labels):

    for annotation in annotations:
        class_label = class_labels[annotation['category_id']]
        
        # Filter click points matching the current class label
        matching_click_points = [cp for cp in click_points if cp['label'] == class_label]
        
        # Check if the click point is within the polygon of the annotation
        for click_point in matching_click_points:
            if check_point_in_polygon(annotation, click_point):
                # valid_click_points.append(click_point)
                click_point['valid'] = True
            else:
                print(f"Point lies outside the polygon for label: {class_label}")
                click_point['valid'] = False

        if len(matching_click_points) == 0:
            print(f"No click points found for label: {class_label}")

    return click_points
    
def check_point_in_polygon(annotation, click_point):
    point = Point(click_point['x'], click_point['y'])
    
    for segmentation in annotation['segmentation']:
        poly = np.array(segmentation).reshape((len(segmentation) // 2, 2))
        if Polygon(poly).contains(point):
            return True
    return False

def get_click_point(data):
    click_points = []
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox

        # get geometric center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        click_points.append({'x': center_x, 'y': center_y, 'label': label})

    return click_points


def main():
    coco_dataset, class_labels = load_dataset(DATA_DIR, ANNOTATIONS_FILE)
    model, processor = load_model('microsoft/Florence-2-base-ft')
    
    index = 2
    image, annotations = coco_dataset[index]
    text_input = create_text_input(annotations, class_labels)
    print(f"text input: {text_input}")

    to_pil = transforms.ToPILImage()
    image = to_pil(image.mul(255).byte())

    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    results = run_example(image, model, processor, task_prompt, text_input=text_input)

    click_points = get_click_point(results['<CAPTION_TO_PHRASE_GROUNDING>'])
    click_points = check_click_points(annotations, click_points, class_labels)
    show(image, annotations, class_labels, click_point=click_points)


if __name__ == "__main__":
    main()