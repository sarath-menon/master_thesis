#%%
import base64
import io
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
from clicking.visualization.core import show_localization_prediction, show_segmentation_prediction
from clicking.pipeline.core import Clicker
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
import copy
import os
import random
from torchvision import transforms, datasets
from matplotlib.path import Path
import matplotlib.patches as patches
import torch
# from shapely.geometry import Point, Polygon

# run this code only in notebook mode
if 'get_ipython' in globals():
    get_ipython().run_line_magic('matplotlib', 'inline')

#%%
# Define the path to the COCO dataset
data_dir = './datasets/label_studio_gen/coco_dataset/images'
annFile = './datasets/label_studio_gen/coco_dataset/result.json'

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create the COCO dataset
coco_dataset = datasets.CocoDetection(root=data_dir, annFile=annFile, transform=transform)
class_labels = [cat['name'] for cat in coco_dataset.coco.cats.values()]
print(f"Dataset size: {len(coco_dataset)}")

# to plot the image with segmentation map and click points
def show(image, annotations, click_point=None, show_segmentation=True):
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
        if show_segmentation:
            for segmentation in annotation['segmentation']:
                poly = np.array(segmentation).reshape((len(segmentation) // 2, 2))
                poly_path = Path(poly)
                patch = patches.PathPatch(poly_path, facecolor='blue', edgecolor='red', linewidth=2, alpha=0.6)
                ax.add_patch(patch)

    if click_point:
        for point in click_point:
            color = 'yellow' if point['valid'] else 'blue'
            shape = '*' if point['valid'] else 'x'
            plt.plot(point['x'], point['y'], shape, color=color, markersize=10)
    
    plt.axis('off')
    plt.grid(False)
    plt.show()

# create text input from labels
def create_text_input(annotations):
    labels = [class_labels[annotation['category_id']] for annotation in annotations]
    text_input = ""
    text_input = ". ".join(labels) + "." if labels else ""
    return text_input

## Check if predicted click point lies within the bounding boxes
def check_click_points(annotations, click_points, verbose=True):

    for annotation in annotations:
        class_label = class_labels[annotation['category_id']]
        
        # Filter click points matching the current class label
        matching_click_points = [cp for cp in click_points if cp['label'] == class_label]
        
        # Check if the click point is within the polygon of the annotation
        for click_point in matching_click_points:

            # if the click point is already valid, then skip it since its the correct click point of another object
            if 'valid' in click_point and click_point['valid']:
                continue

            if check_point_in_polygon(annotation, click_point):
                click_point['valid'] = True
            else:
                click_point['valid'] = False

        if len(matching_click_points) == 0:
            print(f"No click points found for label: {class_label}")

    # printing
    if verbose:
        valid_click_points = [cp['label'] for cp in click_points if cp['valid']]
        invalid_click_points = [cp['label'] for cp in click_points if not cp['valid']]
        print(f"valid click points: {valid_click_points}")
        print(f"invalid click points: {invalid_click_points}")

    return click_points
    
def check_point_in_polygon(annotation, click_point):
    point = Point(click_point['x'], click_point['y'])
    
    for segmentation in annotation['segmentation']:
        poly = np.array(segmentation).reshape((len(segmentation) // 2, 2))
        if Polygon(poly).contains(point):
            return True
    return False

def get_click_point(bboxes, labels):
    click_points = []
    for bbox, label in zip(bboxes, labels):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox

        # get geometric center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        click_points.append({'x': center_x, 'y': center_y, 'label': label})

    return click_points

#%% Create clicker object

api = Clicker()
#%% Localization
index = 4
image_tensor, annotations = coco_dataset[index]
to_pil = transforms.ToPILImage()
image = to_pil(image_tensor)
text_input = create_text_input(annotations)
print(text_input)
response = api.get_localization_prediction(image, text_input)
show_localization_prediction(image, response)

#%% Segmentation

input_boxes = response['bboxes']
results = api.get_segmentation_prediction(image, input_boxes)
masks = np.array(results['masks'])
centroids = api.get_mask_centroid(masks)
show_segmentation_prediction(image, masks, input_boxes, centroids)
#%%
