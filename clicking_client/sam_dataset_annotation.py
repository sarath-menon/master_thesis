#%%
import base64
import io
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
from model_utils.visualization import show_localization_prediction, show_segmentation_prediction
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
import copy
import os
import random
from torchvision import transforms, datasets
from matplotlib.path import Path
import matplotlib.patches as patches
import torch
from shapely.geometry import Point, Polygon

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

# def get_model_prediction(image, text_input, task_prompt, url):
#     # Convert PIL image to base64
#     buffered = io.BytesIO()
#     image.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

#     # Prepare JSON payload
#     payload = {
#         "image": img_str,
#         "text_input": text_input,
#         "task_prompt": task_prompt
#     }
    
#     # Make API call
#     response = requests.get(url, json=payload)
#     return response.json()
#%%
class ClickingAPI:
    def __init__(self, server_url="http://localhost:8082"):
        self.server_url = server_url
        self.detection_endpoint = f"{server_url}/localization"
        self.segmentation_endpoint = f"{server_url}/segmentation"

    def _encode_image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_localization_prediction(self, image, text_input, type='caption_to_phrase'):
        
        if type == 'open_vocabulary':
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        elif type == 'caption_to_phrase':
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>' 
        else:
            raise ValueError(f"Invalid type: {type}")
            
        img_str = self._encode_image_to_base64(image)
        payload = {
            "image": img_str,
            "text_input": text_input,
            "task_prompt": task_prompt
        }
        response = requests.get(self.detection_endpoint, json=payload)
        return response.json()

    def get_mask_centroid(self, mask):
        centroids = []
        for mask in masks:
            centroid = center_of_mass(mask)
            centroid = (centroid[1], centroid[0])
            centroids.append(centroid)
        return centroids

    def get_segmentation_prediction(self, image, input_boxes):
        img_str = self._encode_image_to_base64(image)
        payload = {
            "image": img_str,
            "input_boxes": input_boxes
        }
        response = requests.get(self.segmentation_endpoint, json=payload)
        return response.json()

api = ClickingAPI()
#%% get class labels for annotation
from openai import OpenAI

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

MODEL = "gpt-4o"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_chat_response(user_prompt,image):
    system_prompt = "You are a helpful assistant and an expert in image annotation."

    base64_image = image_to_base64(image)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        stream=False,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return response
#%%

index = 6
image_tensor, annotations = coco_dataset[index]
to_pil = transforms.ToPILImage()
image = to_pil(image_tensor.mul(255).byte())
plt.imshow(image)
plt.axis('off')

#%% get class labels for annotation
import json

user_prompt = """Identify the three key objects in this video game screenshot that are crucial for semantic segmentation data annotation. Explain why each object is significant in 20 words or less. Select only objects that belog to the following categories: 'game user interface', 'game object'. Omit the player and background while labeling. Format your response as JSON with the keys:{label, category, reason}. 
"""

response = get_chat_response(user_prompt, image)
response_json = json.loads(response.choices[0].message.content)
print(response_json)
#%%

text_input = ""
for item in response_json['objects']:
    text_input += f"{item['label']}. "
print(text_input)

#%% Localization
text_input = "a flag"

response = api.get_localization_prediction(image, text_input, type='open_vocabulary')
show_localization_prediction(image, response)
response
#%% Segmentation
input_boxes = response['bboxes']
results = api.get_segmentation_prediction(image, input_boxes)
masks = np.array(results['masks'])
centroids = api.get_mask_centroid(masks)
show_segmentation_prediction(image, masks, input_boxes, centroids)
#%%
