#%%
# from datasets import load_dataset
import numpy as np
import os
import glob
from PIL import Image
import csv
import json

import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

import torchvision
from torchvision import transforms

# Define the path to the COCO dataset
data_dir = './datasets/label_studio_gen/coco_dataset/images'
annFile = './datasets/label_studio_gen/coco_dataset/result.json'

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create the COCO dataset
coco_dataset = torchvision.datasets.CocoDetection(root=data_dir, annFile=annFile, transform=transform)
class_labels = [cat['name'] for cat in coco_dataset.coco.cats.values()]
print(class_labels)


from matplotlib.path import Path
import matplotlib.patches as patches

def plot_click_point(image, click_point, annotations):
    
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

    colors = ['yellow', 'red']
    shapes = ['s', 'o']
    for i, point in enumerate(click_point):
        plt.plot(point['x'], point['y'], shapes[i],  color=colors[i], markersize=10)

    plt.axis('off')
    plt.grid(False)
    plt.show()
#%%

# Example usage:

# plot_click_point(img, [], annotations)

from openai import OpenAI
import os
import io
import torchvision.transforms as transforms
import base64

def image_to_base64(img):
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img.mul(255).byte())
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_shifted_point(img, annotations, response):
    img_width = img.shape[1]
    img_height = img.shape[2]

    distances = json.loads(response.choices[0].message.content)
    print("Distance: ", distances)

    # x_shift = float(distances['x']) * img_width
    # y_shift = float(distances['y']) * img_height
    # print(f"img_width: {img_width}, img_height: {img_height}")
    # print(f"x_shift: {x_shift}, y_shift: {y_shift}")

    # click_points = []
    
    # for annotation in annotations:
    #     # plot class label
    #     class_label_x = annotation['segmentation'][0][0]
    #     class_label_y = annotation['segmentation'][0][1] 
    #     click_points.append({'x': class_label_x - x_shift, 'y': class_label_y - y_shift})
            
    # return click_points


MODEL = "gpt-4o"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

system_prompt = "You are a helpful assistant and an identifying objects in videogame images."

def generate_user_prompt(object_label: str):
    return f"""Consider the object labelled as {object_label} in the image. First, find the geometric center of the {object_label} and the first letter of the text label saying '{object_label}'. Then, how should the label be shifted from its current position such that the top left corner of the text label falls in the geometric center of the {object_label} ? Give distance and direction to be shifted, where the horizontal unit of distance is the width of the label and the vertical unit of distance is the height of the label. Choose the top left corner of the label as the reference point. Choose the direction from one of the following: up, down, left, right. Also give a reason for choosing the direction. Give the output in json format with the following keys: x_distance, x_direction, y_direction, y_distance, reason.
"""

index = 2
img, annotations = coco_dataset[index]
LABEL = class_labels[annotations[0]['category_id']]
print(LABEL)
user_prompt = generate_user_prompt(LABEL)
base64_image = image_to_base64(img)

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


shifted_click_points = get_shifted_point(img, annotations, response)
# plot_click_point(img, shifted_click_points, annotations)
plot_click_point(img, [], annotations)
# %%
