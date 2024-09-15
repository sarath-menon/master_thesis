#%%
from pycocotools.coco import COCO
import numpy as np
# import skimage.io as io
import matplotlib.pyplot as plt
import pylab

%matplotlib inline
#%%

images_path= './datasets/label_studio_gen/coco_dataset/images'
annotations_path= './datasets/label_studio_gen/coco_dataset/result.json'
coco=COCO(annotations_path)
# %% Get all image IDs
import random

img_ids = coco.getImgIds()
# Sample a random image
sample_size = 3  # Change this to sample more images
sampled_img_ids = random.sample(img_ids, sample_size)

#%% Load and display the sampled image(s)
import os

for img_id in sampled_img_ids:
    img = coco.loadImgs(img_id)[0]

    absolute_images_path = os.path.abspath(images_path)
    image_file_path = os.path.join(absolute_images_path, img['file_name'])
    I = plt.imread(image_file_path)
    plt.imshow(I)
    plt.axis('off')
    plt.show()

#%% crete COCO dataset

import os
import json
from PIL import Image

def create_coco_dataset(root_dir, output_file):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    image_id = 1
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                
                with Image.open(image_path) as img:
                    width, height = img.size
                
                relative_path = os.path.relpath(image_path, root_dir)
                
                coco_format["images"].append({
                    "id": image_id,
                    "file_name": relative_path,
                    "width": width,
                    "height": height
                })
                
                image_id += 1
    
    # Add a dummy category (you can modify this as needed)
    coco_format["categories"].append({
        "id": 1,
        "name": "default",
        "supercategory": "none"
    })
    
    output_file_abs = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file_abs), exist_ok=True)
    with open(output_file_abs, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"COCO dataset created and saved to {output_file}")

# Usage
root_directory = './datasets/resized_media/ui_images'
output_json = './datasets/ui_dataset/coco_annotations.json'

create_coco_dataset(root_directory, output_json)
# %%