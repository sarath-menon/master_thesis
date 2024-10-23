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

import os
from datetime import datetime

def create_coco_dataset(root_dir, output_file, do_rename=False):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    image_id = 1

    def get_creation_time(file_path):
        return os.path.getctime(file_path)

    
    if do_rename:
        image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append((os.path.join(root, file), get_creation_time(os.path.join(root, file))))
        
        image_files.sort(key=lambda x: x[1])  # Sort by creation time
        
        for index, (old_path, _) in enumerate(image_files):
            file_extension = os.path.splitext(old_path)[1]
            new_filename = f"{index+1}{file_extension}"
            new_path = os.path.join(os.path.dirname(old_path), new_filename)
            os.rename(old_path, new_path)
    
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
                    "height": height,
                    "metadata": {"instructions": ""}
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
root_directory = './datasets/resized_media/monopoly_images'
output_json = './datasets/monopoly_dataset/coco_annotations.json'

create_coco_dataset(root_directory, output_json)
# %%