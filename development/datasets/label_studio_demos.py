#%%

# from datasets import load_dataset
import numpy as np
import os
import glob
from PIL import Image
import csv

import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

#%%
load_dotenv()
LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL')
API_KEY = os.getenv('API_KEY')

# Connect to the Label Studio API and check the connection
ls_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
projects_list = ls_client.projects.list()
for project in projects_list:
    print(project)

## Get tasks list for specific project

## Get project fron label studio
project_title = "demo"
project_id = None

for project in projects_list:
    if project.title == project_title:
        project_id = project.id
        break

if project_id is None:
    raise ValueError(f"Project with title '{project_title}' not found.")

# Get the project
project = ls_client.projects.get(project_id)
print(f"Project title: {project.title}")

# get the project tasks
tasks = ls_client.tasks.list(project=1)

#%% Get label studio JSON export file
import json

export_generator = ls_client.projects.exports.create_export(project_id,download_all_tasks=True, download_resources=True, ids=[1,2,3])

responses = ""
for response in export_generator:
    print(response)
    responses += response.decode('utf-8')  # Decode bytes to string

responses_json = json.loads(responses)
print(responses_json)

PATH = os.path.join(os.getcwd(), 'generated/label_studio')
if not os.path.exists(PATH):
    os.makedirs(PATH)
with open(os.path.join(PATH, 'label_studio_tasks_export.json'), 'w') as f:
    json.dump(responses_json, f)
#%% Create tasks list 

import os
import json

tasks = []

resized_images_path = "datasets/resized_media/gameplay_images"

# Iterate through each saved image and create a JSON task
for folder in os.listdir(resized_images_path):
    # Check if the current item is a directory
    if os.path.isdir(os.path.join(resized_images_path, folder)):
        for i, image_file in enumerate(os.listdir(os.path.join(resized_images_path, folder))):
            
            # Delete all files with the extension '.DS_Store'
            if image_file.endswith('.DS_Store'):
                os.remove(os.path.join(resized_images_path, folder, image_file))
                continue

            # Check if the current file is a jpg or jpeg
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
                image_path = os.path.join(resized_images_path, folder, image_file)
                image_path = "./data/local-files/?d=" + image_path

                task = {
                    "data": {
                        "img": image_path
                    },
                    "annotations": [],
                    "predictions": []
                }
                print(task)
                tasks.append(task)
print(tasks)

#%% Set image names to indices and scale resolution

ls_client.projects.import_tasks(
    id=project_id,
    request=tasks
)
#%%