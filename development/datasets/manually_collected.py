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

#%% Set image names to indices and scale resolution

raw_images_path = "./datasets/raw_media/gameplay_images"
resized_images_path = "./datasets/resized_media/gameplay_images"

# resolution scale factor
scale_factor = 0.5

# Iterate over each folder in the images path
for folder in os.listdir(raw_images_path):
    # Check if the current item is a directory
    if os.path.isdir(os.path.join(raw_images_path, folder)):
       
        # Get all jpg files in the current folder
        image_files = glob.glob(os.path.join(raw_images_path, folder, '*.jpg')) + glob.glob(os.path.join(raw_images_path, folder, '*.png'))
        
        # Iterate over each file in the current folder
        for i, file in enumerate(image_files):
            
            # Scale the resolution of the image
            img = Image.open(file)
            img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
            
            # Save the image with the same extension as the original file
            file_extension = os.path.splitext(file)[1]
            save_folder = os.path.join(resized_images_path, folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            img.save(os.path.join(save_folder, f'{i}{file_extension}'))
            
            # # Print the image resolution
            # print(f'Image resolution: {img.size[0]} x {img.size[1]}')


#%% Create JSON task file for label-studio

import os
import json

# Create tasks list to store the JSON tasks
tasks = []

resized_images_path = "./datasets/resized_media/gameplay_images"

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
                image_path = "/data/local-files/?d=" + image_path

                task = {
                    "data": {
                        "img": image_path
                    },
                    "annotations": [],
                    "predictions": []
                }
                print(task)
                tasks.append(task)

# Save the tasks as a JSON file
with open("label_studio_tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

#%% List image file paths of existing tasks in JSON

import json

label_studio_gen_folder = "/Users/sarathmenon/Documents/master_thesis/datasets/label_studio_gen"

import os

# Get the list of .json files in the label_studio_gen_folder
json_files = [f for f in os.listdir(label_studio_gen_folder) if f.endswith('.json')]

# If there's more than one file, throw an error
if len(json_files) > 1:
    raise ValueError("More than one .json file found in the label_studio_gen_folder")

config_file = os.path.join(label_studio_gen_folder, json_files[0])

with open(config_file, 'r') as f:
    image_files = []
    data = json.load(f)
    for entry in data:
        print(entry['data']['img'])
        image_files.append(entry['data']['img'])


# ### Add new images rto exsiting label-studio files

# In[437]:


import os
import json

import json

label_studio_gen_folder = "/Users/sarathmenon/Documents/master_thesis/datasets/label_studio_gen"
resized_images_path = "../datasets/resized_media/gameplay_images"

new_image_files =[] 
new_tasks = []

# Get the list of .json files in the label_studio_gen_folder
json_files = [f for f in os.listdir(label_studio_gen_folder) if f.endswith('.json')]

# If there's more than one file, throw an error
if len(json_files) > 1:
    raise ValueError("More than one .json file found in the label_studio_gen_folder")

config_file = os.path.join(label_studio_gen_folder, json_files[0])

with open(config_file, 'r') as f:
    image_files = []
    data = json.load(f)
    for entry in data:
        # print(entry['data']['img'])
        image_files.append(entry['data']['img'])


# Iterate through each saved image and create a JSON task
for folder in os.listdir(resized_images_path):
    # Check if the current item is a directory
    if os.path.isdir(os.path.join(resized_images_path, folder)):
        for i, image_file in enumerate(os.listdir(os.path.join(resized_images_path, folder))):

            # Check if the current file is a jpg or jpeg
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
                image_path = os.path.join(resized_images_path, folder, image_file)
                image_path = "/data/local-files/?d=" + image_path

                if image_path not in image_files:
                    # new_image_files.append(image_path)
                    print(f"image: {image_path} is missing")
                    task = {
                        "data": {
                            "img": image_path
                        },
                        "annotations": [],
                    }
                    new_tasks.append(task)

# Open the existing JSON file
with open(config_file, 'r') as f:
    data = json.load(f)

# Append the new tasks to the existing data
data.extend(new_tasks)

# Write the updated data back to the file
with open(config_file, 'w') as f:
    json.dump(data, f, indent=2)


# ## Create dummy keypoints

# In[11]:


resized_images_path = "../datasets/resized_media/gameplay_images"

# Iterate over each folder in the images path
for folder in os.listdir(resized_images_path):
    # Check if the current item is a directory
    if os.path.isdir(os.path.join(resized_images_path, folder)):
       
        # Get all jpg and png files in the current folder
        image_files = glob.glob(os.path.join(resized_images_path, folder, '*.jpg')) + glob.glob(os.path.join(resized_images_path, folder, '*.png'))
        
        # Create a keypoint file for the current folder
        keypoint_file = os.path.join(resized_images_path, folder, 'keypoints.csv')
        
        with open(keypoint_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'x1', 'y1'])  # Write the header row
            
            # Iterate over each file in the current folder
            for file in image_files:
                # Get the image name
                image_name = os.path.basename(file)
                
                # Write the image name and keypoints to the file
                writer.writerow([image_name, 40, 40])


# # Create pytorch dataset

# ### From JSON file exported from human-label with keypoint annotation for each class

# In[87]:


import os
from torch.utils.data import Dataset
from PIL import Image

class GameClickDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[96]:


resized_images_path = "../datasets/resized_media/gameplay_images/hogwarts_legacy"
keypoints_file = resized_images_path + "/keypoints.csv"

# Define transforms
data_transforms = transforms.Compose([
    transforms.ToTensor()  # Convert PIL Image to tensor
])

def transform_sample(sample):
    image = data_transforms(sample['image'])
    landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
    return {'image': image, 'landmarks': landmarks}

click_dataset = GameClickDataset(csv_file=keypoints_file,
                                    root_dir=resized_images_path,
                                    transform=transform_sample)
dataloader = DataLoader(click_dataset, batch_size=32, shuffle=True)


def show_landmarks(index, title=None):
    """Show image with landmarks"""
    image = click_dataset[index]['image']
    landmarks = click_dataset[index]['landmarks']
    title = f'Sample #{index}'

    image = image.permute(1, 2, 0)  # Transpose the image tensor
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=80, marker='x', c='r')
    plt.axis('off')  # Disable axes
    plt.title(title)

show_landmarks(index=3, title=None)


# ### From CSV file exported from human-label with keypoint annotation for each class

# In[215]:


class GameClickDataset(Dataset):
    """Dataset for game click points from JSON."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Arguments:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_file) as f:
            self.data = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data= self.data[idx]
        img_name = img_data['img'].split('?d=')[-1]
        image = io.imread(img_name)

        landmarks = []
        labels = []

        if 'kp-1' in img_data:
            for kp in img_data['kp-1']:
                width = kp['original_width']
                height = kp['original_height']
                x = kp['x']/100. * width
                y = kp['y']/100. * height
                landmark = np.array([x, y], dtype=float)
                landmarks.append(landmark)
                labels.append(kp['keypointlabels'])

        sample = {'image': image, 'landmarks': landmarks, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[253]:


# resized_images_path = "../datasets/resized_media/gameplay_images"
json_file = resized_images_path + "/label_studio_output.json"

# Define transforms
data_transforms = transforms.Compose([
    transforms.ToTensor()  # Convert PIL Image to tensor
])

def transform_sample(sample):
    image = data_transforms(sample['image'])
    landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
    return {'image': image, 'landmarks': landmarks, 'labels': sample['labels']}



click_dataset = GameClickDataset(json_file=json_file,
                                 root_dir=resized_images_path,
                                 transform=transform_sample)
dataloader = DataLoader(click_dataset, batch_size=32, shuffle=True)

def show_landmarks(index, title=None):
    """Show image with landmarks"""
    image = click_dataset[index]['image']
    landmarks = click_dataset[index]['landmarks']
    labels = click_dataset[index]['labels']
    title = f'Sample #{index}'

    image = image.permute(1, 2, 0)  # Transpose the image tensor
    plt.imshow(image)
    if landmarks.size(0) > 0:
        print(f"Labels: {labels}")
        for i in range(landmarks.size(0)):
            plt.scatter(landmarks[i, 0], landmarks[i, 1], s=80, marker='*', c='w')

            label = labels[i][0]
            plt.text(landmarks[i, 0] - len(label)*15, landmarks[i, 1] - 80, label, fontsize=12, color='yellow', bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')  # Disable axes
    plt.title(title)

show_landmarks(index=0, title=None)


# ### From human-label exported file with COCO stuyle image segmentation info for rectangles and polygons

# In[343]:


import torchvision
from torchvision import transforms

# Define the path to the COCO dataset
data_dir = '../datasets/label_studio_gen/coco_dataset/images'
annFile = '../datasets/label_studio_gen/coco_dataset/result.json'

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create the COCO dataset
coco_dataset = torchvision.datasets.CocoDetection(root=data_dir, annFile=annFile, transform=transform)
class_labels = [cat['name'] for cat in coco_dataset.coco.cats.values()]
print(class_labels)


# In[375]:


from matplotlib.path import Path
import matplotlib.patches as patches

def plot_coco_image_with_segmentation(image, annotations):
    """
    Plots an image from the COCO dataset along with its segmentation map.

    Args:
    image (PIL Image): The image to plot.
    annotations (list): A list of annotations, where each annotation is a dictionary containing 'segmentation' and other keys.
    """
    
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
    plt.axis('off')
    plt.grid(False)
    plt.show()

# Example usage:
index = 11
img, annotations = coco_dataset[index]
plot_coco_image_with_segmentation(img, annotations)


# In[345]:




