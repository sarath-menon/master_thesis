#%%
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
from clicking.visualization.core import show_localization_prediction, show_segmentation_prediction
from clicking.pipeline.core import Clicker
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from matplotlib.path import Path
import matplotlib.patches as patches
import torch
from clicking.vision_model.types import SegmentationResp
from dotenv import load_dotenv

load_dotenv()
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
all_class_labels = [cat['name'] for cat in coco_dataset.coco.cats.values()]
print(f"Dataset size: {len(coco_dataset)}")

# create text input from labels
def create_text_input(annotations):
    labels = [all_class_labels[annotation['category_id']] for annotation in annotations]
    text_input = ""
    text_input = ". ".join(labels) + "." if labels else ""
    return text_input

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def sample_dataset(indices=None, batch_size=None):
    if indices is not None and batch_size is not None:
        raise ValueError("Only one of 'indices' or 'batch_size' should be provided, not both.")
    
    if indices is None and batch_size is None:
        batch_size = 1
    
    if indices is None:
        indices = np.random.choice(len(coco_dataset), size=batch_size, replace=False)

    images = []
    text_inputs = []
    to_pil = transforms.ToPILImage()
    
    for index in indices:
        image_tensor, annotations = coco_dataset[int(index)]
        image = to_pil(image_tensor)
        text_input = create_text_input(annotations)
        images.append(image)
        text_inputs.append(text_input)
    
    return images, text_inputs


#%% sample batch for testing
images, class_labels = sample_dataset(batch_size=3)

# plot first 4 images
for image, text_input in zip(images, class_labels):
    plt.imshow(image)
    plt.axis(False)
    plt.title(text_input)
    plt.show()

#%% get clickable objects from image
from components.clicking.prompt_refinement.core import PromptRefiner, PromptMode

# Create an instance of PromptRefiner
prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

# Call process_prompts asynchronously
async def process_batch_prompts():
    results = await prompt_refiner.process_prompts(images, PromptMode.IMAGE_TO_CLASS_LABEL)
    return results
results = await process_batch_prompts()

# sort objects by category
for result in results:
    result['objects'] = sorted(result['objects'], key=lambda x: x['category'])

# show results
for image, class_label, image_result in zip(images, class_labels, results):
    plt.imshow(image)
    plt.axis(False)
    plt.title(class_label)
    plt.show()

    for object in image_result['objects']:
        print(f"name: {object['name']}")
        print(f"category: {object['category']}")
        print(f"description: {object['description']}")
        print("-" * 50)
#%% get extended object descriptions from short descriptions

# from components.clicking.prompt_refinement.core import PromptRefiner, PromptMode
# import json

# class_labels = text_input.split()
# images = [image for _ in class_labels]

# # Call process_prompts asynchronously
# prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")
# async def process_batch_prompts():
#     results = await prompt_refiner.process_prompts(images, class_labels, PromptMode.EXPANDED_DESCRIPTION)
#     return results

# refined_text_inputs = await process_batch_prompts()
# refined_text_inputs

#%%
from clicking_client import Client
from clicking_client.models import PredictionResp
import io
import base64

client = Client(base_url="http://localhost:8082")

#%% Get available models
from clicking_client.api.default import get_models
from clicking_client.models  import SetModelReq
from clicking_client.api.default import set_model
from clicking.vision_model.types import TaskType

api_response = get_models.sync(client=client)
print(api_response)

#%% set model

request = SetModelReq(name="evf_sam2", variant="sam2", task=TaskType.SEGMENTATION_WITH_TEXT)
set_model.sync(client=client, body=request)

#%% get masks batch 

from clicking_client.api.default import get_prediction
from clicking_client.models import BodyGetPrediction
from clicking_client.types import File
import io
import json
from clicking.visualization.mask import SegmentationMask, SegmentationMode
# from clicking.visualization.core import show_clickpoint_predictions

# Convert PIL Image to bytes and create a File object
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")

predictions = {}

for class_label, text_input in refined_text_inputs.items():
    # Create the request object
    request = BodyGetPrediction(
        image=image_file,
        task=TaskType.SEGMENTATION_WITH_TEXT,
        input_text=text_input 
    )
    
    predictions[class_label] = get_prediction.sync(client=client, body=request)

show_clickpoint_predictions(image, predictions)
