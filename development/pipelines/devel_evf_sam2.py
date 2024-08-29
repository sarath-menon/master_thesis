#%%
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
from clicking.vision_model.core import show_localization_prediction, show_segmentation_prediction
from clicking.pipeline.core import Clicker
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from matplotlib.path import Path
import matplotlib.patches as patches
import torch
from clicking.vision_model.types import SegmentationResp
from clicking.dataset_creator.core import CocoDataset
from dotenv import load_dotenv

load_dotenv()
# run this code only in notebook mode
if 'get_ipython' in globals():
    get_ipython().run_line_magic('matplotlib', 'inline')

#%% Load dataset

coco_dataset = CocoDataset('./datasets/label_studio_gen/coco_dataset/images', './datasets/label_studio_gen/coco_dataset/result.json')
images, class_labels = coco_dataset.sample_dataset(batch_size=3)
    
#%% sample batch for testing
images, class_labels = coco_dataset.sample_dataset(batch_size=3, show_images=True)

#%% get clickable objects from image
from components.clicking.prompt_refinement.core import PromptRefiner, PromptMode

# Create an instance of PromptRefiner
prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

# Call process_prompts asynchronously
results = await prompt_refiner.process_prompts(images, PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS) 

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
        # print(f"Reasoning: {object['reasoning']}")
        print("-" * 50)
#%%
from clicking_client import Client
from clicking_client.models import PredictionResp
import io

client = Client(base_url="http://localhost:8083")

#%% Get available models
from clicking_client.api.default import get_models
from clicking_client.models  import SetModelReq
from clicking_client.api.default import set_model

api_response = get_models.sync(client=client)
print(api_response)
#%% set model
from clicking.vision_model.types import TaskType

request = SetModelReq(name="evf_sam2", variant="sam2", task=TaskType.SEGMENTATION_WITH_TEXT)
set_model.sync(client=client, body=request)

#%% get masks batch 

from clicking_client.api.default import get_prediction
from clicking_client.models import BodyGetPrediction
import io
from clicking.vision_model.types import TaskType
from clicking.vision_model.mask import SegmentationMask, SegmentationMode
from clicking.vision_model.core import show_clickpoint_predictions
from clicking.vision_model.utils import image_to_http_file
from clicking_client.types import File

# def image_to_http_file(image) -> File:
#     # Convert PIL Image to bytes and create a File object
#     image_byte_arr = io.BytesIO()
#     image.save(image_byte_arr, format='JPEG')
#     image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")
#     return image_file


for image, class_label, image_result in zip(images, class_labels, results):
    image_file:File = image_to_http_file(image)
    predictions = {}

    for object in image_result['objects']:
        # Create the request object
        request = BodyGetPrediction(
            image=image_file,
            task=TaskType.SEGMENTATION_WITH_TEXT,
            input_text=object['description']
        )

        predictions[object['name']] = get_prediction.sync(client=client, body=request)

    show_clickpoint_predictions(image, predictions)
#%%