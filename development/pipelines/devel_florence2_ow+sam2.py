#%%
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import center_of_mass
from clicking.pipeline.core import Clicker
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from matplotlib.path import Path
import matplotlib.patches as patches
import torch
from clicking.vision_model.types import SegmentationResp
from clicking.dataset_creator.core import CocoDataset
# import wandb

from dotenv import load_dotenv
load_dotenv()
# run this code only in notebook mode
if 'get_ipython' in globals():
    get_ipython().run_line_magic('matplotlib', 'inline')

#%%
from pydantic import BaseModel
from clicking.vision_model.types import *
from typing import List, Dict
import json
from datetime import datetime
from typing import Optional

class ExperimentTracker(BaseModel):
    images: Optional[List[Image.Image]] = []
    class_labels: Optional[List[str]] = []
    prediction_results: Optional[List[PredictionResp]] = []
    localization_results: Optional[List[LocalizationResp]] = []
    segmentation_results: Optional[List[SegmentationResp]] = []
    experiment_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_config = {
        "arbitrary_types_allowed": True
    }

    def log_images(self, images: List[Image.Image]):
        for i, image in enumerate(images):
            self.images.append(image)
        
        print(f"Logged {len(images)} images.")

    def log_localization(self, image_id: str, predictions: Dict[str, PredictionResp], categories: Dict[str, str]):
        self.localization_results[image_id] = {
            "predictions": {name: pred.dict() for name, pred in predictions.items()},
            "categories": categories
        }

    def log_segmentation(self, image_id: str, segmentation_resp: PredictionResp):
        self.segmentation_results[image_id] = segmentation_resp.dict()

    def save_results(self, filepath: str):
        results = {
            "experiment_id": self.experiment_id,
            "class_labels": self.class_labels,
            "localization_results": self.localization_results,
            "segmentation_results": self.segmentation_results
        }
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

    def load_results(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.experiment_id = data["experiment_id"]
        self.class_labels = data["class_labels"]
        self.localization_results = data["localization_results"]
        self.segmentation_results = data["segmentation_results"]

    def show_images(self):
        for (i, image) in enumerate(self.images):
            plt.imshow(image)
            plt.axis(False)
            plt.title(f"image_{i}")
            plt.show()

# Usage example:
exp_tracker = ExperimentTracker()
print(exp_tracker.experiment_id)

# # After segmentation
# tracker.log_segmentation(f"image_{0}", segmentation_resp)

# # Save results
# tracker.save_results(f"experiment_results_{tracker.experiment_id}.json")

# # Load results later
# loaded_tracker = ExperimentTracker(images=[], class_labels=[], prediction_results=[])
# loaded_tracker.load_results("experiment_results_20230501_120000.json")
#%%

# # Initialize W&B project
# wandb.init(project="clicking")

# Load dataset
coco_dataset = CocoDataset('./datasets/label_studio_gen/coco_dataset/images', './datasets/label_studio_gen/coco_dataset/result.json')
images, class_labels = coco_dataset.sample_dataset(batch_size=3)
exp_tracker.log_images(images)
exp_tracker.show_images()
#%%

#%% get clickable objects from image
from components.clicking.prompt_refinement.core import PromptRefiner, PromptMode

# Create an instance of PromptRefiner
prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

# Call process_prompts asynchronously
results = await prompt_refiner.process_prompts(images, PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS) 

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
import base64

client = Client(base_url="http://localhost:8082")

#%% Get available models
from clicking_client.api.default import get_models
from clicking_client.models  import SetModelReq
from clicking_client.api.default import set_model
from clicking.vision_model.types import TaskType

api_response = get_models.sync(client=client)
print(api_response)

#%% set localization model

request = SetModelReq(name="florence2", variant="florence-2-base", task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB)

set_model.sync(client=client, body=request)

#%% get localization prediction

from clicking_client.api.default import get_prediction
from clicking_client.models import BodyGetPrediction
from clicking_client.types import File
import io
import json
from clicking.visualization.bbox import BoundingBox
from clicking.vision_model.utils import image_to_http_file
from clicking.visualization.core import show_localization_predictions

all_predictions = []

for image, class_label, image_result in zip(images, class_labels, results):
    image_file:File = image_to_http_file(image)

    predictions = {}
    categories = {}

    for object in image_result['objects']:
        # Create the request object
        request = BodyGetPrediction(
            image=image_file,
            task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
            input_text=object['description']
        )

        predictions[object['name']] = get_prediction.sync(client=client, body=request)
        categories[object['name']] = object['category']

        all_predictions.append(predictions)

    show_localization_predictions(image, predictions, categories)

#%% verify bounding boxes
# convert bboxes to BoundingBox type
from clicking.visualization.bbox import BoundingBox, BBoxMode
import matplotlib.pyplot as plt
from clicking.visualization.core import overlay_bounding_box
from clicking.visualization.bbox import BoundingBox, BBoxMode
from clicking.output_corrector.core import OutputCorrector

for image, predictions in zip(images, all_predictions):
    for prediction in all_predictions:
        bboxes = prediction.bboxes
        bboxes = [BoundingBox((bbox[0], bbox[1], bbox[2], bbox[3]), BBoxMode.XYXY) for bbox in prediction.bboxes]

    overlayed_image = overlay_bounding_box(image.copy(), bboxes[0], thickness=10, padding=20)

    plt.grid(False)
    plt.axis('off')
    plt.imshow(overlayed_image)

# output_corrector = OutputCorrector()
# response = output_corrector.verify_bbox(overlayed_image, text_input)
# print(response)

#%% set segmentation model

request = SetModelReq(name="sam2", variant="sam2_hiera_tiny", task=TaskType.SEGMENTATION_WITH_BBOX)
set_model.sync(client=client, body=request)
#%% Segmentation
from clicking_client.api.default import get_prediction
from clicking_client.models import BodyGetPrediction
from clicking_client.types import File
import io
import json
from clicking.visualization.mask import SegmentationMask, SegmentationMode

# Convert PIL Image to bytes and create a File object
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")

# Create the request object
request = BodyGetPrediction(
    image=image_file,
    task=TaskType.SEGMENTATION_WITH_BBOX,
    input_boxes=json.dumps(prediction.bboxes)  # Convert bboxes to JSON string
)

segmentation_resp = get_prediction.sync(client=client, body=request)
print(f"inference time: {segmentation_resp.inference_time}")

prediction = segmentation_resp.prediction
masks = [SegmentationMask(mask, mode=SegmentationMode.COCO_RLE) for mask in prediction.masks]
show_segmentation_prediction(image, masks)

# %% verify masks

response = output_corrector.verify_mask(image, masks[0], text_input)
print(response)

# %% get click point

from clicking.visualization.core import show_clickpoint
from clicking.segmentation.utils import get_mask_centroid

centroid = get_mask_centroid(masks[0].get(mode=SegmentationMode.BINARY_MASK))
show_clickpoint(image, centroid, text_input)
