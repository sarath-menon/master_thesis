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
import uuid

class ImagePredictionResult(BaseModel):
    id: int
    image: Optional[Image.Image] = None
    description: Optional[List[Dict]] = []
    localization_result: Optional[LocalizationResp] = None
    segmentation_result: Optional[SegmentationResp] = None

    model_config = {
        "arbitrary_types_allowed": True
    }


class ExperimentTracker(BaseModel):
    results: Optional[List[ImagePredictionResult]] = []
    experiment_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_image(self, id: int, image: Image.Image):
        for result in self.results:
            if result.id == id:
                print(f"Image with id {id} already exists.")
                return
        self.results.append(ImagePredictionResult(id=id, image=image))

    def get_image_ids(self):
        return (result.id for result in self.results)
    
    def add_image_descriptions(self, descriptions: list):
        for result, description in zip(self.results, descriptions):
            result.description = description

    def add_localization_result(self, image_id: str, prediction: PredictionResp):
        for result in self.results:
            if result.id != image_id:
                continue
            result.localization_result = prediction
            return

    def add_segmentation_result(self, image_id: str, segmentation_resp: PredictionResp):
        for result in self.results:
            if result.id != image_id:
                continue
            result.segmentation_result = segmentation_resp
            return

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
        for (i, result) in enumerate(self.results):
            plt.imshow(result.image)
            plt.axis(False)
            plt.title(f"image_{i}")
            plt.show()

    def show_image_descriptions(self):
        for (i, result) in enumerate(self.results):
            print(f"image_{i}")
            print(result.description)

    def generate_print_functions(self):
        def create_print_function(field_name):
            def print_function(self, image_ids: List[str]):
                for image_id in image_ids:
                    self._print_field_for_image(image_id, field_name)
            return print_function

        fields = [field for field in ImagePredictionResult.__fields__ if field != 'id']
        for field in fields:
            setattr(ExperimentTracker, f"print_{field}", create_print_function(field))

    def _print_field_for_image(self, image_id: str, field_name: str):
        result = self._get_result_by_id(image_id)
        if not result:
            print(f"No result found for image ID: {image_id}")
            return

        value = getattr(result, field_name)
        print(f"Image ID: {image_id}, {field_name.capitalize()}:")
        self._display_value(value)
        print("-" * 50)

    def _get_result_by_id(self, image_id: str):
        return next((r for r in self.results if r.id == image_id), None)

    def _display_value(self, value):
        if isinstance(value, Image.Image):
            plt.imshow(value)
            plt.axis('off')
            plt.show()
        elif isinstance(value, list):
            for item in value:
                self._display_value(item)
        elif isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, list):
                    print(f"{key}:")
                    for item in val:
                        self._display_value(item)
                        print("-" * 50)
                else:
                    print(f"{key}: {val}")
        else:
            print(value)

    def generate_getter_functions(self):
        def create_getter_function(field_name):
            def getter_function(self, image_ids: List[int]):
                return [getattr(self._get_result_by_id(image_id), field_name) for image_id in image_ids if self._get_result_by_id(image_id)]
            return getter_function

        fields = [field for field in ImagePredictionResult.__fields__ if field != 'id']
        for field in fields:
            setattr(ExperimentTracker, f"get_{field}", create_getter_function(field))

    def _get_result_by_id(self, image_id: int):
        return next((r for r in self.results if r.id == image_id), None)


# Usage example:
exp_tracker = ExperimentTracker()
exp_tracker.generate_print_functions()
exp_tracker.generate_getter_functions()
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

#%% sample dataset

image_ids = [22, 31, 34]
images, class_labels = coco_dataset.sample_dataset(indices=image_ids)

for id, image in zip(image_ids, images):
    exp_tracker.add_image(id, image)

exp_tracker.print_image(image_ids)

# %% get clickable objects from image
# from components.clicking.prompt_refinement.core import PromptRefiner, PromptMode

# # Create an instance of PromptRefiner
# prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

# # Call process_prompts asynchronously
# prompt_refiner_results = await prompt_refiner.process_prompts(images, PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS) 

for image, class_label, image_result in zip(images, class_labels, prompt_refiner_results):
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

exp_tracker.add_image_descriptions(prompt_refiner_results)
exp_tracker.print_description(image_ids)

#%%
from clicking_client import Client
from clicking_client.models import PredictionResp
import io

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
# from clicking.visualization.core import show_localization_predictions

for result in exp_tracker.results:

    image = result.image
    image_file:File = image_to_http_file(image)

    predictions = {}
    descriptions = {}
    categories = {}

    for object in result.description['objects']:
        # Create the request object
        request = BodyGetPrediction(
            image=image_file,
            task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
            input_text=object['description']
        )

        predictions[object['name']] = get_prediction.sync(client=client, body=request)
        categories[object['name']] = object['category']
        descriptions[object['name']] = object['description']

    exp_tracker.add_localization_result(result.id, predictions)
    show_localization_predictions(image, predictions, categories, descriptions)

#%% verify bounding boxes
# convert bboxes to BoundingBox type
from clicking.visualization.bbox import BoundingBox, BBoxMode
import matplotlib.pyplot as plt
from clicking.visualization.core import overlay_bounding_box
from clicking.visualization.bbox import BoundingBox, BBoxMode
from clicking.output_corrector.core import OutputCorrector

localization_results =  exp_tracker.get_localization_result(image_ids)
descriptions = exp_tracker.get_description(image_ids)
images = exp_tracker.get_image(image_ids)

for image, description, localization_results in zip(images, descriptions, localization_results):
    # show_localization_predictions(image, localization_result, categories)
    print(image)

    for object_name, prediction in localization_results.items():
        print(f"object_name: {object_name}")
        bboxes_list = []

        for bbox in prediction.prediction.bboxes:
            bboxes_list.append(BoundingBox((bbox[0], bbox[1], bbox[2], bbox[3]), BBoxMode.XYXY))

        if len(bboxes_list) == 0:
            continue
        
        overlayed_image = overlay_bounding_box(image, bboxes_list[0], thickness=10, padding=20)

    plt.figure()
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
