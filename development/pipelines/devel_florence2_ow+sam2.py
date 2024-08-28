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

#%% show results
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

    show_localization_predictions(image, predictions, categories)

#%% verify bounding boxes
# convert bboxes to BoundingBox type
from clicking.visualization.bbox import BoundingBox, BBoxMode
import matplotlib.pyplot as plt
from clicking.visualization.core import overlay_bounding_box
from clicking.visualization.bbox import BoundingBox, BBoxMode
from clicking.output_corrector.core import OutputCorrector

bboxes = [BoundingBox((bbox[0], bbox[1], bbox[2], bbox[3]), BBoxMode.XYXY) for bbox in prediction.bboxes]

overlayed_image = overlay_bounding_box(image.copy(), bboxes[0], thickness=10, padding=20)

plt.grid(False)
plt.axis('off')
plt.imshow(overlayed_image)

output_corrector = OutputCorrector()
response = output_corrector.verify_bbox(overlayed_image, text_input)
print(response)

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
