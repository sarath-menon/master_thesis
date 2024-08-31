#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from dotenv import load_dotenv
from clicking.vision_model.visualization import show_localization_predictions, show_segmentation_predictions
from clicking.common.types import PipelineState, ClickingImage, ImageObject
from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction, get_models
from clicking.vision_model.types import TaskType
import io
import base64
import json
from dataclasses import dataclass, field
from typing import List

load_dotenv()

@dataclass
class PipelineState:
    images: List[ClickingImage] = field(default_factory=list)

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
object_names = [cat['name'] for cat in coco_dataset.coco.cats.values()]
print(f"Dataset size: {len(coco_dataset)}")

def create_text_input(annotations):
    labels = [object_names[annotation['category_id']] for annotation in annotations]
    return ". ".join(labels) + "." if labels else ""

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

#%% set image and text input
index = 5
image_tensor, annotations = coco_dataset[index]
to_pil = transforms.ToPILImage()
image = to_pil(image_tensor)
text_input = create_text_input(annotations)
print(text_input)

#%%
client = Client(base_url="http://localhost:8083")

#%% Get available models
api_response = get_models.sync(client=client)
print(api_response)

#%% set localization model
request = SetModelReq(name="florence2", variant="florence-2-base", task=TaskType.LOCALIZATION_WITH_TEXT_GROUNDED)
set_model.sync(client=client, body=request)

#%% get localization prediction
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")

request = BodyGetPrediction(
    image=image_file,
    task=TaskType.LOCALIZATION_WITH_TEXT_GROUNDED,
    input_text=text_input
)
localization_resp = get_prediction.sync(client=client, body=request)
print(f"inference time: {localization_resp.inference_time}")

prediction = localization_resp.prediction
bboxes = [BoundingBox(bbox=bbox, mode=BBoxMode.XYXY) for bbox in prediction.bboxes]

# Create ClickingImage and ImageObject instances
clicking_image = ClickingImage(id=index, image=image)
for bbox, label in zip(bboxes, prediction.labels):
    clicking_image.objects.append(ImageObject(name=label, bbox=bbox))

# Create PipelineState
pipeline_state = PipelineState(images=[clicking_image])

# Visualize localization results
show_localization_predictions(clicking_image)

#%% set segmentation model
request = SetModelReq(name="sam2", variant="sam2_hiera_tiny", task=TaskType.SEGMENTATION_WITH_BBOX)
set_model.sync(client=client, body=request)

#%% Segmentation
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")

request = BodyGetPrediction(
    image=image_file,
    task=TaskType.SEGMENTATION_WITH_BBOX,
    input_boxes=json.dumps([obj.bbox.get(mode=BBoxMode.XYXY) for obj in clicking_image.objects])
)

segmentation_resp = get_prediction.sync(client=client, body=request)
print(f"inference time: {segmentation_resp.inference_time}")

prediction = segmentation_resp.prediction
for obj, mask in zip(clicking_image.objects, prediction.masks):
    obj.mask = SegmentationMask(coco_rle=mask, mode=SegmentationMode.COCO_RLE)

# Update PipelineState
pipeline_state.images[0] = clicking_image

# Visualize segmentation results
show_segmentation_predictions(clicking_image)

#%% verify masks
output_corrector = OutputCorrector()
for obj in clicking_image.objects:
    response = output_corrector.verify_mask(image, obj.mask, obj.name)
    print(f"Verification for {obj.name}: {response}")

#%% get click point
from clicking.segmentation.utils import get_mask_centroid
from clicking.visualization.core import show_clickpoint

for obj in clicking_image.objects:
    centroid = get_mask_centroid(obj.mask.get(mode=SegmentationMode.BINARY_MASK))
    obj.click_point = centroid
    show_clickpoint(image, centroid, obj.name)

# Final PipelineState with all information
print(pipeline_state)