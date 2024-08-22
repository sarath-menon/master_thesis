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
class_labels = [cat['name'] for cat in coco_dataset.coco.cats.values()]
print(f"Dataset size: {len(coco_dataset)}")

# create text input from labels
def create_text_input(annotations):
    labels = [class_labels[annotation['category_id']] for annotation in annotations]
    text_input = ""
    text_input = ". ".join(labels) + "." if labels else ""
    return text_input

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
#%% set image and text input
index = 5
image_tensor, annotations = coco_dataset[index]
to_pil = transforms.ToPILImage()
image = to_pil(image_tensor)
text_input = create_text_input(annotations)
print(text_input)

#%%
from clicking_client import Client
from clicking_client.models import PredictionResp
import io
import base64

client = Client(base_url="http://localhost:8083")

#%% Get available models
from clicking_client.api.default import get_models
from clicking_client.models  import SetModelReq
from clicking_client.api.default import set_model
from clicking.vision_model.types import TaskType


api_response = get_models.sync(client=client)
print(api_response)

#%% set localization model

request = SetModelReq(name="florence2", variant="florence-2-base", task=TaskType.LOCALIZATION_WITH_TEXT_GROUNDED)

set_model.sync(client=client, body=request)

#%% get localization prediction

from clicking_client.api.default import get_prediction
from clicking_client.models import BodyGetPrediction
from clicking_client.types import File
import io
import json
from clicking.visualization.bbox import BoundingBox

# Convert PIL Image to bytes and create a File object
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")

# Create the request object
request = BodyGetPrediction(
    image=image_file,
    task= TaskType.LOCALIZATION_WITH_TEXT_GROUNDED,
    input_text=text_input
)
localization_resp = get_prediction.sync(client=client, body=request)
print(f"inference time: {localization_resp.inference_time}")

prediction = localization_resp.prediction
show_localization_prediction(image.copy(), prediction.bboxes, prediction.labels)

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
# %%

## EVF_SAM2

request = SetModelReq(name="evf_sam2", variant="sam2", task=TaskType.SEGMENTATION_WITH_TEXT)
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
    task=TaskType.SEGMENTATION_WITH_TEXT,
    input_text=text_input
)
    
segmentation_resp = get_prediction.sync(client=client, body=request)
print(f"inference time: {segmentation_resp.inference_time}")

prediction = segmentation_resp.prediction
masks = [SegmentationMask(mask, mode=SegmentationMode.COCO_RLE) for mask in prediction.masks]
show_segmentation_prediction(image, masks)
