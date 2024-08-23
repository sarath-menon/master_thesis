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
#%% set segmentation model

request = SetModelReq(name="sam2", variant="sam2_hiera_large", task=TaskType.SEGMENTATION_AUTO_ANNOTATION)
set_model.sync(client=client, body=request)

#%% set image and text input
index = 6
image_tensor, annotations = coco_dataset[index]
to_pil = transforms.ToPILImage()
image = to_pil(image_tensor)
text_input = create_text_input(annotations)
print(text_input)

plt.imshow(image)
plt.axis('off')
plt.show()
#%% Segmentation
from clicking_client.api.default import get_auto_annotation
from clicking_client.models import BodyGetAutoAnnotation
from clicking_client.types import File
import io
import json
from clicking.visualization.mask import SegmentationMask, SegmentationMode

import requests
from io import BytesIO

def get_image_from_url(url):
  response = requests.get(url)
  return Image.open(BytesIO(response.content))

# image = Image.open('images/cars.jpg')
# image = get_image_from_url("https://nichegamer.com/wp-content/uploads/2022/12/hogwarts-legacy-12-18-22-1.jpg")
# image = get_image_from_url("https://i.ytimg.com/vi/5kcdRBHM7kM/maxresdefault.jpg")

# Convert PIL Image to bytes and create a File object
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")


# Create the request object
body = BodyGetAutoAnnotation(
    image=image_file,
)

segmentation_resp = get_auto_annotation.sync(client=client, body=body, task=TaskType.SEGMENTATION_AUTO_ANNOTATION)
print(f"inference time: {segmentation_resp.inference_time}")

prediction = segmentation_resp.prediction
masks = [SegmentationMask(mask['segmentation'], mode=SegmentationMode.COCO_RLE) for mask in prediction.masks]

show_segmentation_prediction(image, masks)
print(f"Number of masks: {len(masks)}")

# %% verify masks

import numpy as np
import matplotlib.pyplot as plt

image_np = np.array(image)

def crop_using_bbox(image_np, bbox, padding=10):
    x1, y1, width, height = map(int, bbox)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)

    width += 2 * padding
    height += 2 * padding

    x2 = x1 + width
    y2 = y1 + height
    cropped_image = image_np[y1:y2, x1:x2]
    return cropped_image


mask_count = len(prediction.masks)
plt.figure(figsize=(10, 3 * mask_count))  # Adjust the figure size dynamically for a single column layout

for i, mask in enumerate(prediction.masks):
    plt.subplot(mask_count, 1, i + 1)  # Set all plots in a single column

    # Crop bounding box around segmented pixels
    cropped_image = crop_using_bbox(image_np, mask['bbox'])

    plt.imshow(cropped_image)
    plt.title(f"i: {i}, area: {str(mask['area'])}")
    plt.axis('off')

plt.show()

#%% plot all masks

mask_count = len(prediction.masks)
plt.figure(figsize=(10, 3 * mask_count))  # Adjust the figure size dynamically for a single column layout

for i, mask in enumerate(masks):
    plt.subplot(mask_count, 1, i + 1)  # Set all plots in a single column
    plt.imshow(mask.get(mode=SegmentationMode.BINARY_MASK), cmap='gray')  # Assuming mask is a numpy array
    plt.axis('off')
    plt.title(f"i: {i}")
    plt.tight_layout()
plt.show()
