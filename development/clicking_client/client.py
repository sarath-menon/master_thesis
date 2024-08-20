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
from clicking_client.models import PredictionReq, SegmentationPredResp
import io
import base64

client = Client(base_url="http://localhost:8082")

#%% Get available models
from clicking_client.api.default import get_available_localization_models
api_response = get_available_localization_models.sync(client=client)
print(api_response)

#%% set localization model
from clicking_client.models  import SetModelRequest
from clicking_client.api.default import set_localization_model

request = SetModelRequest(name="florence2", variant="florence-2-base")

set_localization_model.sync(client=client, body=request)

#%% get localization prediction
from clicking_client.api.default import get_localization_prediction
from clicking_client.models import PredictionReq

request = PredictionReq(image=image_to_base64(image), text_input=text_input, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>')
localization_response = get_localization_prediction.sync(client=client, body=request)

print(f"inference time: {localization_response.inference_time}")
show_localization_prediction(image, localization_response.bboxes, localization_response.labels)

#%% verify bounding boxes
# convert bboxes to BoundingBox type
from clicking.visualization.bbox import BoundingBox, BBoxMode
import matplotlib.pyplot as plt
from clicking.visualization.core import overlay_bounding_box
from clicking.visualization.bbox import BoundingBox, BBoxMode
from clicking.output_corrector.core import OutputCorrector

bboxes = [BoundingBox((bbox[0], bbox[1], bbox[2], bbox[3]), BBoxMode.XYXY) for bbox in localization_response.bboxes]

overlayed_image = overlay_bounding_box(image, bboxes[0], thickness=10, padding=20)

plt.grid(False)
plt.axis('off')
plt.imshow(overlayed_image)

output_corrector = OutputCorrector()
response = output_corrector.get_feedback(image, text_input)
print(response)

#%% set segmentation model
from clicking_client.models  import SetModelRequest
from clicking_client.api.default import set_segmentation_model

request = SetModelRequest(name="sam2", variant="sam2_hiera_tiny")

set_segmentation_model.sync(client=client, body=request)
#%% Segmentation
from clicking_client.api.default import get_segmentation_prediction
from clicking_client.models import BodyGetSegmentationPrediction
from clicking_client.types import File
import io
import json

# Convert PIL Image to bytes and create a File object
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")

# Create the request object
request = BodyGetSegmentationPrediction(
    image=image_file,
    task_prompt='bbox',
    input_boxes=json.dumps(localization_response.bboxes)  # Convert bboxes to JSON string
)

segmentation_response = get_segmentation_prediction.sync(client=client, body=request)
print(f"inference time: {segmentation_response.inference_time}")

print(segmentation_response)

# %%
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import cv2
import matplotlib.pyplot as plt

# Assuming 'image' is your original PIL Image
image_array = np.array(image)

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Display the original image
ax.imshow(image)

borders = False
mask_alpha = 0.7

for mask in segmentation_response.masks:
    m = mask_utils.decode(mask)
    color_mask = np.random.random(3)

    # Create color overlay with correct shape and alpha channel
    color_overlay = np.zeros((*image_array.shape[:2], 4))
    color_overlay[m == 1] = [*color_mask, mask_alpha] 
    color_overlay[m == 0] = [0, 0, 0, 0]  
    ax.imshow(color_overlay)

    if borders:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=2)

ax.axis('off')
plt.tight_layout()
plt.show()
