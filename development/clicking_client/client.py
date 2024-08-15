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
# from shapely.geometry import Point, Polygon

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

#%% Create clicker object

# api = Clicker()

#%%
import clicking_client
from pprint import pprint

# Create a client instance
configuration = clicking_client.Configuration(
    host = "http://localhost:8082"
)

api_client = clicking_client.ApiClient(configuration)
client = clicking_client.DefaultApi(api_client)



#%% Get available models
api_response = client.get_available_localization_models()
print(api_response)

#%% set model
from clicking_client import SetModelRequest

request = SetModelRequest(name="florence2", variant="florence-2-base")

client.set_localization_model(request)
#%% Localization
# from clicking_client import PredictionReq, PredictionResp
# import io
# import base64

index = 4
image_tensor, annotations = coco_dataset[index]
to_pil = transforms.ToPILImage()
image = to_pil(image_tensor)
text_input = create_text_input(annotations)
print(text_input)


# req = PredictionReq(image=image_to_base64(image), text_input=text_input, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>')
# response = client.get_localization_prediction(req)
# response.read()
#%%

import requests

# Define the URL for the API call
url = "http://localhost:8082/localization/prediction"

# Make the GET request to the API
response = requests.get(url, json={"image": image_to_base64(image), 
  "text_input": "plant.", 
  "task_prompt": "<CAPTION_TO_PHRASE_GROUNDING>"
})

# Check if the request was successful
if response.status_code == 200:
    print("Successfully retrieved model information.")
    model_info = response.json()
    pprint(model_info)
else:
    print("Failed to retrieve model information. Status code:", response.status_code)

#%%

from clicking_client import Client
from clicking_client.models import PredictionReq, PredictionResp
from clicking_client.api.default import get_localization_prediction
import io
import base64

client = Client(base_url="http://localhost:8082")

response = get_localization_prediction.sync(client=client, body=PredictionReq(image=image_to_base64(image), text_input=text_input, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>'))

print(response)
show_localization_prediction(image, response.bboxes, response.labels)
#%% Segmentation

input_boxes = response['bboxes']
results = api.get_segmentation_prediction(image, input_boxes)
masks = np.array(results['masks'])
centroids = api.get_mask_centroid(masks)
show_segmentation_prediction(image, masks, input_boxes, centroids)
#%%
