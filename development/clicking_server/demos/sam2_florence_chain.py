#%%
import base64
import io
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import center_of_mass
from model_utils.visualization import show_localization_prediction, show_segmentation_prediction
import matplotlib.pyplot as plt

class ClickingAPI:
    def __init__(self, server_url="http://localhost:8082"):
        self.server_url = server_url
        self.detection_endpoint = f"{server_url}/localization"
        self.segmentation_endpoint = f"{server_url}/segmentation"

    def _encode_image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_localization_prediction(self, image, text_input):
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        img_str = self._encode_image_to_base64(image)
        payload = {
            "image": img_str,
            "text_input": text_input,
            "task_prompt": task_prompt
        }
        response = requests.get(self.detection_endpoint, json=payload)
        return response.json()

    def get_mask_centroid(self, mask):
        centroids = []
        for mask in masks:
            centroid = center_of_mass(mask)
            centroids.append(centroid)
        return centroids

    def get_segmentation_prediction(self, image, input_boxes):
        img_str = self._encode_image_to_base64(image)
        payload = {
            "image": img_str,
            "input_boxes": input_boxes
        }
        response = requests.get(self.segmentation_endpoint, json=payload)
        return response.json()
#%% Localization
api = ClickingAPI()
image = Image.open('images/truck.jpg')
text_input = "tire."
response = api.get_localization_prediction(image, text_input)
show_localization_prediction(image, response)

#%% Segmentation
input_boxes = response['bboxes']
results = api.get_segmentation_prediction(image, input_boxes)
masks = np.array(results['masks'])
centroids = api.get_mask_centroid(masks)
show_segmentation_prediction(image, masks, input_boxes, centroids)


# %%
image