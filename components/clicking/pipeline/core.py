import base64
import io
import numpy as np
import requests
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass

class Clicker:
    def __init__(self, server_url="http://localhost:8082"):
        self.server_url = server_url
        self.detection_endpoint = f"{server_url}/localization"
        self.segmentation_endpoint = f"{server_url}/segmentation"

    def _encode_image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_localization_prediction(self, image, text_input, type='caption_to_phrase'):
        
        if type == 'open_vocabulary':
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        elif type == 'caption_to_phrase':
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>' 
        else:
            raise ValueError(f"Invalid type: {type}")
            
        img_str = self._encode_image_to_base64(image)
        payload = {
            "image": img_str,
            "text_input": text_input,
            "task_prompt": task_prompt
        }
        response = requests.get(self.detection_endpoint, json=payload)
        return response.json()

    def get_mask_centroid(self, masks):
        centroids = []
        for mask in masks:
            centroid = center_of_mass(mask)
            centroid = (centroid[2], centroid[1])
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