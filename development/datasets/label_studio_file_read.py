#%%

import json
import os

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from pycocotools import mask as mask_utils

from clicking.vision_model.visualization import show_segmentation_predictions
from clicking.vision_model.types import SegmentationResults, SegmentationMask, ImageWithDescriptions
from clicking.common.mask import SegmentationMode

class Point(BaseModel):
    x: float
    y: float

class PolygonLabel(BaseModel):
    points: List[Point]
    closed: bool
    polygonlabels: List[str]

class AnnotationResult(BaseModel):
    original_width: int
    original_height: int
    image_rotation: int
    value: PolygonLabel
    id: str
    from_name: str
    to_name: str
    type: str
    origin: str

class Annotation(BaseModel):
    id: int
    result: List[AnnotationResult]
    created_at: datetime
    updated_at: datetime
    lead_time: float

class ImageData(BaseModel):
    img: str

class ImageAnnotation(BaseModel):
    id: int
    annotations: List[Annotation]
    data: ImageData
    created_at: datetime
    updated_at: datetime
    inner_id: int
    total_annotations: int

class ImageAnnotationCollection(BaseModel):
    images: List[ImageAnnotation]

# Function to convert points from [x, y] format to Point objects
def convert_points(points_list):
    return [Point(x=point[0], y=point[1]) for point in points_list]

# Function to parse the JSON data
def parse_json_data(json_data):
    images = []
    for item in json_data:
        annotations = []
        for ann in item.get('annotations', []):
            results = []
            for res in ann.get('result', []):
                polygon_label = PolygonLabel(
                    points=convert_points(res['value']['points']),
                    closed=res['value']['closed'],
                    polygonlabels=res['value']['polygonlabels']
                )
                results.append(AnnotationResult(
                    original_width=res.get('original_width', 0),
                    original_height=res.get('original_height', 0),
                    image_rotation=res.get('image_rotation', 0),
                    value=polygon_label,
                    id=res.get('id', ''),
                    from_name=res.get('from_name', ''),
                    to_name=res.get('to_name', ''),
                    type=res.get('type', ''),
                    origin=res.get('origin', '')
                ))
            annotations.append(Annotation(
                id=ann.get('id', 0),
                result=results,
                created_at=ann.get('created_at', datetime.now()),
                updated_at=ann.get('updated_at', datetime.now()),
                lead_time=ann.get('lead_time', 0.0)
            ))
        images.append(ImageAnnotation(
            id=item.get('id', 0),
            annotations=annotations,
            data=ImageData(img=item['data']['img']),
            created_at=item.get('created_at', datetime.now()),
            updated_at=item.get('updated_at', datetime.now()),
            inner_id=item.get('inner_id', 0),
            total_annotations=item.get('total_annotations', 0)
        ))
    return ImageAnnotationCollection(images=images)

#%%
from clicking.vision_model.types import ImageWithDescriptions
from urllib.parse import urlparse, unquote
from pycocotools import mask as mask_utils

def extract_image_path(url):
    parsed = urlparse(url)
    query = parsed.query
    if query.startswith('d='):
        path = unquote(query[2:])
        if path.startswith('..'):
            path = path[2:]  # Remove leading '..'
        return f".{path}"
    return url

def polygon_to_rle(polygon, height, width):
    rles = mask_utils.frPyObjects([polygon], height, width)
    rle = mask_utils.merge(rles)
    return rle



def convert_to_segmentation_results(parsed_data):
    segmentation_results = SegmentationResults(processed_samples=[], predictions={})
    
    for image_annotation in parsed_data.images:
        image_path = extract_image_path(image_annotation.data.img)
        image = Image.open(image_path)
        
        # Get the first polygon label as the object name
        object_name = "Unknown"
        for annotation in image_annotation.annotations:
            for result in annotation.result:
                if result.value.polygonlabels:
                    object_name = result.value.polygonlabels[0]
                    break
            if object_name != "Unknown":
                break
        
        segmentation_results.processed_samples.append(
            ImageWithDescriptions(image=image, id=image_annotation.id, object_name=object_name)
        )
        
        masks = []
        for annotation in image_annotation.annotations:
            for result in annotation.result:
                points = result.value.points
                img_width, img_height = image.size
                
                # Convert percentage points to pixel coordinates
                pixel_points = [
                    (point.x * img_width / 100, point.y * img_height / 100) 
                    for point in points
                ]
                
                # Convert to COCO RLE format
                polygon = [coord for point in pixel_points for coord in point]
                rle = mask_utils.frPyObjects([polygon], img_height, img_width)
                rle = mask_utils.merge(rle)
                
                mask = SegmentationMask(
                    mask=rle,
                    mode=SegmentationMode.COCO_RLE,
                    object_name=result.value.polygonlabels[0] if result.value.polygonlabels else "Unknown",
                    description=""
                )
                masks.append(mask)
        
        segmentation_results.predictions[image_annotation.id] = masks
    
    return segmentation_results

#%%
from clicking.vision_model.types import SegmentationResults, SegmentationMask
LABEL_STUDIO_EXPORTED_FILE = "./datasets/label_studio_gen/label_studio_tasks_export.json"

LABEL_STUDIO_EXPORTED_DATASET = "./datasets/label_studio_gen/coco_dataset"

# Set the number of samples to be shown
num_samples = 2

tasks_file = os.path.join(LABEL_STUDIO_EXPORTED_FILE)
with open(tasks_file, 'r') as file:
    json_data = json.load(file)

parsed_data = parse_json_data(json_data)

# Convert parsed data to SegmentationResults
segmentation_results = convert_to_segmentation_results(parsed_data)


# Create a new SegmentationResults object with filtered samples and predictions
filtered_samples = segmentation_results.processed_samples[:num_samples]
filtered_predictions = {
    sample.id: segmentation_results.predictions[sample.id]
    for sample in filtered_samples
}

filtered_results = SegmentationResults(
    processed_samples=filtered_samples,
    predictions=filtered_predictions
)

# Plot segmentation maps using the show_segmentation_predictions function
show_segmentation_predictions(filtered_results)

# %%
import json
from typing import Dict, List
import base64

def save_annotations_to_json(segmentation_results: SegmentationResults, output_file: str):
    data = []
    
    for sample in segmentation_results.processed_samples:
        image_id = sample.id
        image_path = sample.image.filename  # Assuming the PIL Image object has a filename attribute
        
        annotations = []
        for mask in segmentation_results.predictions[image_id]:
            rle = mask.get(mode=SegmentationMode.COCO_RLE)
            # Convert RLE to base64 string
            rle_base64 = base64.b64encode(rle['counts']).decode('utf-8')
            annotation = {
                "segmentation": {
                    "counts": rle_base64,
                    "size": rle['size']
                },
                "object_name": mask.object_name,
                "description": mask.description
            }
            annotations.append(annotation)
        
        image_data = {
            "image_id": image_id,
            "image_path": image_path,
            "annotations": annotations
        }
        data.append(image_data)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Annotations saved to {output_file}")

# After creating filtered_results, add:
output_file = "./datasets/label_studio_gen/annotations.json"
save_annotations_to_json(filtered_results, output_file)


