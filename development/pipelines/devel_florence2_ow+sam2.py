#%%

%load_ext autoreload
%autoreload 2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from clicking.pipeline.core import Pipeline
from clicking.dataset_creator.core import CocoDataset
from clicking.prompt_refinement.core import PromptRefiner
from clicking.vision_model.types import TaskType
from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction
from clicking.common.types import *
import pickle
import os
from datetime import datetime
import asyncio
from typing import Callable, Union
import inspect
from dataclasses import dataclass, field
import yaml
from clicking.vision_model.visualization import show_localization_predictions, show_segmentation_predictions
from clicking_client.types import File
from io import BytesIO
import json
import nest_asyncio
from clicking.pipeline.core import PipelineState
# Load the configuration file
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

#%%
client = Client(base_url=config['api']['base_url'])

prompt_refiner = PromptRefiner(prompt_path=config['prompts']['refinement_path'], config=config)

coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

#%%
def image_to_http_file(image):
    # Convert PIL Image to bytes and create a File object
    image_byte_arr = BytesIO()
    image.save(image_byte_arr, format='JPEG')
    image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")
    return image_file

# Modify the relevant steps to use PipelineState

def sample_dataset(state: PipelineState) -> PipelineState:
    state.images = coco_dataset.sample_dataset(state.images)
    return state

def process_prompts(state: PipelineState) -> PipelineState:
    for clicking_image in state.images:
        clicking_image = prompt_refiner.process_prompts(clicking_image)
    return state

from prettytable import PrettyTable

def verify_bboxes(state: PipelineState) -> PipelineState:
    results = []
    for clicking_image in state.images:
        clicking_image = output_corrector.verify_bboxes(clicking_image)
        for obj in clicking_image.predicted_objects:
            result = {
                'object_name': obj.name,
                'judgement': 'correct' if obj.bbox else 'incorrect',
                'reasoning': f'The object is clearly a {obj.name}.' if obj.bbox else 'No bounding box found.'
            }
            results.append(result)
    
    table = PrettyTable()
    table.field_names = ["Object Name", "Judgement", "Reasoning"]
    for result in results:
        table.add_row([result['object_name'], result['judgement'], result['reasoning']])
    
    print(table)
    return state

class LocalizationProcessor:
    def __init__(self, client: Client):
        self.client = client

    def get_localization_results(self, state: PipelineState) -> PipelineState:
        try:
            set_model.sync(client=self.client, body=SetModelReq(
                name=config['models']['localization']['name'],
                variant=config['models']['localization']['variant'],
                task=TaskType[config['models']['localization']['task']]
            ))
        except Exception as e:
            print(f"Error setting localization model: {str(e)}")
            return state
        
        for clicking_image in state.images:
            image_file = image_to_http_file(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:
                request = BodyGetPrediction(image=image_file)
                try:
                    response = get_prediction.sync(
                        client=self.client,
                        body=request,
                        task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
                        input_text=obj.description
                    )

                    if response.prediction.bboxes:
                        obj.bbox = BoundingBox(bbox=response.prediction.bboxes[0], mode=BBoxMode.XYWH)
                    else:
                        print(f"No bounding box found for {obj.name}")
                except Exception as e:
                    print(f"Error getting prediction for image {clicking_image.id}, object {obj.name}: {str(e)}")
        
        return state

class SegmentationProcessor:
    def __init__(self, client: Client):
        self.client = client

    def get_segmentation_results(self, state: PipelineState) -> PipelineState:
        try:
            set_model.sync(client=self.client, body=SetModelReq(
                name=config['models']['segmentation']['name'],
                variant=config['models']['segmentation']['variant'],
                task=TaskType[config['models']['segmentation']['task']]
            ))
        except Exception as e:
            print(f"Error setting segmentation model: {str(e)}")
            return state
        
        for clicking_image in state.images:
            image_file = image_to_http_file(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:
                request = BodyGetPrediction(image=image_file)
                try:
                    response = get_prediction.sync(
                        client=self.client,
                        body=request,
                        task=TaskType.SEGMENTATION_WITH_BBOX,
                        input_boxes=json.dumps(obj.bbox.get(mode=BBoxMode.XYWH))
                    )
                    
                    if response.prediction.masks:
                        obj.mask = SegmentationMask(coco_rle=response.prediction.masks[0], mode=SegmentationMode.COCO_RLE)
                    else:
                        print(f"No segmentation mask found for {obj.name}")
                except Exception as e:
                    print(f"Error processing segmentation for image {clicking_image.id}, object {obj.name}: {str(e)}")
        
        return state


#%%
nest_asyncio.apply()

pipeline = Pipeline(config=config)
localization_processor = LocalizationProcessor(client)
segmentation_processor = SegmentationProcessor(client)
output_corrector = OutputCorrector(prompt_path=config['prompts']['output_corrector_path'])

pipeline.add_step("Sample Dataset", sample_dataset)
pipeline.add_step("Process Prompts", process_prompts)
pipeline.add_step("Get Localization Results", localization_processor.get_localization_results)
pipeline.add_step("Verify bboxes", verify_bboxes)
pipeline.add_step("Get Segmentation Results", segmentation_processor.get_segmentation_results)

# Print the pipeline structure
pipeline.print_pipeline()

# Perform static analysis before running the pipeline
pipeline.static_analysis()

# Run the entire pipeline
image_ids = [38, 31, 34]
results = asyncio.run(pipeline.run(image_ids))
#%%

# Run from a specific step using cached data
result = asyncio.run(pipeline.run_from_step("Get Localization Results"))

# # Or provide an initial state if needed
# initial_state = PipelineState(images=[22, 31, 34])
# initial_state.processed_prompts = pipeline.get_step_result("Process Prompts").processed_prompts
# result = asyncio.run(pipeline.run_from_step("Get Localization Results", initial_state))

#%%
# # Access cached results for logging or analysis
# localization_results = pipeline.get_step_result("Get Localization Results")
# segmentation_results = pipeline.get_step_result("Get Segmentation Results")

# Visualize results
for clicking_image in results.images:
    show_localization_predictions(clicking_image, object_names_to_show=['Sewing Machine']) 
    show_segmentation_predictions(clicking_image, object_names_to_show=['Sewing Machine'])
#%% print predicted and true objects
from clicking.common.logging import print_image_objects, print_object_descriptions, selva

# Call the function with the results
print_image_objects(results.images)

#%%
output_corrector_results =  output_corrector.verify_bboxes(results.images[0])
#%%
id = 0
show_localization_predictions(results.images[id], object_names_to_show=['Sewing Machine']) 
show_segmentation_predictions(results.images[id], object_names_to_show=['Sewing Machine'])