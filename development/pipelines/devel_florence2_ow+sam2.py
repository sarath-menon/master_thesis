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
from clicking.vision_model.data_structures import TaskType
from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction
from clicking.common.data_structures import *
import pickle
import os
from datetime import datetime
import asyncio
from typing import Callable, Union
import inspect
from dataclasses import dataclass, field
import yaml
from clicking.vision_model.visualization import show_localization_predictions, show_segmentation_predictions
from io import BytesIO
import json
import nest_asyncio
from clicking.pipeline.core import PipelineState
from fastapi import File, UploadFile


#%%
from clicking_client.types import File

def image_to_http_file(image):
    # Convert PIL Image to bytes and create a File object
    image_byte_arr = BytesIO()
    image.save(image_byte_arr, format='PNG')
    image_file = File(file_name="image.png", payload=image_byte_arr.getvalue(), mime_type="image/png")
    return image_file

def process_prompts(state: PipelineState) -> PipelineState:
    state.images = prompt_refiner.process_prompts(state.images)
    return state

from prettytable import PrettyTable

def verify_bboxes(state: PipelineState) -> PipelineState:
    results = []
    for clicking_image in state.images:
        clicking_image = output_corrector.verify_bboxes(clicking_image)
        for obj in clicking_image.predicted_objects:
            result = {
                'object_name': obj.name,
                'judgement': obj.validity.is_valid,
                'reasoning': obj.validity.reason
            }
            results.append(result)
    
    table = PrettyTable()
    table.field_names = ["Object Name", "Judgement", "Reasoning"]
    for result in results:
        table.add_row([result['object_name'], result['judgement'], result['reasoning']])
    
    print(table)
    return state

class LocalizationProcessor:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.set_localization_model()

    def set_localization_model(self):
        try:
            set_model.sync(client=self.client, body=SetModelReq(
                name=self.config['models']['localization']['name'],
                variant=self.config['models']['localization']['variant'],
                task=TaskType[self.config['models']['localization']['task']]
            ))
        except Exception as e:
            print(f"Error setting localization model: {str(e)}")

    def get_localization_results(self, state: PipelineState) -> PipelineState:
        
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

                    if len(response.prediction.bboxes) > 1:
                        print(f"Multiple bounding boxes found for {obj.name}")
                        obj.bbox = BoundingBox(bbox=response.prediction.bboxes[0], mode=BBoxMode.XYWH)
                    elif len(response.prediction.bboxes) == 1:
                        obj.bbox = BoundingBox(bbox=response.prediction.bboxes[0], mode=BBoxMode.XYWH)
                    else:
                        print(f"No bounding box found for {obj.name}")

                except Exception as e:
                    print(f"Error getting prediction for image {clicking_image.id}, object {obj.name}: {str(e)}")
        
        return state

class SegmentationProcessor:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.set_segmentation_model()

    def set_segmentation_model(self):
        try:
            set_model.sync(client=self.client, body=SetModelReq(
                name=self.config['models']['segmentation']['name'],
                variant=self.config['models']['segmentation']['variant'],
                task=TaskType[self.config['models']['segmentation']['task']]
            ))
        except Exception as e:
            print(f"Error setting segmentation model: {str(e)}")

    def get_segmentation_results(self, state: PipelineState) -> PipelineState:
        
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
# Load the configuration file
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

client = Client(base_url=config['api']['cloud_url'], timeout=50)
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

prompt_refiner = PromptRefiner(prompt_path=config['prompts']['refinement_path'], config=config)
localization_processor = LocalizationProcessor(client, config=config)
segmentation_processor = SegmentationProcessor(client, config=config)
output_corrector = OutputCorrector(prompt_path=config['prompts']['output_corrector_path'])


#%%
from clicking.common.logging import print_object_descriptions
nest_asyncio.apply()

# sample images
image_ids = [22, 31, 42]
clicking_images = coco_dataset.sample_dataset(image_ids)

pipeline = Pipeline(config=config)

pipeline.add_step("Process Prompts", process_prompts)
pipeline.add_step("Get Localization Results", localization_processor.get_localization_results)
pipeline.add_step("Verify bboxes", verify_bboxes)
pipeline.add_step("Get Segmentation Results", segmentation_processor.get_segmentation_results)

# Print the pipeline structure
pipeline.print_pipeline()

# Perform static analysis before running the pipeline
pipeline.static_analysis()

#%% Run the entire pipeline, stopping after "Verify bboxes" step
# image_ids = [i for i in range(coco_dataset.length())]
image_ids = [22, 31, 42]

results = asyncio.run(pipeline.run( 
    # initial_images=image_ids, 
    initial_state=loaded_state1,
    start_from_step="Get Localization Results",
    stop_after_step="Get Localization Results",
))

# Print and visualize results
# print_object_descriptions(results.images)

#%%

# for result in results.images:
#     print(result.predicted_objects[0].validity)

for clicking_image in results.images:
    show_localization_predictions(clicking_image)

#%%

# # replace pipeline step
# pipeline.replace_step("Verify bboxes", verify_bboxes)
    
# # Or provide an initial state if needed
# initial_state = PipelineState(images=[22, 31, 34])
# initial_state.processed_prompts = pipeline.get_step_result("Process Prompts").processed_prompts

# result = asyncio.run(pipeline.run_from_step("Get Localization Results", initial_state))

#%%

# Visualize results
for clicking_image in results.images:
    show_localization_predictions(clicking_image) 
    # show_segmentation_predictions(clicking_image)
#%% print predicted and true objects
from clicking.common.logging import print_image_objects, print_object_descriptions

# Call the function with the results
# print_image_objects(results.images)
print_object_descriptions(results.images, show_image=True, show_stats=True)

#%%
output_corrector_results =  output_corrector.verify_bboxes(results.images[0])

#%%
id = 0
show_localization_predictions(results.images[id], object_names_to_show=['Sewing Machine']) 
#%%
from clicking.evaluator.core import save_validity_results

# Example usage:
EVALS_PATH = "./datasets/evals/output_corrector"
save_validity_results(results, EVALS_PATH)
#%%
from clicking.evaluator.core import evaluate_validity_results

# Example usage:
EVALS_PATH = "./evals/output_corrector"
ground_truth_file = f'{EVALS_PATH}/ground_truth.json'
predictions_file = f'{EVALS_PATH}/validity_results.json'

evaluation_results = evaluate_validity_results(ground_truth_file, predictions_file)

#%%
from clicking.evaluator.core import save_image_descriptions
 
EVALS_PATH = "./datasets/evals/output_corrector"
save_image_descriptions(results.images, EVALS_PATH, prompt_path=config['prompts']['refinement_path'])

# %%
from clicking.evaluator.core import load_pipeline_results

results = load_pipeline_results("./datasets/evals/output_corrector/image_descriptions.json", coco_dataset)

#%%
import pickle

# Example usage:
loaded_state1 = pipeline.load_state()
#%%

 # Example usage:
pipeline.save_state(loaded_state, save_as_json=True, log_to_wandb=False)
#%%
pipeline.save_state_as_json(loaded_state)

#%%

selv = pipeline.create_state_json(loaded_state)
import pandas as pd

#%%
