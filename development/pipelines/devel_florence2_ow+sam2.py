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
from clicking_client.data_structures import File
from io import BytesIO
import json
import nest_asyncio
from clicking.pipeline.core import PipelineState

#%%
def image_to_http_file(image):
    # Convert PIL Image to bytes and create a File object
    image_byte_arr = BytesIO()
    image.save(image_byte_arr, format='PNG')
    image_file = File(file_name="image.png", payload=image_byte_arr.getvalue(), mime_type="image/png")
    return image_file

# Modify the relevant steps to use PipelineState

def sample_dataset(state: PipelineState) -> PipelineState:
    state.images = coco_dataset.sample_dataset(state.images)
    return state

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

client = Client(base_url=config['api']['local_url'])
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

prompt_refiner = PromptRefiner(prompt_path=config['prompts']['refinement_path'], config=config)
localization_processor = LocalizationProcessor(client, config=config)
segmentation_processor = SegmentationProcessor(client, config=config)
output_corrector = OutputCorrector(prompt_path=config['prompts']['output_corrector_path'])

#%%
nest_asyncio.apply()

pipeline = Pipeline(config=config)

pipeline.add_step("Sample Dataset", sample_dataset)
pipeline.add_step("Process Prompts", process_prompts)
pipeline.add_step("Get Localization Results", localization_processor.get_localization_results)
pipeline.add_step("Verify bboxes", verify_bboxes)
# pipeline.add_step("Get Segmentation Results", segmentation_processor.get_segmentation_results)

# Print the pipeline structure
pipeline.print_pipeline()

# Perform static analysis before running the pipeline
pipeline.static_analysis()

# Run the entire pipeline
image_ids = [38, 31]
results = asyncio.run(pipeline.run(image_ids))

# for obj in results.images[0].predicted_objects:
#     print(obj.validity)


# Visualize results
for clicking_image in results.images:
    show_localization_predictions(clicking_image) 
    # show_segmentation_predictions(clicking_image)
 
#%%

# # replace pipeline step
# pipeline.replace_step("Verify bboxes", verify_bboxes)

# Run from a specific step using cached data
result = asyncio.run(pipeline.run_from_step("Verify bboxes"))
    
# # Or provide an initial state if needed
# initial_state = PipelineState(images=[22, 31, 34])
# initial_state.processed_prompts = pipeline.get_step_result("Process Prompts").processed_prompts

# result = asyncio.run(pipeline.run_from_step("Get Localization Results", initial_state))

#%%
# # Access cached results for logging or analysis
# localization_results = pipeline.get_step_result("Get Localization Results")
# segmentation_results = pipeline.get_step_result("Get Segmentation Results")

result = asyncio.run(pipeline.run_from_step("Get Localization Results"))

# Visualize results
for clicking_image in results.images:
    show_localization_predictions(clicking_image) 
    # show_segmentation_predictions(clicking_image)
#%% print predicted and true objects
from clicking.common.logging import print_image_objects, print_object_descriptions, selva

# Call the function with the results
print_image_objects(results.images)

#%%
output_corrector_results =  output_corrector.verify_bboxes(results.images[0])

#%%
id = 0
show_localization_predictions(results.images[id], object_names_to_show=['Sewing Machine']) 
#%%

def save_validity_results(results: PipelineState, output_file: str):
    validity_data = []
    
    for clicking_image in results.images:
        for obj in clicking_image.predicted_objects:
            validity_data.append({
                'image_id': clicking_image.id,
                # 'object_id': obj.id,
                'object_name': obj.name,
                'is_valid': obj.validity.is_valid,
                'reason': obj.validity.reason
            })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(validity_data, f, indent=2)
    
    print(f"Validity results saved to {output_file}")

# Example usage:
EVALS_PATH = "./evals/output_corrector"
save_validity_results(results, f'{EVALS_PATH}/validity_results.json')

