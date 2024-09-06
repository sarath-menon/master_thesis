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
from clicking.image_processor.localization import Localization
from clicking.image_processor.segmentation import Segmentation

#%%
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
    
    return state

def filter_by_object_category(state: PipelineState, category: ObjectCategory) -> PipelineState:
    for clicking_image in state.images:
        clicking_image.predicted_objects = [
            obj for obj in clicking_image.predicted_objects
            if obj.category == category
        ]
    return state
import random

def filter_images(state: PipelineState, image_ids: List[int] = None, sample_size: int = None) -> PipelineState:
    if image_ids is not None and sample_size is not None:
        raise ValueError("Cannot specify both image_ids and sample_size. Choose one filtering method.")
    
    # convert image_ids to list of strings
    if image_ids is not None:
        image_ids = [str(id) for id in image_ids]
    
    if image_ids is not None:
        filtered_images = [
            img for img in state.images
            if img.id in image_ids
        ]
    elif sample_size is not None:
        if sample_size > len(state.images):
            raise ValueError(f"Sample size {sample_size} is larger than the number of available images {len(state.images)}")
        filtered_images = random.sample(state.images, sample_size)
    else:
        return state  # Return original state if no filtering is specified
    
    return PipelineState(images=filtered_images)

#%%
# Load the configuration file
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

client = Client(base_url=config['api']['local_url'], timeout=50)

prompt_refiner = PromptRefiner(prompt_path=config['prompts']['refinement_path'], config=config)
localization_processor = Localization(client, config=config)
segmentation_processor = Segmentation(client, config=config)
output_corrector = OutputCorrector(prompt_path=config['prompts']['output_corrector_path'])

#%%
from clicking.common.logging import print_object_descriptions
nest_asyncio.apply()

# sample images
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

image_ids = [22, 31, 42]
clicking_images = coco_dataset.sample_dataset()


#%%
pipeline = Pipeline(config=config)

pipeline.add_step("Process Prompts", process_prompts)
pipeline.add_step("Get Localization Results", localization_processor.get_localization_results)
pipeline.add_step("Verify bboxes", verify_bboxes)
pipeline.add_step("Get Segmentation Results", segmentation_processor.get_segmentation_results)

pipeline.print_pipeline()
pipeline.static_analysis()
#%% Run the entire pipeline, stopping after "Verify bboxes" step
from clicking.common.logging import print_object_descriptions

loaded_state = pipeline.load_state()
print("loaded_state", len(loaded_state.images))
# loaded_state = filter_images(loaded_state, image_ids=[0,12,37])
loaded_state = filter_images(loaded_state, sample_size=3)
print("loaded_state", len(loaded_state.images))
loaded_state = filter_by_object_category(loaded_state, ObjectCategory.GAME_ASSET)

print_object_descriptions(loaded_state.images, show_image=True, show_stats=False)
#%%

results = asyncio.run(pipeline.run( 
    # initial_images=clicking_images, 
    initial_state=loaded_state,
    start_from_step="Get Localization Results",
    stop_after_step="Get Localization Results",
))

#%% Visualize results
for clicking_image in results.images:
    show_localization_predictions(clicking_image) 
    # show_segmentation_predictions(clicking_image)

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
loaded_state = pipeline.load_state()
#%%

 # Example usage:
pipeline.save_state(results, save_as_json=True, log_to_wandb=False)
#%%
pipeline.save_state_as_json(loaded_state)

#%%

selv = pipeline.create_state_json(loaded_state)
import pandas as pd

#%%
