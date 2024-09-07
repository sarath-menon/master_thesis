#%%

%load_ext autoreload
%autoreload 2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from clicking.pipeline.core import Pipeline, PipelineState, PipelineStep, PipelineMode, PipelineModeSequence
from clicking.dataset_creator.core import CocoDataset
from clicking.prompt_refinement.core import PromptRefiner, PromptMode
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
from clicking.image_processor.localization import Localization, InputMode
from clicking.image_processor.segmentation import Segmentation

#%%

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
# Define the modes dictionary
modes_dict = {
    "prompt_mode": PromptMode,
    "localization_input_mode": InputMode,
    "localization_mode": TaskType,
    "segmentation_mode": TaskType
}

# Create pipeline steps
pipeline_steps = [
    PipelineStep(
        name="Process Prompts",
        function=prompt_refiner.process_prompts,
        mode_keys=["prompt_mode"]
    ),
    PipelineStep(
        name="Get Localization Results",
        function=localization_processor.get_localization_results,
        mode_keys=["localization_mode", "localization_input_mode"]
    ),
    PipelineStep(
        name="Verify bboxes",
        function=verify_bboxes,
        mode_keys=[]
    ),
    PipelineStep(
        name="Get Segmentation Results",
        function=segmentation_processor.get_segmentation_results,
        mode_keys=["segmentation_mode"]
    )
]

# Create pipeline and add steps
pipeline = Pipeline(config=config)
for step in pipeline_steps:
    pipeline.add_step(step)

# Create pipeline modes
pipeline_modes = PipelineModeSequence.from_config(config, modes_dict)
pipeline_modes.print_mode_sequences()

# Load initial state
loaded_state = pipeline.load_state()
loaded_state = loaded_state.filter_by_id(image_ids=[0, 12, 37])
loaded_state = loaded_state.filter_by_object_category(ObjectCategory.GAME_ASSET)

# Run pipeline for all modes
all_results = asyncio.run(pipeline.run_for_all_modes(
    initial_state=loaded_state,
    pipeline_modes=pipeline_modes,
    start_from_step="Get Localization Results",
    stop_after_step="Get Localization Results"
))

# Print summary of results
pipeline.print_mode_results_summary(all_results)

# Generate and save the config schema
config_schema = PipelineModeSequence.generate_config_schema(modes_dict)
with open('config_schema.json', 'w') as f:
    json.dump(config_schema, f, indent=2)

print("Config schema generated and saved to 'config_schema.json'")
    
# %%
