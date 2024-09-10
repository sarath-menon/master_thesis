#%%

%load_ext autoreload
%autoreload 2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from clicking.pipeline.core import Pipeline, PipelineState, PipelineStep, PipelineMode, PipelineModeSequence, PipelineModes
from clicking.dataset_creator.core import CocoDataset
from clicking.prompt_refinement.core import PromptRefiner, PromptMode
from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector, BBoxVerificationMode
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
from clicking.image_processor.visualization import show_localization_predictions, show_segmentation_predictions
from io import BytesIO
import json
import nest_asyncio
from clicking.image_processor.localization import Localization, LocalizerInput
from clicking.image_processor.segmentation import Segmentation

#%%
# Load the configuration file
CONFIG_PATH = "./development/pipelines/config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

client = Client(base_url=config['api']['local_url'], timeout=50)
#%%

prompt_refiner = PromptRefiner(config=config)
localization_processor = Localization(client, config=config)
segmentation_processor = Segmentation(client, config=config)
output_corrector = OutputCorrector(config=config)

#%%
from clicking.common.logging import print_object_descriptions
nest_asyncio.apply()

# sample images
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

#%%
# Define the pipeline modes
from clicking.output_corrector.core import BBoxVerificationMode  

pipeline_modes = PipelineModes({
    "prompt_mode": PromptMode,
    "localization_input_mode": LocalizerInput, 
    "localization_mode": TaskType,
    "segmentation_mode": TaskType,
    "bbox_verification_mode": BBoxVerificationMode
})

# Create pipeline steps
pipeline_steps = [
    PipelineStep(
        name="Process Prompts",
        function=prompt_refiner.process_prompts,
        mode_keys=["prompt_mode"],
        use_cache=True
    ),
    PipelineStep(
        name="Filter categories",
        function=lambda state: state.filter_by_object_category(ObjectCategory.GAME_ASSET),
        mode_keys=[],
    ),
    PipelineStep(
        name="Get Localization Results",
        function=localization_processor.get_localization_results,
        mode_keys=["localization_mode", "localization_input_mode"]
    ),
    PipelineStep(
        name="Verify bboxes",
        function=output_corrector.verify_bboxes,
        mode_keys=["bbox_verification_mode"]
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
pipeline_mode_sequence = PipelineModeSequence.from_config(config, pipeline_modes)
pipeline_mode_sequence.print_mode_sequences()

#%%
image_ids = [9,13,23]
# clicking_images = coco_dataset.sample_dataset()

# Load initial state
loaded_state = pipeline.load_state()
loaded_state = loaded_state.filter_by_ids(sample_size=3)
loaded_state = loaded_state.filter_by_object_category(ObjectCategory.GAME_ASSET)

def remove_full_stops(description: str) -> str:
    if description.endswith('.'):
        return description[:-1]
    return description

# for image in loaded_state.images:
#     for obj in image.predicted_objects:
#         obj.description = remove_full_stops(obj.description)

#%%

all_results = asyncio.run(pipeline.run_for_all_modes(
    #initial_images=clicking_images,
    initial_state=loaded_state,
    pipeline_modes=pipeline_mode_sequence,
    start_from_step="Filter categories",
    # stop_after_step="Filter categories"
))

# Print summary of results
pipeline.print_mode_results_summary(all_results)

result =  all_results.get_run_by_mode_name("object_detection_open_vocab")
#result =  all_results.get_run_by_mode_name("object_detection_text_grounded")
#%%
for image in result.images:
    show_segmentation_predictions(image, show_descriptions=False)
#%%
from clicking.image_processor.visualization import show_localization_predictions

for image in result.images:
    show_localization_predictions(image, show_descriptions=False)
    for obj in image.predicted_objects:
        print(obj.name, obj.bbox)
# %% Generate and save the config schema
# config_schema = PipelineModeSequence.generate_config_schema(pipeline_modes)
# with open('config_schema.json', 'w') as f:
#     json.dump(config_schema, f, indent=2)

# print("Config schema generated and saved to 'config_schema.json'")
# import copy
#%%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(result.images)
#%%
from clicking.common.logging import show_object_validity
show_object_validity(result)

# pipeline.save_state(result)
# %%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(result.images)
for image in result.images:
    for obj in image.predicted_objects:
        print(obj.name, obj.description)
# %%
from clicking.image_processor.segmentation_text import SegmentationText

evf_sam2 = SegmentationText(client, config=config)

#%%
evf_sam2.get_segmentation_results(loaded_state, segmentation_mode=TaskType.SEGMENTATION_WITH_TEXT)
#%%
from clicking.common.mask import SegmentationMode

for clicking_image in loaded_state.images:
    image = clicking_image.image
    for obj in clicking_image.predicted_objects:
        obj.validity.status = ValidityStatus.UNKNOWN

        extracted_area = obj.mask.extract_area(image)

        if extracted_area.width >  image.width/2 or extracted_area.height > image.height/2:
            obj.validity.status = ValidityStatus.INVALID
            print(f"Invalid mask: {obj.name}")

            print(f"Image size: {image.width,}")
            print(f"Mask size: {extracted_area.size}")
        # obj.mask.denoise_mask()


#%%
for clicking_image in loaded_state.images:
    show_segmentation_predictions(clicking_image, show_descriptions=False)
# %%
