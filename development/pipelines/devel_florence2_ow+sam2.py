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
from clicking.vision_model.data_structures import TaskType
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
from clicking.vision_model.visualization import show_localization_predictions, show_segmentation_predictions
from io import BytesIO
import json
import nest_asyncio
from clicking.image_processor.localization import Localization, LocalizerInput
from clicking.image_processor.segmentation import Segmentation

#%%

# Load the configuration file
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

#%%
client = Client(base_url=config['api']['local_url'], timeout=50)

prompt_refiner = PromptRefiner(config=config)
localization_processor = Localization(client, config=config)
segmentation_processor = Segmentation(client, config=config)
output_corrector = OutputCorrector(config=config)

#%%
from clicking.common.logging import print_object_descriptions
nest_asyncio.apply()

# sample images
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

image_ids = [22, 31, 42]
clicking_images = coco_dataset.sample_dataset()
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
# Load initial state
loaded_state = pipeline.load_state()
loaded_state = loaded_state.filter_by_ids(image_ids=image_ids)
# loaded_state = loaded_state.filter_by_object_category(ObjectCategory.GAME_ASSET)

#%%

all_results = asyncio.run(pipeline.run_for_all_modes(
    #initial_images=clicking_images,
    initial_state=loaded_state,
    pipeline_modes=pipeline_mode_sequence,
    start_from_step="Filter categories",
    stop_after_step="Filter categories"
))

# Print summary of results
pipeline.print_mode_results_summary(all_results)

result =  all_results.get_run_by_mode_name("object_detection_open_vocab")
#result =  all_results.get_run_by_mode_name("object_detection_text_grounded")
#
#%%
from clicking.vision_model.visualization import show_localization_predictions

for image in result.images:
    show_localization_predictions(image, show_descriptions=False)
    for obj in image.predicted_objects:
        print(obj.name, obj.bbox)

#%%
for image in result.images:
    show_segmentation_predictions(image, show_descriptions=False)
# %% Generate and save the config schema
# config_schema = PipelineModeSequence.generate_config_schema(pipeline_modes)
# with open('config_schema.json', 'w') as f:
#     json.dump(config_schema, f, indent=2)

# print("Config schema generated and saved to 'config_schema.json'")
# import copy


#%%
# output_corrector = OutputCorrector(config=config)
# output_corrector.verify_bboxes(result, bbox_verification_mode=BBoxVerificationMode.CROP, show_images=True)

# from clicking.common.logging import show_object_validity
# show_object_validity(result)
# #%%
# segmentation_processor = Segmentation(client, config=config)
# segmentation_processor.get_segmentation_results(result)

pipeline.save_state(result)
# %%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(result.images)