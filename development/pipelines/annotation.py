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
from clicking.output_corrector.core import OutputCorrector, VerificationMode
from clicking_client import Client
from clicking.common.data_structures import *
import asyncio
import yaml
from clicking.image_processor.visualization import show_localization_predictions, show_segmentation_predictions
from io import BytesIO
from clicking.image_processor.localization import Localization, LocalizerInput
from clicking.image_processor.segmentation import Segmentation
from clicking.image_processor.segmentation_text import SegmentationText

import nest_asyncio
nest_asyncio.apply()
#%%
# Load the configuration file
CONFIG_PATH = "./development/pipelines/annotation_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

prompt_refiner = PromptRefiner(config=config)
#%%
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'], use_gcp_urls=True)

clicking_images = coco_dataset.sample_dataset()
#%%
# Define the pipeline modes
from clicking.output_corrector.core import VerificationMode  

pipeline_modes = PipelineModes({
    "prompt_mode": PromptMode,
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
all_results = asyncio.run(pipeline.run_for_all_modes(
    initial_images=clicking_images,
    # initial_state=loaded_state,
    pipeline_modes=pipeline_mode_sequence,
    # start_from_step="Filter categories",
    # stop_after_step="Get Localization Results"
))

# Print summary of results
pipeline.print_mode_results_summary(all_results)

result =  all_results.get_run_by_mode_name("description_generation") 
#%%
pipeline.save_state(result, name="obj_descriptions")
#%%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(result.images, show_image=True)
#%%
from clicking.evaluator.core import save_descriptions

save_descriptions(result, './datasets/annotation_dataset/coco_dataset_gcp/descriptions')
#%%
import sys
import json
import os

input_json = './datasets/annotation_dataset/coco_dataset_gcp/descriptions/descriptions_results.json'
output_folder = './output_tasks'  # Set output folder manually

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(input_json) as inp:
    tasks = json.load(inp)

for i, v in enumerate(tasks):
    with open(os.path.join(output_folder, f'task_{i}.json'), 'w') as f:
        json.dump(v, f)

#%%
from clicking.evaluator.core import save_descriptions_labelbox

file_name = 'descriptions_labelbox.json'
output_folder = './output_tasks'  # Set output folder manually
save_descriptions_labelbox(result, output_folder, file_name)
#%%
