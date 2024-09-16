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
CONFIG_PATH = "./development/pipelines/game_object_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

client = Client(base_url=config['api']['local_url'], timeout=120)
#%%

prompt_refiner = PromptRefiner(config=config)
# localization_processor = Localization(client, config=config)
# segmentation_processor = Segmentation(client, config=config)
output_corrector = OutputCorrector(config=config)

#segmentation_text = SegmentationText(client, config=config

#%%
# Create pipeline and add steps
pipeline = Pipeline(config=config)

loaded_state_1 = pipeline.load_state('.pipeline_cache/florence2_ow_obj_name/pipeline_state.pkl')
loaded_state_2 = pipeline.load_state('.pipeline_cache/florence2_ow_obj_description/pipeline_state.pkl')
loaded_state_3 = pipeline.load_state('.pipeline_cache/evf_obj_description/pipeline_state.pkl')

from clicking.evaluator.core import show_validity_statistics

states = [loaded_state_1, loaded_state_2, loaded_state_3]
state_labels = ["Florence2 Sam2 Obj Name", "Florence2 Sam2 Obj Description", "EVF Obj Description"]
show_validity_statistics(states, state_labels)

#%%
current_run = loaded_state_2

#%%
from clicking.image_processor.visualization import show_invalid_objects
show_invalid_objects(current_run, mode=VerificationMode.CROP_BBOX)

#%%
from clicking.image_processor.visualization import compare_invalid_objects

# Example usage:
invalid_object_stats, invalid_objects = compare_invalid_objects(states, state_labels, visualize=True, show_details=True)

#%%

from clicking.image_processor.visualization import show_invalid_object_images

# Usage example:
show_invalid_object_images(states, state_labels, invalid_object_stats)

#%%
for image in current_run.images:
    show_segmentation_predictions(image, show_descriptions=False)
#%%
from clicking.image_processor.visualization import show_localization_predictions

for image in current_run.images:
    show_localization_predictions(image, show_descriptions=False)
    # for obj in image.predicted_objects:
    #     print(obj.name, obj.bbox)
#%%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(current_run.images, show_stats=True)
#%%
from clicking.common.logging import show_object_validity
show_object_validity(current_run)

# %%
from clicking.common.logging import print_object_descriptions

# print_object_descriptions(result.images)
for image in current_run.images:
    for obj in image.predicted_objects:
        print(obj.name, obj.mask)
# %%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(current_run.images)
for image in current_run.images:
    for obj in image.predicted_objects:
        print(obj.name, obj.description)

        # obj.mask.denoise_mask()
#%%
for clicking_image in current_run.images:
    show_segmentation_predictions(clicking_image, show_descriptions=False)

#%%
pipeline.save_state(current_run, name="evf_obj_description")
#%%

output_corrector = OutputCorrector(config=config)
output_corrector.verify(current_run, verification_mode=VerificationMode.CROP_MASK)