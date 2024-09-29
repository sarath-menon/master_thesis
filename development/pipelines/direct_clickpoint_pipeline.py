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
from clicking.image_processor.pointing import Pointing, PointingInput
from clicking.image_processor.segmentation import Segmentation
from clicking.image_processor.segmentation_text import SegmentationText

import nest_asyncio
nest_asyncio.apply()
#%%
# Load the configuration file«
CONFIG_PATH = "./development/pipelines/game_object_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

client = Client(base_url=config['api']['local_url'], timeout=120)
#%%
prompt_refiner = PromptRefiner(config=config)
#localization_processor = Localization(client, config=config)
#segmentation_processor = Segmentation(client, config=config)
# output_corrector = OutputCorrector(config=config)
# segmentation_text = SegmentationText(client, config=config)
pointing_processor = Pointing(client, config=config)
#%%
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

clicking_images = coco_dataset.sample_dataset(num_samples=3)
#%%
# Define the pipeline modes
from clicking.output_corrector.core import VerificationMode  

pipeline_modes = PipelineModes({
    "prompt_mode": PromptMode,
    "pointing_input_mode": PointingInput, 
    "pointing_mode": TaskType
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
        name="Get Clickpoints",
        function=pointing_processor.get_pointing_results,
        mode_keys=["pointing_mode", "pointing_input_mode"]
    ),
]

# Create pipeline and add steps
pipeline = Pipeline(config=config)
for step in pipeline_steps:
    pipeline.add_step(step)

# Create pipeline modes
pipeline_mode_sequence = PipelineModeSequence.from_config(config, pipeline_modes)
pipeline_mode_sequence.print_mode_sequences()

#%%
image_ids = [26,24,18]
# clicking_images = coco_dataset.sample_dataset()

# Load initial state
loaded_state = pipeline.load_state('.pipeline_cache/obj_descriptions/pipeline_state.pkl')
loaded_state = loaded_state.filter_by_ids(image_ids)

def remove_full_stops(description: str) -> str:
    if description.endswith('.'):
        return description[:-1]
    return description

for image in loaded_state.images:
    for obj in image.predicted_objects:
        obj.description = remove_full_stops(obj.description)
        obj.bbox = None
        obj.mask = None
        obj.validity.status = ValidityStatus.UNKNOWN
#%%
all_results = asyncio.run(pipeline.run_for_all_modes(
    #initial_images=clicking_images,
    initial_state=loaded_state,
    pipeline_modes=pipeline_mode_sequence,
    start_from_step="Filter categories",
    stop_after_step="Get Clickpoints"
))

# Print summary of results
pipeline.print_mode_results_summary(all_results)

result =  all_results.get_run_by_mode_name("direct_clickpoint") 
#%%
for clicking_image in loaded_state.images:
    plt.imshow(clicking_image.image)
    plt.show()
    for obj in clicking_image.predicted_objects:
        print(obj.description)
    # show_segmentation_predictions(image, show_descriptions=False)
#%%
pointing_processor.get_pointing_results(loaded_state, TaskType.CLICKPOINT_WITH_TEXT, PointingInput.OBJ_NAME)
#%%
