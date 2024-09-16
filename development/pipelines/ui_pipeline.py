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
from clicking.image_processor.ocr import OCR

import nest_asyncio
nest_asyncio.apply()
#%%
# Load the configuration file
CONFIG_PATH = "./development/pipelines/ui_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

client = Client(base_url=config['api']['local_url'], timeout=120)
#%%
prompt_refiner = PromptRefiner(config=config)
ocr_processor = OCR(client, config=config)
#%%
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])


clicking_images = coco_dataset.sample_dataset(num_samples=50)

#%%
# Define the pipeline modes
from clicking.output_corrector.core import VerificationMode  

pipeline_modes = PipelineModes({
    "prompt_mode": PromptMode,
    "localization_input_mode": LocalizerInput, 
    "localization_mode": TaskType,
    "segmentation_mode": TaskType,
    "verification_mode": VerificationMode
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
        name="Get OCR Results",
        function=ocr_processor.get_ocr_results,
        mode_keys=["ocr_mode"],
        use_cache=True
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
loaded_state = pipeline.load_state('.ui_pipeline_cache/ui_ocr/pipeline_state.pkl')

for image in loaded_state.images:
    for obj in image.ui_elements:
        obj.bbox = None
#%%
all_results = asyncio.run(pipeline.run_for_all_modes(
    #initial_images=clicking_images,
    initial_state=loaded_state,
    pipeline_modes=pipeline_mode_sequence,
    start_from_step="Filter categories",
    #stop_after_step="Process Prompts"
))

#% Print summary of results
# pipeline.print_mode_results_summary(all_results)
result =  all_results.get_run_by_mode_name("basic")
# %%
pipeline.save_state(result, name="ui_ocr")
# %%
from clicking.evaluator.core import plot_ui_element_histogram

# Call the function with the result
plot_ui_element_histogram(result.images)
#%%
from clicking.image_processor.visualization import show_ui_elements

images_copy = result.images.copy()
for image in images_copy: 
    show_ui_elements(image, bbox_thickness=5)


# %%
