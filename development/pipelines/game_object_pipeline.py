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
# Load the configuration file«
CONFIG_PATH = "./development/pipelines/game_object_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

client = Client(base_url=config['api']['cloud_url'], timeout=120)
#%%
prompt_refiner = PromptRefiner(config=config)
#localization_processor = Localization(client, config=config)
segmentation_processor = Segmentation(client, config=config)
# output_corrector = OutputCorrector(config=config)
# segmentation_text = SegmentationText(client, config=config)
#%%
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])

image_ids = [26,24,18]
clicking_images = coco_dataset.sample_dataset()
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
        name="Get Localization Results",
        function=localization_processor.get_localization_results,
        mode_keys=["localization_mode", "localization_input_mode"]
    ),
    PipelineStep(
        name="Verify bboxes",
        function=output_corrector.verify,
        mode_keys=["verification_mode"]
    ),
    PipelineStep(
        name="Get Segmentation Results",
        function=segmentation_processor.get_segmentation_results,
        mode_keys=["segmentation_mode"]
    )
]

# # Create pipeline steps
# pipeline_steps = [
#     PipelineStep(
#         name="Process Prompts",
#         function=prompt_refiner.process_prompts,
#         mode_keys=["prompt_mode"],
#         use_cache=True
#     ),
#     PipelineStep(
#         name="Filter categories",
#         function=lambda state: state.filter_by_object_category(ObjectCategory.GAME_ASSET),
#         mode_keys=[],
#     ),
#     PipelineStep(
#         name="Get Segmentation Results",
#         function=segmentation_text.get_segmentation_results,
#         mode_keys=["segmentation_mode"]
#     ),
# ]

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
loaded_state = pipeline.load_state('.pipeline_cache/florence2_ow_obj_name/pipeline_state.pkl')
# loaded_state = loaded_state.filter_by_ids(image_ids)

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
    stop_after_step="Get Localization Results"
))

# Print summary of results
pipeline.print_mode_results_summary(all_results)

result =  all_results.get_run_by_mode_name("open_vocab_object_name") 
#%%
for image in result.images:
    show_segmentation_predictions(image, show_descriptions=False)
#%%
from clicking.image_processor.visualization import show_localization_predictions

for image in result.images:
    show_localization_predictions(image, show_descriptions=False)
    # for obj in image.predicted_objects:
    #     print(obj.name, obj.bbox)
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

# %%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(result.images)
for image in result.images:
    for obj in image.predicted_objects:
        print(obj.name, obj.mask)
# %%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(result.images)
for image in result.images:
    for obj in image.predicted_objects:
        print(obj.name, obj.description)

#%%
from clicking.common.mask import SegmentationMode

for clicking_image in loaded_state.images:
    image = clicking_image.image
    for obj in clicking_image.predicted_:
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

#%%
pipeline.save_state(result, name="obj_descriptions")

# %%
from clicking.common.logging import show_object_validity
show_object_validity(result)

# %%
localization_processor = Localization(client, config=config)
#%%
new_state =  localization_processor.get_localization_results(loaded_state, localization_mode=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB, localization_input_mode=LocalizerInput.OBJ_NAME)
# %%
for image in new_state.images:
    show_localization_predictions(image, show_descriptions=False)
    for obj in image.predicted_objects:
        print(f"Obj {obj.name} bbox: {obj.bbox}")
# %%

#segmentation_processor = Segmentation(client, config=config)

segmentation_processor.get_segmentation_results(loaded_state, segmentation_mode=TaskType.SEGMENTATION_WITH_CLICKPOINT)
