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
localization_processor = Localization(client, config=config)
segmentation_processor = Segmentation(client, config=config)
output_corrector = OutputCorrector(config=config)

#segmentation_text = SegmentationText(client, config=config

#%%
# Define the pipeline modes
from clicking.output_corrector.core import VerificationMode  

pipeline_modes = PipelineModes({
    "prompt_mode": PromptMode,
    "localization_input_mode": LocalizerInput, 
    "localization_mode": TaskType,
    "segmentation_mode": TaskType,
    "bbox_verification_mode": VerificationMode
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
        mode_keys=["bbox_verification_mode"]
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
# Load initial state

# # obj_descriptions
# loaded_state = pipeline.load_state('.pipeline_cache/obj_descriptions/pipeline_state.pkl')

loaded_state_1 = pipeline.load_state('.pipeline_cache/florence2_ow_obj_name/pipeline_state.pkl')
loaded_state_2 = pipeline.load_state('.pipeline_cache/florence2_ow_obj_description/pipeline_state.pkl')
loaded_state_3 = pipeline.load_state('.pipeline_cache/florence2_ow_obj_evf_description/pipeline_state.pkl')
#%%
for clicking_image in loaded_state_3.images:
    image = clicking_image.image
    for obj in clicking_image.predicted_objects:
        obj.validity.status = ValidityStatus.UNKNOWN
        obj.mask.denoise_mask()

        extracted_area = obj.mask.extract_area(image)

        if extracted_area.width >= image.width or extracted_area.height >= image.height:
            obj.validity.status = ValidityStatus.INVALID
            print(f"Invalid mask: {obj.name}")

            print(f"Image size: {image.width,}")
            print(f"Mask size: {extracted_area.size}")
        
#%%

from clicking.evaluator.core import show_validity_statistics

states = [loaded_state_1, loaded_state_2, loaded_state_3]
labels = ["Florence2 OW Obj Name", "Florence2 OW Obj Description", "Florence2 OW Obj EVF Description"]
show_validity_statistics(states, labels)

#%%
output_corrector = OutputCorrector(config=config)
output_corrector.verify(loaded_state_3, bbox_verification_mode=VerificationMode.
CLICKPOINT)
#%%

for image in loaded_state_3.images:
    for obj in image.predicted_objects:
        print(obj.name)
        plt.imshow(obj.mask.extract_area(image.image, padding=10))
        plt.axis('off')
        plt.show()
#%%
for image in loaded_state_3.images:
    show_segmentation_predictions(image, show_descriptions=False)
#%%
from clicking.image_processor.visualization import show_localization_predictions

for image in loaded_state.images:
    show_localization_predictions(image, show_descriptions=False)
    # for obj in image.predicted_objects:
    #     print(obj.name, obj.bbox)
#%%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(loaded_state.images, show_stats=True)
#%%
from clicking.common.logging import show_object_validity
show_object_validity(loaded_state_3)

# %%
from clicking.common.logging import print_object_descriptions

# print_object_descriptions(result.images)
for image in loaded_state.images:
    for obj in image.predicted_objects:
        print(obj.name, obj.mask)
# %%
from clicking.common.logging import print_object_descriptions

print_object_descriptions(loaded_state.images)
for image in loaded_state.images:
    for obj in image.predicted_objects:
        print(obj.name, obj.description)

        # obj.mask.denoise_mask()
#%%
for clicking_image in loaded_state.images:
    show_segmentation_predictions(clicking_image, show_descriptions=False)

#%%
pipeline.save_state(result, name="florence2_ow_obj_evf_description")

# %%
from clicking.common.logging import show_object_validity
show_object_validity(result)

# %%
#localization_processor = Localization(client, config=config)
localization_processor.get_localization_results(loaded_state, localization_mode=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB, localization_input_mode=LocalizerInput.OBJ_NAME)
# %%
for image in loaded_state.images:
    show_localization_predictions(image, show_descriptions=False)
    for obj in image.predicted_objects:
        print(f"Obj {obj.name} bbox: {obj.bbox}")
# %%

#segmentation_processor = Segmentation(client, config=config)

segmentation_processor.get_segmentation_results(result, segmentation_mode=TaskType.SEGMENTATION_WITH_CLICKPOINT)
for image in loaded_state.images:
    show_segmentation_predictions(image, show_descriptions=False)
    for obj in image.predicted_:
        print(f"Obj {obj.name} mask: {obj.mask}")
# %%
from clicking.image_processor.ocr import OCR
