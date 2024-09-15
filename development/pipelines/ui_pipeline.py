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
from clicking.output_corrector.core import OutputCorrector, BBoxVerificationMode
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

client = Client(base_url=config['api']['local_url'], timeout=50)
#%%

prompt_refiner = PromptRefiner(config=config)
ocr_processor = OCR(client, config=config)
#%%
coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])
clicking_images = coco_dataset.sample_dataset(num_samples=3)

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
image_ids = [26,24,18]
# clicking_images = coco_dataset.sample_dataset()

# Load initial state
loaded_state = pipeline.load_state()
loaded_state = loaded_state.filter_by_ids(image_ids)
# loaded_state = loaded_state.filter_by_object_category(ObjectCategory.GAME_ASSET)

for image in loaded_state.images:
    plt.imshow(image.image)
    plt.axis('off')
    plt.show()

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
    initial_images=clicking_images,
    # initial_state=loaded_state,
    pipeline_modes=pipeline_mode_sequence,
    # start_from_step="Filter categories",
    # stop_after_step="Get Localization Results"
))

# # Print summary of results
# pipeline.print_mode_results_summary(all_results)

result =  all_results.get_run_by_mode_name("object_detection_open_vocab")
#result =  all_results.get_run_by_mode_name("object_detection_text_grounded")
#%%
from clicking.image_processor.visualization import show_ocr_boxes

for image, ocr_result in zip(clicking_images, result):
    print(ocr_result.prediction.labels)
    show_ocr_boxes(image, ocr_result.prediction)

#%%
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)
prompt_refiner = PromptRefiner(config=config)

#%%
from clicking.prompt_refinement.core import UIResponse

results = asyncio.run(prompt_refiner.process_single_prompt(clicking_images[1].image, mode=PromptMode.IMAGE_TO_UI_ELEMENTS, output_type=UIResponse))

#%%
state = PipelineState(images=clicking_images)
state = prompt_refiner.process_prompts(state, mode=PromptMode.IMAGE_TO_UI_ELEMENTS)
#%%
for image in state.images:
    # plt.imshow(image.image)
    # plt.axis('off')
    # plt.show()
    for element in image.ui_elements:
        print(element.name, element.icon)
#%%
clicking_images[1].image
#%%
for result in results:
    print(result.name, result.icon)

#%%
from clicking.image_processor.visualization import show_ocr_boxes

ocr_result = ocr_processor.get_ocr_results(PipelineState(images =clicking_images))
#%%
from clicking.common.logging import print_ocr_results

for image, result in zip(clicking_images, ocr_result):
    # remove token '</s>' from the labels
    result.prediction.labels = [label.replace('</s>', '') for label in result.prediction.labels]
    print_ocr_results(result)
    show_ocr_boxes(image, result.prediction)
# %%

for response in ocr_result:
    for i,image in enumerate(state.images):
        if image.id != response.id:
            continue
        plt.imshow(image.image)
        plt.axis('off')
        plt.show()
        for element in image.ui_elements:
            found = False
            for label in response.prediction.labels: 
                if element.name.lower() in label.lower():
                    # print('Matched', element.name, label)
                    found = True
                    break
            if not found:
                print('Not found', element.name)