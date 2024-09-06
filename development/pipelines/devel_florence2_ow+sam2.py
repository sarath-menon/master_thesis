#%%

%load_ext autoreload
%autoreload 2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from clicking.pipeline.core import Pipeline
from clicking.dataset_creator.core import CocoDataset
from clicking.prompt_refinement.core import PromptRefiner
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
from clicking.pipeline.core import PipelineState
from clicking.image_processor.localization import Localization
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
from clicking.prompt_refinement.core import PromptMode

pipeline = Pipeline(config=config)

pipeline.add_step("Process Prompts",
    lambda state: prompt_refiner.process_prompts(state.images, 
    mode=PromptMode.OBJECTS_LIST_TO_DESCRIPTIONS)
)
pipeline.add_step("Get Localization Results", 
    lambda state: localization_processor.get_localization_results(state, mode=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB)
)
pipeline.add_step("Verify bboxes", verify_bboxes)
pipeline.add_step("Get Segmentation Results", segmentation_processor.get_segmentation_results)

pipeline.print_pipeline()
#%% Run the entire pipeline, stopping after "Verify bboxes" step
from clicking.common.logging import print_object_descriptions

loaded_state = pipeline.load_state()
loaded_state = loaded_state.filter_by_id(image_ids=[0,12,37])
loaded_state = loaded_state.filter_by_object_category(ObjectCategory.GAME_ASSET)
print_object_descriptions(loaded_state.images, show_image=True, show_stats=False)
#%%
results = asyncio.run(pipeline.run( 
    # initial_images=clicking_images, 
    initial_state=loaded_state,
    start_from_step="Get Localization Results",
    stop_after_step="Get Localization Results",
))

#%% Visualize results
for clicking_image in results.images:
    show_localization_predictions(clicking_image) 
    # show_segmentation_predictions(clicking_image)

#%%
output_corrector_results =  output_corrector.verify_bboxes(results.images[0])

#%%
id = 0
show_localization_predictions(results.images[id], object_names_to_show=['Sewing Machine']) 
#%%
from clicking.evaluator.core import save_validity_results

# Example usage:
EVALS_PATH = "./datasets/evals/output_corrector"
save_validity_results(results, EVALS_PATH)
#%%
from clicking.evaluator.core import evaluate_validity_results

# Example usage:
EVALS_PATH = "./evals/output_corrector"
ground_truth_file = f'{EVALS_PATH}/ground_truth.json'
predictions_file = f'{EVALS_PATH}/validity_results.json'

evaluation_results = evaluate_validity_results(ground_truth_file, predictions_file)

#%%
from clicking.evaluator.core import save_image_descriptions
 
EVALS_PATH = "./datasets/evals/output_corrector"
save_image_descriptions(results.images, EVALS_PATH, prompt_path=config['prompts']['refinement_path'])

# %%
from clicking.evaluator.core import load_pipeline_results

results = load_pipeline_results("./datasets/evals/output_corrector/image_descriptions.json", coco_dataset)

#%%
import pickle

# Example usage:
loaded_state = pipeline.load_state()
#%%

 # Example usage:
pipeline.save_state(results, save_as_json=True, log_to_wandb=False)
#%%
pipeline.save_state_as_json(loaded_state)

#%%

selv = pipeline.create_state_json(loaded_state)
import pandas as pd

# %%
from dataclasses import dataclass, field
from typing import Dict, List, Any
from itertools import product
from prettytable import PrettyTable
from clicking.prompt_refinement.core import PromptMode
from clicking.image_processor.localization import InputMode
from clicking.vision_model.data_structures import TaskType

def load_pipeline_mode_sequences(config, sequence_name=None):
    sequences = config.get('pipeline_mode_sequences', {})
    
    def verify_enum(enum_class, value, field_name):
        if not hasattr(enum_class, value):
            raise ValueError(f"Invalid {field_name}: {value} is not a valid {enum_class.__name__}")
        return getattr(enum_class, value)

    if sequence_name:
        if sequence_name not in sequences:
            raise ValueError(f"Sequence '{sequence_name}' not found in config")
        seq = sequences[sequence_name]
        return [{
            "name": sequence_name,
            "prompt_modes": verify_enum(PromptMode, seq['prompt_mode'], "prompt_mode"),
            "localization_input_modes": verify_enum(InputMode, seq['localization_input_mode'], "localization_input_mode"),
            "localization_modes": verify_enum(TaskType, seq['localization_mode'], "localization_mode"),
            "segmentation_modes": verify_enum(TaskType, seq['segmentation_mode'], "segmentation_mode")
        }]
    
    return [
        {
            "name": name,
            "prompt_modes": verify_enum(PromptMode, seq['prompt_mode'], "prompt_mode"),
            "localization_input_modes": verify_enum(InputMode, seq['localization_input_mode'], "localization_input_mode"),
            "localization_modes": verify_enum(TaskType, seq['localization_mode'], "localization_mode"),
            "segmentation_modes": verify_enum(TaskType, seq['segmentation_mode'], "segmentation_mode")
        }
        for name, seq in sequences.items()
    ]

@dataclass
class PipelineModes:
    modes: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config):
        return cls(modes=load_pipeline_mode_sequences(config))

    def get_mode_combinations(self):
        return self.modes

    def print_mode_sequences(self):
        if not self.modes:
            print("No mode sequences found.")
            return

        table = PrettyTable(["Index"] + list(self.modes[0].keys()))
        for i, combination in enumerate(self.modes):
            table.add_row([i] + list(combination.values()))
        print(table)


def run_pipeline_for_all_modes(initial_state: PipelineState, pipeline_modes: PipelineModes) -> List[Dict]:
    results = []

    for i, combination in enumerate(pipeline_modes.get_mode_combinations()):
        print(f"Running combination {i + 1}/{len(pipeline_modes.get_mode_combinations())}")
        
        pipeline = Pipeline(config=config)
        
        pipeline.add_step("Process Prompts", lambda state: prompt_refiner.process_prompts(state.images, mode=combination["prompt_modes"]))
        pipeline.add_step("Get Localization Results", lambda state: localization_processor.get_localization_results(
            state, mode=combination["localization_modes"], input_mode=combination["localization_input_modes"]
        ))
        pipeline.add_step("Get Segmentation Results", lambda state: segmentation_processor.get_segmentation_results(state, mode=combination["segmentation_modes"]))

        pipeline_modes.print_mode_sequences()

        pipeline_result = asyncio.run(pipeline.run(
            initial_state=initial_state,
            start_from_step="Get Localization Results",
            stop_after_step="Get Localization Results",
        ))

        results.append({"combination": i, **combination, "pipeline_result": pipeline_result})

    return results

#%% Usage example:

import nest_asyncio
nest_asyncio.apply()

with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
pipeline_modes = PipelineModes.from_config(config)
pipeline_modes.print_mode_sequences()

#%%

loaded_state = pipeline.load_state()
loaded_state = loaded_state.filter_by_id(image_ids=[0, 12, 37])
loaded_state = loaded_state.filter_by_object_category(ObjectCategory.GAME_ASSET)

all_results = run_pipeline_for_all_modes(loaded_state, pipeline_modes)

# Print a summary of the results
summary_table = PrettyTable()
summary_table.field_names = ["Combination", "Name", "Prompt Mode", "Localization Input Mode", "Localization Mode", "Segmentation Mode", "Num Images", "Num Objects"]

for result in all_results:
    summary_table.add_row([
        result["combination"],
        result["name"],
        result["prompt_modes"],
        result["localization_input_modes"],
        result["localization_modes"],
        result["segmentation_modes"],
        len(result["pipeline_result"].images),
        sum(len(img.predicted_objects) for img in result["pipeline_result"].images)
    ])

print(summary_table)
# %%
from clicking.vision_model.data_structures import PromptMode, InputMode, TaskType

enums_dict = {
    "prompt_mode": PromptMode,
    "localization_input_mode": InputMode,
    "localization_mode": TaskType,
    "segmentation_mode": TaskType
}

def generate_config_schema(enums_dict):
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "pipeline_mode_sequences": {
                "type": "object",
                "patternProperties": {
                    "^[a-zA-Z0-9_]+$": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                "additionalProperties": False
            }
        }
    }

    properties = schema["properties"]["pipeline_mode_sequences"]["patternProperties"]["^[a-zA-Z0-9_]+$"]["properties"]
    required = schema["properties"]["pipeline_mode_sequences"]["patternProperties"]["^[a-zA-Z0-9_]+$"]["required"]

    for field_name, enum_class in enums_dict.items():
        properties[field_name] = {
            "type": "string",
            "enum": [e.name for e in enum_class]
        }
        required.append(field_name)

    # Special case for segmentation_mode
    properties["segmentation_mode"]["enum"] = ["SEGMENTATION_WITH_BBOX"]

    return schema

# Generate and save the config schema
import json

config_schema = generate_config_schema(enums_dict)
with open('config_schema.json', 'w') as f:
    json.dump(config_schema, f, indent=2)

print("Config schema generated and saved to 'config_schema.json'")

# Generate the config schema
generate_config_schema(enums_dict)
