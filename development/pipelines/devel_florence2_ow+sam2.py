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
def generate_config_schema(modes_dict: Dict):
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "pipeline_mode_sequences": {
                "type": "object",
                "patternProperties": {
                    "^[a-zA-Z0-9_]+$": {
                        "type": "object",
                        "properties": {field: {"type": "string", "enum": [e.name for e in enum_class]} for field, enum_class in modes_dict.items()},
                        "required": list(modes_dict.keys())
                    }
                },
                "additionalProperties": False
            }
        }
    }

    schema["properties"]["pipeline_mode_sequences"]["patternProperties"]["^[a-zA-Z0-9_]+$"]["properties"]["segmentation_mode"]["enum"] = ["SEGMENTATION_WITH_BBOX"]

    return schema
    
# Generate and save the config schema
config_schema = generate_config_schema(modes_dict)
with open('config_schema.json', 'w') as f:
    json.dump(config_schema, f, indent=2)

print("Config schema generated and saved to 'config_schema.json'")

#%%

from dataclasses import dataclass, field
from typing import Dict, List, Any, Type
from prettytable import PrettyTable
import asyncio
from clicking.pipeline.core import Pipeline, PipelineState
from clicking.image_processor.localization import Localization, InputMode

@dataclass
class PipelineMode:
    name: str
    modes: Dict[str, Any]

@dataclass
class PipelineModes:
    modes: List[PipelineMode] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Dict, modes_dict: Dict[str, Type]):
        sequences = config.get('pipeline_mode_sequences', {})
        modes = []
        for name, seq in sequences.items():
            mode_values = {}
            for mode_name, enum_class in modes_dict.items():
                mode_values[mode_name] = enum_class[seq[mode_name]]
            modes.append(PipelineMode(name=name, modes=mode_values))
        return cls(modes=modes)

    def print_mode_sequences(self):
        if not self.modes:
            print("No mode sequences found.")
            return

        headers = ["Index", "Name"] + list(self.modes[0].modes.keys())
        table = PrettyTable(headers)
        for i, mode in enumerate(self.modes):
            row = [i, mode.name] + list(mode.modes.values())
            table.add_row(row)
        print(table)

def run_pipeline_for_all_modes(
    initial_state: PipelineState, 
    pipeline_modes: PipelineModes, 
    config: Dict,
    pipeline_structure: List[PipelineStep]
) -> List[Dict]:
    results = []

    for i, mode in enumerate(pipeline_modes.modes):
        print(f"Running combination {i + 1}/{len(pipeline_modes.modes)}")
        
        pipeline = Pipeline(config=config)
        
        for step in pipeline_structure:
            pipeline.add_step(
                PipelineStep(
                    name=step.name,
                    function=lambda state, step=step, mode=mode: step.function(
                        state, **{k: mode.modes[k] for k in step.mode_keys}
                    ),
                    mode_keys=step.mode_keys
                )
            )

        pipeline_result = asyncio.run(pipeline.run(
            initial_state=initial_state,
            start_from_step=pipeline_structure[1].name,
            stop_after_step=pipeline_structure[1].name,
        ))

        results.append({"combination": i, **mode.modes, "pipeline_result": pipeline_result})

    return results

def print_summary(results: List[Dict]):
    if not results:
        print("No results to summarize.")
        return

    headers = ["Combination", "Name"] + list(results[0].keys())[2:-1] + ["Num Images", "Num Objects"]
    summary_table = PrettyTable(headers)

    for result in results:
        row = [
            result["combination"],
            result.get("name", "N/A"),
            *[result[key] for key in headers[2:-2]],
            len(result["pipeline_result"].images),
            sum(len(img.predicted_objects) for img in result["pipeline_result"].images)
        ]
        summary_table.add_row(row)

    print(summary_table)

# Usage example:
modes_dict = {
    "prompt_mode": PromptMode,
    "localization_input_mode": InputMode,
    "localization_mode": TaskType,
    "segmentation_mode": TaskType
}

from components.clicking.pipeline.core import PipelineStep, PipelineState

pipeline_structure: List[PipelineStep] = [
    PipelineStep(
        name="Process Prompts",
        function=prompt_refiner.process_prompts,
        mode_keys=["prompt_mode"]
    ),
    PipelineStep(
        name="Get Localization Results",
        function=localization_processor.get_localization_results,
        mode_keys=["localization_mode", "localization_input_mode"]
    ),
    PipelineStep(
        name="Get Segmentation Results",
        function=segmentation_processor.get_segmentation_results,
        mode_keys=["segmentation_mode"]
    )
]

#%% The rest of the code remains the same
pipeline_modes = PipelineModes.from_config(config, modes_dict)
pipeline_modes.print_mode_sequences()

loaded_state = pipeline.load_state()
loaded_state = loaded_state.filter_by_id(image_ids=[0, 12, 37])
loaded_state = loaded_state.filter_by_object_category(ObjectCategory.GAME_ASSET)

all_results = run_pipeline_for_all_modes(loaded_state, pipeline_modes, config, pipeline_structure)
print_summary(all_results)
# %%
