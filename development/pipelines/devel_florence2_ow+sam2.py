#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, NamedTuple
from clicking.pipeline.core import Pipeline
from clicking.dataset_creator.core import CocoDataset, DatasetSample
from clicking.prompt_refinement.core import PromptRefiner, PromptMode, ProcessedPrompts, ImageWithDescriptions
from clicking.vision_model.types import TaskType, LocalizationResults, SegmentationResults
from clicking.vision_model.bbox import BoundingBox, BBoxMode
from clicking.vision_model.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction
from tabulate import tabulate
import pickle
import os
from datetime import datetime
import asyncio
from typing import Callable, List, Any, Dict, Tuple, Union
import inspect
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os
from datetime import datetime
from tabulate import tabulate
import asyncio
from clicking.vision_model.visualization import show_localization_predictions, show_segmentation_predictions
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

# Load the configuration file
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

@dataclass
class PipelineState:
    image_ids: List[int] = field(default_factory=list)
    dataset_sample: Optional[DatasetSample] = None
    processed_prompts: Optional[ProcessedPrompts] = None
    localization_results: Optional[LocalizationResults] = None
    segmentation_results: Optional[SegmentationResults] = None

#%%

client = Client(base_url=config['api']['base_url'])

prompt_refiner = PromptRefiner(prompt_path=config['prompts']['refinement_path'])

coco_dataset = CocoDataset(config['dataset']['images_path'], config['dataset']['annotations_path'])
#%%

from typing import Callable, List, Any, Dict, Tuple, TypedDict, get_origin, get_args
import inspect
import matplotlib.pyplot as plt
from PIL import Image 
from typing import Type
import pickle
from tabulate import tabulate
from datetime import datetime
from typing import Union
import asyncio

class Pipeline:
    def __init__(self):
        self.steps: List[Tuple[str, Callable, str, bool]] = []
        self.visualization_functions: Dict[Type, Callable] = {
            LocalizationResults: show_localization_predictions,
            SegmentationResults: show_segmentation_predictions
        }
        self.cache_dir = config['pipeline']['cache_dir']
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_filename = self._generate_cache_filename()
        self.cache_data: Dict[str, Any] = {}

    def _generate_cache_filename(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.cache_dir, f"pipeline_cache_{timestamp}.pkl")

    def _save_cache(self):
        with open(self.cache_filename, 'wb') as f:
            pickle.dump(self.cache_data, f)

    def _load_cache(self):
        if not os.path.exists(self.cache_filename):
            return

        try:
            with open(self.cache_filename, 'rb') as f:
                self.cache_data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print(f"Warning: Cache file is corrupted. Ignoring cache.")
            os.remove(self.cache_filename)
            self.cache_data = {}

    async def run(self, initial_input: List[int]) -> PipelineState:
        self.cache_data = {}  # Reset cache for a new run
        return await self._run_internal(initial_input)

    async def run_from_step(self, step_name: str, initial_state: PipelineState = None) -> PipelineState:
        step_index = self._find_step_index(step_name)
        if step_index == -1:
            raise ValueError(f"Step '{step_name}' not found in the pipeline.")
        
        if initial_state is None:
            self._load_cache()
            if step_name not in self.cache_data:
                raise ValueError(f"No cached state found for step '{step_name}'. Please provide an initial state.")
            initial_state = self.cache_data[step_name]
        
        return await self._run_internal(initial_state, start_index=step_index)

    async def _run_internal(self, initial_input: Union[List[int], PipelineState], start_index: int = 0) -> PipelineState:
        state = initial_input if isinstance(initial_input, PipelineState) else PipelineState(image_ids=initial_input)
        
        for i, (step_name, step_func, log_var, verbose) in enumerate(self.steps[start_index:], start=start_index):
            if step_name in self.cache_data:
                print(f"Using cached input for step: {step_name}")
                state = self.cache_data[step_name]
            else:
                state = await asyncio.to_thread(step_func, state)
                self.cache_data[step_name] = state
                self._save_cache()
            
            if verbose:
                self._log_step_result(step_name, getattr(state, log_var), log_var)
        
        return state

    def get_step_result(self, step_name: str) -> Any:
        self._load_cache()
        return self.cache_data.get(step_name)

    def _find_step_index(self, step_name: str) -> int:
        for i, (name, _, _, _) in enumerate(self.steps):
            if name == step_name:
                return i
        return -1

    def _log_step_result(self, step_name: str, result: Any, log_var: str):
        print(f"\n--- Step: {step_name} ---")
        print(f"Logging variable: {log_var}")
        self._recursive_log(step_name, result)
        
        result_type = type(result)
        if result_type in self.visualization_functions:
            self.visualization_functions[result_type](result)

    def _recursive_log(self, step_name: str, result: Any, prefix: str = ""):
        if isinstance(result, Image.Image):
            self._display_image(step_name, prefix, result)
        elif isinstance(result, list):
            self._log_list(step_name, result, prefix)
        elif isinstance(result, dict):
            self._log_dict(step_name, result, prefix)
        elif isinstance(result, tuple):
            for item in result:
                self._recursive_log(step_name, item, prefix)
        else:
            print(f"{prefix}{result}")

    def _log_list(self, step_name: str, result: List[Any], prefix: str):
        if not result:
            print(f"{prefix}Empty list")
            return

        if all(isinstance(item, Image.Image) for item in result):
            self._display_image_list(step_name, prefix, result)
            return

        for i, item in enumerate(result):
            if i > 0:
                print(f"{prefix}---")
            self._recursive_log(step_name, item, prefix)

    def _log_dict(self, step_name: str, result: Dict[str, Any], prefix: str):
        for key, value in result.items():
            if key != "objects":
                print(f"{prefix}{key}: ", end="")
                self._recursive_log(step_name, value, "")
            else:
                self._recursive_log(step_name, value, prefix)

    def _display_image(self, step_name: str, prefix: str, image: Image.Image):
        plt.figure(figsize=tuple(config['visualization']['figsize']))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{step_name} - Image")
        plt.show()

    def _display_image_list(self, step_name: str, prefix: str, images: List[Image.Image]):
        if not images:
            print(f"{prefix}[]")
            return
        
        fig, axes = plt.subplots(1, len(images), figsize=(config['visualization']['figsize'][0]*len(images), config['visualization']['figsize'][1]))
        if len(images) == 1:
            axes = [axes]
        for i, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{step_name} - Image {i+1}")
        plt.tight_layout()
        plt.show()

    def static_analysis(self):
        if not self.steps:
            raise ValueError("Pipeline has no steps.")
        
        for i in range(len(self.steps) - 1):
            current_step = self.steps[i]
            next_step = self.steps[i + 1]
            
            current_return_type = inspect.signature(current_step[1]).return_annotation
            next_param_type = list(inspect.signature(next_step[1]).parameters.values())[0].annotation
            
            if current_return_type != Any and next_param_type != Any:
                if not issubclass(current_return_type, next_param_type):
                    raise TypeError(f"Output type of {current_step[0]} ({current_return_type}) "
                                    f"is not compatible with input type of {next_step[0]} ({next_param_type})")
        
        print("Static analysis complete. All types are compatible.")

    def print_pipeline(self):
        headers = ["Step", "Function Name", "Input Type"]
        table_data = []
        
        for i, (step_name, func, log_var, _) in enumerate(self.steps, 1):
            input_type = list(inspect.signature(func).parameters.values())[0].annotation.__name__
            table_data.append([f"{i}. {step_name}", func.__name__, input_type])
        
        print("Pipeline Steps:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def add_step(self, step_name: str, func: Callable, log_var: str, verbose: bool = True):
        if self.steps and not self._are_types_compatible(self.steps[-1][1], func):
            last_step_func = self.steps[-1][1]
            last_step_output_type = inspect.signature(last_step_func).return_annotation.__name__
            next_step_input_type = list(inspect.signature(func).parameters.values())[0].annotation.__name__

            raise TypeError(f"Output type of {last_step_func.__name__} ({last_step_output_type}) is not compatible with input type of {func.__name__} ({next_step_input_type})")
        self.steps.append((step_name, func, log_var, verbose))

    def _are_types_compatible(self, prev_func: Callable, next_func: Callable) -> bool:
        prev_return_type = inspect.signature(prev_func).return_annotation
        next_param_type = list(inspect.signature(next_func).parameters.values())[0].annotation
        
        if prev_return_type == Any or next_param_type == Any:
            return True
        
        return issubclass(prev_return_type, next_param_type)

#%%
from clicking_client.types import File
from io import BytesIO
import json

def image_to_http_file(image):
    # Convert PIL Image to bytes and create a File object
    image_byte_arr = BytesIO()
    image.save(image_byte_arr, format='JPEG')
    image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")
    return image_file

# Modify the relevant steps to use PipelineState

def sample_dataset(state: PipelineState) -> PipelineState:
    state.dataset_sample = coco_dataset.sample_dataset(state.image_ids)
    return state

def process_prompts(state: PipelineState) -> PipelineState:
    state.processed_prompts = prompt_refiner.process_prompts(state.dataset_sample)
    return state

class LocalizationProcessor:
    def __init__(self, client: Client):
        self.client = client

    def get_localization_results(self, state: PipelineState) -> PipelineState:
        try:
            set_model.sync(client=self.client, body=SetModelReq(
                name=config['models']['localization']['name'],
                variant=config['models']['localization']['variant'],
                task=TaskType[config['models']['localization']['task']]
            ))
        except Exception as e:
            print(f"Error setting localization model: {str(e)}")
            return state
        
        all_predictions = {}
        for sample in state.processed_prompts.samples:
            image_file = image_to_http_file(sample.image)
            image_id = sample.image_id

            all_predictions[image_id] = []
            for obj in sample.description["objects"]:
                request = BodyGetPrediction(
                    image=image_file,
                )
                try:
                    response = get_prediction.sync(client=self.client,
                        body=request,
                        task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
                        input_text=obj["description"]
                    )

                    bboxes = [BoundingBox(bbox, mode=BBoxMode.XYWH, object_name=obj["name"], description=obj["description"]) 
                              for bbox in response.prediction.bboxes]
                    all_predictions[image_id].extend(bboxes)
                except Exception as e:
                    print(f"Error getting prediction for image {image_id}, object {obj['name']}: {str(e)}")
        
        state.localization_results = LocalizationResults(
            processed_samples=state.processed_prompts.samples,
            predictions=all_predictions
        )
        return state

class SegmentationProcessor:
    def __init__(self, client: Client):
        self.client = client

    def get_segmentation_results(self, state: PipelineState) -> PipelineState:
        try:
            set_model.sync(client=self.client, body=SetModelReq(
                name=config['models']['segmentation']['name'],
                variant=config['models']['segmentation']['variant'],
                task=TaskType[config['models']['segmentation']['task']]
            ))
        except Exception as e:
            print(f"Error setting segmentation model: {str(e)}")
            return state
        
        segmentation_results = {}
        for sample in state.localization_results.processed_samples:
            image_file = image_to_http_file(sample.image)
            image_id = sample.image_id

            seg_predictions = []
            for bbox in state.localization_results.predictions[image_id]:
                request = BodyGetPrediction(
                    image=image_file
                )
                try:
                    response = get_prediction.sync(client=self.client,
                        body=request,
                        task=TaskType.SEGMENTATION_WITH_BBOX,
                        input_boxes=json.dumps(bbox.get(mode=BBoxMode.XYWH))  
                    )

                    if response is None or response.prediction is None:
                        print(f"Warning: No prediction received for image {image_id}, bbox {bbox}")
                        continue
                    
                    for mask_data in response.prediction.masks:
                        seg_mask = SegmentationMask(
                            mask=mask_data,
                            mode=SegmentationMode.COCO_RLE,
                            object_name=bbox.object_name,
                            description=bbox.description
                        )
                        seg_predictions.append(seg_mask)
                except Exception as e:
                    print(f"Error processing segmentation for image {image_id}, bbox {bbox}: {str(e)}")
                    continue
            
            segmentation_results[image_id] = seg_predictions
        
        state.segmentation_results = SegmentationResults(
            processed_samples=state.localization_results.processed_samples,
            predictions=segmentation_results
        )
        return state

#%%
import nest_asyncio
nest_asyncio.apply()

pipeline = Pipeline()
localization_processor = LocalizationProcessor(client)
segmentation_processor = SegmentationProcessor(client)
output_corrector = OutputCorrector(prompt_path=config['prompts']['output_corrector_path'])

pipeline.add_step("Sample Dataset", sample_dataset, "dataset_sample", True)
pipeline.add_step("Process Prompts", process_prompts, "processed_prompts", True)
pipeline.add_step("Get Localization Results", localization_processor.get_localization_results, "localization_results", True)
pipeline.add_step("Get Segmentation Results", segmentation_processor.get_segmentation_results, "segmentation_results", True)

# Print the pipeline structure
pipeline.print_pipeline()

# Perform static analysis before running the pipeline
pipeline.static_analysis()

# Run the entire pipeline
image_ids = [22, 31, 34]
result = asyncio.run(pipeline.run(image_ids))
#%%

# Run from a specific step using cached data
result = asyncio.run(pipeline.run_from_step("Get Localization Results"))

# # Or provide an initial state if needed
# initial_state = PipelineState(image_ids=[22, 31, 34])
# initial_state.processed_prompts = pipeline.get_step_result("Process Prompts").processed_prompts
# result = asyncio.run(pipeline.run_from_step("Get Localization Results", initial_state))

#%%
# Access cached results for logging or analysis
localization_results = pipeline.get_step_result("Get Localization Results")
segmentation_results = pipeline.get_step_result("Get Segmentation Results")

# Visualize results
show_localization_predictions(localization_results)
show_segmentation_predictions(segmentation_results)
#%%

