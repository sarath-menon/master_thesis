from typing import Callable, List, Any, Dict, Tuple, TypedDict, get_origin, get_args, TypeVar, Generic
import inspect
import matplotlib.pyplot as plt
from PIL import Image

from typing import Type
from tabulate import tabulate
from dataclasses import dataclass, field
from datetime import datetime
import os
import pickle
from typing import Union, Optional
from clicking.common.data_structures import *
import asyncio
import json
import markdown
import bleach
import wandb
import yaml
import shutil
import random
from typing import Dict, List, Any, Type
from dataclasses import dataclass, field
from prettytable import PrettyTable
import asyncio
from tqdm import tqdm
import copy
from clicking.common.data_structures import PipelineState
from functools import lru_cache
from functools import wraps
import hashlib
import pickle
from tqdm.asyncio import tqdm as async_tqdm
import time

T = TypeVar('T')

class PipelineModes:
    def __init__(self, modes: Dict[str, Type]):
        self.modes = modes

    def __getattr__(self, name):
        return self.modes[name]

def custom_cache(func):
    cache = {}
    @wraps(func)
    def wrapper(state: PipelineState, *args, **kwargs):
        # Create a unique key based on the function name, state hash, and arguments
        key = hashlib.md5(pickle.dumps((func.__name__, hash(state), args, kwargs))).hexdigest()
        if key not in cache:
            cache[key] = func(state, *args, **kwargs)
        return cache[key]
    return wrapper

@dataclass
class PipelineStep(Generic[T]):
    name: str
    function: Callable[[PipelineState, T], PipelineState]
    mode_keys: List[str]
    use_cache: bool = False

    def __post_init__(self):
        if self.use_cache:
            self.function = custom_cache(self.function)

@dataclass
class PipelineMode:
    name: str
    modes: Dict[str, Any]

class PipelineModeSequence:
    def __init__(self, modes=None):
        self.modes = modes if modes is not None else []

    @classmethod
    def from_config(cls, config: Dict, pipeline_modes: PipelineModes):
        sequences = config.get('pipeline_mode_sequences', {})
        modes = []
        for name, seq in sequences.items():
            mode_values = {}
            for mode_name, enum_class in pipeline_modes.modes.items():
                if mode_name in seq:
                    mode_values[mode_name] = enum_class[seq[mode_name]]
            modes.append(PipelineMode(name=name, modes=mode_values))
        return cls(modes=modes)

    def print_mode_sequences(self):
        if not self.modes:
            print("No mode sequences found.")
            return

        # Get all unique mode keys across all modes
        all_mode_keys = set()
        for mode in self.modes:
            all_mode_keys.update(mode.modes.keys())

        # Create headers for each mode and its name
        headers = ["Mode"] + [f"{mode.name} ({i})" for i, mode in enumerate(self.modes)]
        table = PrettyTable(headers)

        # Add rows for each mode key
        for key in all_mode_keys:
            row = [key] + [mode.modes.get(key, "-") for mode in self.modes]
            table.add_row(row)

        print(table)

    @classmethod
    def generate_config_schema(cls, pipeline_modes: PipelineModes):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "pipeline_mode_sequences": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z0-9_]+$": {
                            "type": "object",
                            "properties": {field: {"type": "string", "enum": [e.name for e in enum_class]} for field, enum_class in pipeline_modes.modes.items()},
                            "required": list(pipeline_modes.modes.keys())
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
        return schema

@dataclass
class PipelineSingleRun:
    combination: int
    name: str
    modes: Dict[str, Any]
    result: PipelineState

@dataclass
class PipelineRunResults:
    results: Dict[str, PipelineSingleRun]

    def get_run_by_mode_name(self, mode_name: str) -> Optional[PipelineSingleRun]:
        if mode_name not in self.results:
            print(f"Error: Mode '{mode_name}' does not exist.")
            return None
        return self.results[mode_name].result
    
class Pipeline:
    def __init__(self, config: Dict[str, Any], cache_folder= "./cache"):
        self.steps: List[PipelineStep] = []
        self.config = config
        self.cache_dir = config['pipeline']['cache_dir']
        os.makedirs(self.cache_dir, exist_ok=True)

    async def run(
        self,
        initial_images: List[ClickingImage] = None,
        initial_state: PipelineState = None,
        start_from_step: str = None,
        stop_after_step: str = None
    ) -> PipelineState:
        start_index = 0
        if start_from_step:
            start_index = self._find_step_index(start_from_step)
            if start_index == -1:
                raise ValueError(f"Invalid start_from_step: '{start_from_step}'. Step not found in the pipeline.")

        if stop_after_step and self._find_step_index(stop_after_step) == -1:
            raise ValueError(f"Invalid stop_after_step: '{stop_after_step}'. Step not found in the pipeline.")

        if initial_state is None and initial_images is None:
            raise ValueError("Either initial_state or initial_images must be provided.")
        
        if initial_images and start_from_step:
            raise ValueError("start_from_step is not supported when initial_images are provided.")

        try:
            initial_input = initial_state if initial_state else PipelineState(images=initial_images)
            result = await asyncio.shield(self._run_internal(initial_input, start_index=start_index, stop_after_step=stop_after_step))
            
            return result
        except asyncio.CancelledError:
            print("Pipeline execution was cancelled.")
            raise

    async def _run_internal(self, initial_input: Union[List[ClickingImage], PipelineState], start_index: int = 0, stop_after_step: str = None) -> PipelineState:
        state = initial_input if isinstance(initial_input, PipelineState) else PipelineState(images=initial_input)
        
        for i, step in enumerate(self.steps[start_index:], start=start_index):
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()

            try:
                print(f"Starting execution of step '{step.name}'")
                state = await asyncio.to_thread(step.function, state)
            except asyncio.CancelledError:
                print(f"Step '{step.name}' was cancelled.")
                raise
            
            if stop_after_step and step.name == stop_after_step:
                print(f"Stopping execution after step: {step.name}")
                break
        
        return state


    def _find_step_index(self, step_name: str) -> int:
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                return i
        return -1

    def print_pipeline(self):
        headers = ["No.", "Step Name", "Mode Keys", "Cached"]
        table_data = []
        
        for i, step in enumerate(self.steps, 1):
            table_data.append([i, step.name, ", ".join(step.mode_keys), "Yes" if step.use_cache else "No"])
        
        print("Pipeline Steps:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def load_state(self, file_path: str = None) -> PipelineState:
        try:
            if file_path is None:
                # Get all subdirectories in the cache directory
                subdirs = [d for d in os.listdir(self.cache_dir) if os.path.isdir(os.path.join(self.cache_dir, d))]
                if not subdirs:
                    raise FileNotFoundError("No subdirectories found in cache directory.")
                
                # Find the latest created subdirectory
                latest_subdir = max(subdirs, key=lambda d: os.path.getctime(os.path.join(self.cache_dir, d)))
                latest_subdir_path = os.path.join(self.cache_dir, latest_subdir)
                
                # Look for pipeline_state.pkl in the latest subdirectory
                file_path = os.path.join(latest_subdir_path, "pipeline_state.pkl")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"pipeline_state.pkl not found in the latest subdirectory: {latest_subdir_path}")
            
            try:
                with open(file_path, 'rb') as f:
                    state = pickle.load(f)
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
                return None
            except pickle.UnpicklingError:
                print(f"Error: Unable to unpickle file at {file_path}")
                return None
            except Exception as e:
                print(f"Unexpected error occurred while loading state: {str(e)}")
                return None

            creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            print(f"State from {creation_time.strftime('%d-%m-%Y %H:%M:%S')} loaded successfully")
            return state
        except Exception as e:
            print(f"Error loading pipeline state: {str(e)}")
            return None

    def save_state(self, state: PipelineState, save_as_json: bool = False, log_to_wandb: bool = False, name: str = None):
        try:
            from datetime import datetime
            import threading
            import wandb

            if name is None:
                current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                new_folder_path = os.path.join(self.cache_dir, current_time)
            else:
                new_folder_path = os.path.join(self.cache_dir, name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            pickle_thread = threading.Thread(target=self.save_state_pickle, args=(state, new_folder_path))
            pickle_thread.start()
            self.save_config_and_prompts(new_folder_path)

            metadata_thread = threading.Thread(target=self.save_metadata, args=(state, new_folder_path))
            metadata_thread.start()

            if save_as_json:
                json_thread = threading.Thread(target=self.save_state_as_json, args=(state, new_folder_path))
                json_thread.start()

            pickle_thread.join()
            metadata_thread.join()

            config_file_path = os.path.join(new_folder_path, "config.yml")
            with open(config_file_path, 'w') as f:
                yaml.dump(self.config, f)
            print(f"Config saved as YAML at {config_file_path}")

            if save_as_json:
                json_thread.join()
            
            if log_to_wandb:
                run=  wandb.init(project="clicking", name=f"state_{current_time}")
                wandb.save(os.path.join(new_folder_path, "*"))
                wandb.finish()
        
        except Exception as e:
            print(f"Error saving pipeline state: {str(e)}")

    def save_config_and_prompts(self, new_folder_path: str):
        try:
            config_file_path = os.path.join(new_folder_path, "config.yml")
            with open(config_file_path, 'w') as f:
                yaml.dump(self.config, f)
            print(f"Config saved as YAML at {config_file_path}")
        except Exception as e:
            print(f"Error saving config: {str(e)}")

        try:
            prompts_src_path = os.path.join('prompts')
            prompts_dest_path = os.path.join(new_folder_path, 'prompts')
            shutil.copytree(prompts_src_path, prompts_dest_path)
            print(f"Prompts folder copied to {prompts_dest_path}")
        except Exception as e:
            print(f"Error saving prompts: {str(e)}")

    def save_state_pickle(self, state: PipelineState, folder_path: str):
        try:
            new_file_path = os.path.join(folder_path, "pipeline_state.pkl")
            
            with open(new_file_path, 'wb') as f:
                pickle.dump(state, f)
            print(f"Pipeline state saved successfully to {new_file_path}")
        except Exception as e:
            print(f"Error saving pipeline state as pickle: {str(e)}")

    def save_metadata(self, state: PipelineState, folder_path: str):
        num_objects = len(state.images)
        metadata = f"Number of objects in PipelineState: {num_objects}"

        def markdown_to_plain_text(markdown_text):
            html = markdown.markdown(markdown_text)
            return bleach.clean(html, tags=[], strip=True)

        with open(os.path.join(folder_path, "metadata.md"), 'w') as f:
            f.write(markdown_to_plain_text(metadata))


    def create_state_json(self, state: PipelineState):
        def serialize_object(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [serialize_object(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): serialize_object(v) for k, v in obj.items()}
            elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return serialize_object(obj.__dict__)
            else:
                return str(obj)

        def serialize_image(img):
            if isinstance(img, ClickingImage):
                return {
                    "image_id": img.id,
                    "annotated_objects": serialize_object(img.annotated_objects),
                    "predicted_objects": serialize_object(img.predicted_objects)
                }
            return str(img)

        json_data = json.dumps(
            {"images": [serialize_image(img) for img in state.images]},
            indent=2,
            default=str
        )
        return json_data

    def save_state_as_json(self, state: PipelineState, folder_path: str = "pipeline_state"):
        try:
            new_file_path = os.path.join(folder_path, "pipeline_state.json")
            json_data = self.create_state_json(state)
            
            with open(os.path.join(os.getcwd(), new_file_path), 'w+') as f:
                f.write(json_data)
            
            print(f"Pipeline state saved as JSON successfully to {new_file_path}")
        except Exception as e:
            print(f"Error saving pipeline state as JSON: {str(e)}")


    async def run_for_all_modes(
        self,
        initial_state: Optional[PipelineState] = None,
        initial_images: Optional[List[ClickingImage]] = None,
        pipeline_modes: PipelineModeSequence = None,
        start_from_step: Optional[str] = None,
        stop_after_step: Optional[str] = None
    ) -> PipelineRunResults:
        results = {}

        if initial_state is None and initial_images is None:
            raise ValueError("Either initial_state or initial_images must be provided.")

        start_time = time.time()

        for i, mode in enumerate(pipeline_modes.modes):
            print(f"Running mode: {mode.name}")
            
            # Create deep copies of the initial state or images for each run
            initial_state_copy = copy.deepcopy(initial_state) if initial_state else None
            initial_images_copy = copy.deepcopy(initial_images) if initial_images else None
            
            # Create a new Pipeline instance for each mode
            temp_pipeline = Pipeline(self.config)
            
            # Create a copy of the steps
            temp_steps = [PipelineStep(step.name, step.function, step.mode_keys, step.use_cache) for step in self.steps]
            
            # Apply mode-specific configurations
            for step in temp_steps:
                step_modes = {k: mode.modes[k] for k in step.mode_keys if k in mode.modes}
                original_function = step.function
                def create_step_function(orig_func, modes):
                    return lambda state: orig_func(state, **modes)
                step.function = create_step_function(original_function, step_modes)
                if step.use_cache:
                    step.function = custom_cache(step.function)

            # Set the modified steps to the new pipeline instance
            temp_pipeline.steps = temp_steps

            result = await temp_pipeline.run(
                initial_state=initial_state_copy,
                initial_images=initial_images_copy,
                start_from_step=start_from_step,
                stop_after_step=stop_after_step,
            )

            single_run = PipelineSingleRun(
                combination=i,
                name=mode.name,
                modes=mode.modes,
                result=result
            )
            results[mode.name] = single_run

            end_time = time.time() 
            elapsed_time = end_time - start_time
            print(f"Completed mode '{mode.name}' in {elapsed_time:.2f} seconds")

        return PipelineRunResults(results)

    def print_mode_results_summary(self, results: PipelineRunResults):
        if not results.results:
            print("No results to summarize.")
            return

        first_result = next(iter(results.results.values()))
        headers = ["Combination", "Name"] + list(first_result.modes.keys()) + ["Num Images", "Num Objects"]
        summary_table = PrettyTable(headers)

        for result in results.results.values():
            row = [
                result.combination,
                result.name,
                *[result.modes[key] for key in headers[2:-2]],
                len(result.result.images),
                sum(len(img.predicted_objects) for img in result.result.images)
            ]
            summary_table.add_row(row)

        print(summary_table)
