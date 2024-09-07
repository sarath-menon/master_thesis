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
from clicking.common.data_structures import ClickingImage, ObjectCategory
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

T = TypeVar('T')


@dataclass
class PipelineState:
    images: List[ClickingImage] = field(default_factory=list)

    def filter_by_object_category(self, category: ObjectCategory):
        for clicking_image in self.images:
            clicking_image.predicted_objects = [
                obj for obj in clicking_image.predicted_objects
                if obj.category == category
            ]
        return self

    def filter_by_id(self, image_ids: Optional[List[int]] = None, sample_size: Optional[int] = None):
        if image_ids is not None and sample_size is not None:
            raise ValueError("Cannot specify both image_ids and sample_size. Choose one filtering method.")
        
        if image_ids is not None:
            image_ids = [str(id) for id in image_ids]
            self.images = [img for img in self.images if img.id in image_ids]
        elif sample_size is not None:
            if sample_size > len(self.images):
                raise ValueError(f"Sample size {sample_size} is larger than the number of available images {len(self.images)}")
            self.images = random.sample(self.images, sample_size)
        
        return self

@dataclass
class PipelineStep(Generic[T]):
    name: str
    function: Callable[[PipelineState, T], PipelineState]
    mode_keys: List[str]

@dataclass
class PipelineMode:
    name: str
    modes: Dict[str, Any]

class PipelineModeSequence:
    def __init__(self, modes=None):
        self.modes = modes if modes is not None else []

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

    @classmethod
    def generate_config_schema(cls, modes_dict: Dict[str, Type]):
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
        return schema

class Pipeline:
    def __init__(self, config: Dict[str, Any], cache_folder= "./cache"):
        self.steps: List[PipelineStep] = []
        self.config = config
        self.cache_dir = config['pipeline']['cache_dir']
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_filename = self._generate_cache_filename()
        self.cache_data: Dict[str, Any] = {}
        self.last_run_cache: Dict[str, Any] = {}

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

    async def run(
        self,
        initial_images: List[ClickingImage] = None,
        initial_state: PipelineState = None,
        start_from_step: str = None,
        stop_after_step: str = None,
        reset_cache: bool = False
    ) -> PipelineState:
        start_index = 0
        if start_from_step:
            start_index = self._find_step_index(start_from_step)
            if start_index == -1:
                raise ValueError(f"Invalid start_from_step: '{start_from_step}'. Step not found in the pipeline.")

        if stop_after_step and self._find_step_index(stop_after_step) == -1:
            raise ValueError(f"Invalid stop_after_step: '{stop_after_step}'. Step not found in the pipeline.")

        if start_index > 0 and not reset_cache:
            self._load_cache()
            if start_from_step not in self.cache_data and start_from_step not in self.last_run_cache:
                if initial_state is None and initial_images is None:
                    raise ValueError(f"No cached state found for step '{start_from_step}' and no initial input provided. Please provide an initial state or images, or run from the beginning.")
            else:
                initial_state = self.cache_data.get(start_from_step) or self.last_run_cache.get(start_from_step)
                # Clear the state for all steps after start_from_step
                for step in self.steps[start_index + 1:]:
                    self.cache_data.pop(step.name, None)
                    self.last_run_cache.pop(step.name, None)
        elif initial_state is None and initial_images is None:
            raise ValueError("Either initial_state or initial_images must be provided when starting from the beginning of the pipeline.")

        # Reset cache when explicitly requested or starting from the beginning
        if reset_cache or start_index == 0:
            self.cache_data = {}  # Reset cache for a new run

        try:
            initial_input = initial_state if initial_state else PipelineState(images=initial_images)
            result = await asyncio.shield(self._run_internal(initial_input, start_index=start_index, stop_after_step=stop_after_step))
            
            # Store the cache from this run
            self.last_run_cache = self.cache_data.copy()
            
            return result
        except asyncio.CancelledError:
            print("Pipeline execution was cancelled.")
            raise

    async def _run_internal(self, initial_input: Union[List[ClickingImage], PipelineState], start_index: int = 0, stop_after_step: str = None) -> PipelineState:
        state = initial_input if isinstance(initial_input, PipelineState) else PipelineState(images=initial_input)
        
        for i, step in enumerate(self.steps[start_index:], start=start_index):
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()

            if step.name in self.cache_data:
                print(f"Using cached input for step: {step.name}")
                state = self.cache_data[step.name]
            elif step.name in self.last_run_cache:
                print(f"Using last run cached input for step: {step.name}")
                state = self.last_run_cache[step.name]
            
            try:
                state = await asyncio.wait_for(asyncio.to_thread(step.function, state), timeout=None)
            except asyncio.CancelledError:
                print(f"Step '{step.name}' was cancelled.")
                raise
            
            if i < len(self.steps) - 1:
                self.cache_data[self.steps[i+1].name] = state
                self._save_cache()
            
            if stop_after_step and step.name == stop_after_step:
                print(f"Stopping execution after step: {step.name}")
                break
        
        return state

    def get_step_result(self, step_name: str) -> Any:
        self._load_cache()
        return self.cache_data.get(step_name)

    def _find_step_index(self, step_name: str) -> int:
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                return i
        return -1

    def print_pipeline(self):
        headers = ["No.", "Step Name", "Mode Keys"]
        table_data = []
        
        for i, step in enumerate(self.steps, 1):
            table_data.append([i, step.name, ", ".join(step.mode_keys)])
        
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
            
            with open(file_path, 'rb') as f:
                state = pickle.load(f)

            creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            print(f"State from {creation_time.strftime('%d-%m-%Y %H:%M:%S')} loaded successfully")
            return state
        except Exception as e:
            print(f"Error loading pipeline state: {str(e)}")
            return None

    def save_state(self, state: PipelineState, save_as_json: bool = False, log_to_wandb: bool = False):
        try:
            from datetime import datetime
            import threading
            import wandb

            current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            new_folder_path = os.path.join(self.cache_dir, current_time)
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
        initial_state: PipelineState,
        pipeline_modes: PipelineModeSequence,
        start_from_step: str = None,
        stop_after_step: str = None
    ) -> List[Dict]:
        results = []

        for i, mode in enumerate(pipeline_modes.modes):
            print(f"Running combination {i + 1}/{len(pipeline_modes.modes)}")
            
            # Create a copy of the steps to avoid modifying the original pipeline
            temp_steps = [PipelineStep(step.name, step.function, step.mode_keys) for step in self.steps]
            
            # Apply mode-specific configurations
            for step in temp_steps:
                step_modes = {k: mode.modes[k] for k in step.mode_keys if k in mode.modes}
                original_function = step.function
                step.function = lambda state, orig_func=original_function, modes=step_modes: orig_func(state, **modes)

            # Create a temporary pipeline with the modified steps
            temp_pipeline = Pipeline(self.config)
            temp_pipeline.steps = temp_steps

            pipeline_result = await temp_pipeline.run(
                initial_state=initial_state,
                start_from_step=start_from_step,
                stop_after_step=stop_after_step,
            )

            results.append({
                "combination": i,
                "name": mode.name,
                "modes": mode.modes,
                "pipeline_result": pipeline_result
            })

        return results

    def print_mode_results_summary(self, results: List[Dict]):
        if not results:
            print("No results to summarize.")
            return

        headers = ["Combination", "Name"] + list(results[0]["modes"].keys()) + ["Num Images", "Num Objects"]
        summary_table = PrettyTable(headers)

        for result in results:
            row = [
                result["combination"],
                result["name"],
                *[result["modes"][key] for key in headers[2:-2]],
                len(result["pipeline_result"].images),
                sum(len(img.predicted_objects) for img in result["pipeline_result"].images)
            ]
            summary_table.add_row(row)

        print(summary_table)
