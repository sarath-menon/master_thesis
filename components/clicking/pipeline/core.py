from typing import Callable, List, Any, Dict, Tuple, TypedDict, get_origin, get_args
import inspect
import matplotlib.pyplot as plt
from PIL import Image

from typing import Type
from tabulate import tabulate
from dataclasses import dataclass
from datetime import datetime
import os
import pickle
from typing import Union
from dataclasses import dataclass, field
from clicking.common.data_structures import ClickingImage
import asyncio

@dataclass
class PipelineState:
    images: List[ClickingImage] = field(default_factory=list)

class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        self.steps: List[Tuple[str, Callable]] = []
        self.config = config
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

    async def run(self, initial_input: Union[List[int], PipelineState], start_from_step: str = None, stop_after_step: str = None) -> PipelineState:
        start_index = 0
        if start_from_step:
            start_index = self._find_step_index(start_from_step)
            if start_index == -1:
                raise ValueError(f"Invalid start_from_step: '{start_from_step}'. Step not found in the pipeline.")

        if stop_after_step and self._find_step_index(stop_after_step) == -1:
            raise ValueError(f"Invalid stop_after_step: '{stop_after_step}'. Step not found in the pipeline.")

        # print(f"Running pipeline from step: {self.steps[start_index][start_index]} to step: {stop_after_step}")
        
        if start_index > 0:
            self._load_cache()
            if start_from_step not in self.cache_data:
                raise ValueError(f"No cached state found for step '{start_from_step}'. Please provide an initial state or run from the beginning.")
            initial_input = self.cache_data[start_from_step]
        
        self.cache_data = {}  # Reset cache for a new run
        return await self._run_internal(initial_input, start_index=start_index, stop_after_step=stop_after_step)

    async def _run_internal(self, initial_input: Union[List[int], PipelineState], start_index: int = 0, stop_after_step: str = None) -> PipelineState:
        state = initial_input if isinstance(initial_input, PipelineState) else PipelineState(images=initial_input)
        
        for i, (step_name, step_func) in enumerate(self.steps[start_index:], start=start_index):
            if step_name in self.cache_data:
                print(f"Using cached input for step: {step_name}")
                state = self.cache_data[step_name]
            
            state = await asyncio.to_thread(step_func, state)
            
            if i < len(self.steps) - 1:
                self.cache_data[self.steps[i+1][0]] = state
                self._save_cache()
            
            if stop_after_step and step_name == stop_after_step:
                print(f"Stopping execution after step: {step_name}")
                break
        
        return state

    def get_step_result(self, step_name: str) -> Any:
        self._load_cache()
        return self.cache_data.get(step_name)

    def _find_step_index(self, step_name: str) -> int:
        for i, (name, _) in enumerate(self.steps):
            if name == step_name:
                return i
        return -1

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
        
        for i, (step_name, func) in enumerate(self.steps, 1):
            input_type = list(inspect.signature(func).parameters.values())[0].annotation.__name__
            table_data.append([f"{i}. {step_name}", func.__name__, input_type])
        
        print("Pipeline Steps:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def add_step(self, step_name: str, func: Callable):
        if self.steps and not self._are_types_compatible(self.steps[-1][1], func):
            last_step_func = self.steps[-1][1]
            last_step_output_type = inspect.signature(last_step_func).return_annotation.__name__
            next_step_input_type = list(inspect.signature(func).parameters.values())[0].annotation.__name__

            raise TypeError(f"Output type of {last_step_func.__name__} ({last_step_output_type}) is not compatible with input type of {func.__name__} ({next_step_input_type})")
        self.steps.append((step_name, func))

    def _are_types_compatible(self, prev_func: Callable, next_func: Callable) -> bool:
        prev_return_type = inspect.signature(prev_func).return_annotation
        next_param_type = list(inspect.signature(next_func).parameters.values())[0].annotation
        
        if prev_return_type == Any or next_param_type == Any:
            return True
        
        return issubclass(prev_return_type, next_param_type)

    def replace_step(self, step_name: str, new_func: Callable):
        step_index = self._find_step_index(step_name)
        if step_index == -1:
            raise ValueError(f"Step '{step_name}' not found in the pipeline.")

        prev_step = self.steps[step_index - 1][1] if step_index > 0 else None
        next_step = self.steps[step_index + 1][1] if step_index < len(self.steps) - 1 else None

        if prev_step and not self._are_types_compatible(prev_step, new_func):
            raise TypeError(f"Output type of previous step is not compatible with input type of new function.")
        
        if next_step and not self._are_types_compatible(new_func, next_step):
            raise TypeError(f"Output type of new function is not compatible with input type of next step.")

        self.steps[step_index] = (step_name, new_func)
        print(f"Step '{step_name}' has been replaced successfully.")

