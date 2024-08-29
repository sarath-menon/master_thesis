#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, NamedTuple
from clicking.pipeline.core import Pipeline
from clicking.dataset_creator.core import CocoDataset
from clicking.dataset_creator.types import DatasetSample
from clicking.prompt_refinement.core import PromptRefiner, PromptMode, ProcessedResult, ProcessedSample
from clicking.vision_model.types import TaskType, LocalizationResults
from clicking.vision_model.visualization import show_localization_predictions, show_segmentation_prediction
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

#%%

client = Client(base_url="http://localhost:8082")

prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

coco_dataset = CocoDataset('./datasets/label_studio_gen/coco_dataset/images', './datasets/label_studio_gen/coco_dataset/result.json')
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
        self.steps: List[Union[Tuple[str, Callable, bool], List[Tuple[str, Callable, bool]]]] = []
        self.visualization_functions: Dict[Type, Callable] = {
            LocalizationResults: show_localization_predictions
        }
        self.cache_dir = ".pipeline_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_filename = self._generate_cache_filename()
        self.cache_data = {}

    def _generate_cache_filename(self):
        return os.path.join(self.cache_dir, f"pipeline_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

    def _save_cache(self, step_name: str, data: Any):
        self.cache_data[step_name] = data
        with open(self.cache_filename, 'wb') as f:
            pickle.dump(self.cache_data, f)

    def _load_cache(self, step_name: str) -> Any:
        if not os.path.exists(self.cache_filename):
            return None
        
        try:
            with open(self.cache_filename, 'rb') as f:
                self.cache_data = pickle.load(f)
            return self.cache_data.get(step_name)
        except (EOFError, pickle.UnpicklingError):
            print(f"Warning: Cache file is corrupted. Ignoring cache.")
            os.remove(self.cache_filename)
            self.cache_data = {}
        return None

    async def run(self, initial_input: Any) -> Any:
        result = initial_input
        for step in self.steps:
            if isinstance(step, list):  # Parallel steps
                tasks = []
                for step_name, step_func, verbose in step:
                    cache = self._load_cache(step_name)
                    if cache is not None:
                        print(f"Using cached input for step: {step_name}")
                        tasks.append(asyncio.create_task(asyncio.to_thread(lambda: cache)))
                    else:
                        tasks.append(asyncio.create_task(asyncio.to_thread(step_func, result)))
                
                parallel_results = await asyncio.gather(*tasks)
                
                for (step_name, _, verbose), step_result in zip(step, parallel_results):
                    self._save_cache(step_name, step_result)
                    if verbose:
                        self._log_step_result(step_name, step_result)
                
                result = parallel_results
            else:  # Sequential step
                step_name, step_func, verbose = step
                cache = self._load_cache(step_name)
                if cache is not None:
                    print(f"Using cached input for step: {step_name}")
                    result = cache
                else:
                    result = await asyncio.to_thread(step_func, result)
                    self._save_cache(step_name, result)
                
                if verbose:
                    self._log_step_result(step_name, result)
        
        return result

    async def run_from_step(self, start_step_name: str):
        start_index = next((i for i, step in enumerate(self.steps) if (isinstance(step, tuple) and step[0] == start_step_name) or (isinstance(step, list) and any(s[0] == start_step_name for s in step))), None)
        if start_index is None:
            raise ValueError(f"Step '{start_step_name}' not found in the pipeline")

        prev_step = self.steps[start_index - 1] if start_index > 0 else None
        prev_step_name = prev_step[0] if isinstance(prev_step, tuple) else prev_step[-1][0] if isinstance(prev_step, list) else None
        initial_input = self._load_cache(prev_step_name) if prev_step_name else None

        if initial_input is None:
            raise ValueError(f"No cached input found for step: {start_step_name}")

        result = initial_input
        for step in self.steps[start_index:]:
            if isinstance(step, list):  # Parallel steps
                tasks = []
                for step_name, step_func, verbose in step:
                    tasks.append(asyncio.create_task(asyncio.to_thread(step_func, result)))
                
                parallel_results = await asyncio.gather(*tasks)
                
                for (step_name, _, verbose), step_result in zip(step, parallel_results):
                    self._save_cache(step_name, step_result)
                    if verbose:
                        self._log_step_result(step_name, step_result)
                
                result = parallel_results
            else:  # Sequential step
                step_name, step_func, verbose = step
                result = await asyncio.to_thread(step_func, result)
                self._save_cache(step_name, result)
                
                if verbose:
                    self._log_step_result(step_name, result)
        
        return result

    def _are_types_compatible(self, prev_func: Union[Callable, List[Tuple[str, Callable, bool]], Tuple[str, Callable, bool]], next_func: Union[Callable, List[Tuple[str, Callable, bool]], Tuple[str, Callable, bool]]) -> bool:
        def get_return_type(func):
            if isinstance(func, tuple):
                return inspect.signature(func[1]).return_annotation
            return inspect.signature(func).return_annotation

        def get_param_type(func):
            if isinstance(func, tuple):
                return list(inspect.signature(func[1]).parameters.values())[0].annotation
            return list(inspect.signature(func).parameters.values())[0].annotation

        if isinstance(prev_func, list):
            prev_return_types = [get_return_type(step) for step in prev_func]
        else:
            prev_return_types = [get_return_type(prev_func)]
        
        if isinstance(next_func, list):
            next_param_types = [get_param_type(step) for step in next_func]
        else:
            next_param_types = [get_param_type(next_func)]
        
        for prev_return_type in prev_return_types:
            for next_param_type in next_param_types:
                if prev_return_type == Any or next_param_type == Any:
                    continue
                
                if get_origin(prev_return_type) is not None:
                    if not self._check_complex_type_compatibility(prev_return_type, next_param_type):
                        return False
                elif not issubclass(prev_return_type, next_param_type):
                    return False
        
        return True

    def _check_complex_type_compatibility(self, type1, type2):
        origin1, origin2 = get_origin(type1), get_origin(type2)
        args1, args2 = get_args(type1), get_args(type2)
        
        if origin1 is TypedDict and origin2 is TypedDict:
            # For TypedDict, check if all keys in type2 exist in type1
            return all(key in type1.__annotations__ for key in type2.__annotations__)
        
        if origin1 is origin2:
            # For other complex types (List, Dict, etc.), check their arguments
            return all(self._are_types_compatible(arg1, arg2) for arg1, arg2 in zip(args1, args2))
        
        return False

    def _log_step_result(self, step_name: str, result: Any):
        print(f"\n--- Step: {step_name} ---")
        self._recursive_log(step_name, result)
        
        # Check if there's a visualization function for this result type
        result_type = type(result)
        if result_type in self.visualization_functions:
            self.visualization_functions[result_type](result)

    def _recursive_log(self, step_name: str, result: Any, prefix: str = ""):
        keys_to_avoid = ["objects"]

        if isinstance(result, Image.Image):
            self._display_image(step_name, prefix, result)
            return

        if isinstance(result, list):
            self._log_list(step_name, result, prefix)
            return

        if isinstance(result, dict):
            self._log_dict(step_name, result, prefix, keys_to_avoid)
            return

        if isinstance(result, tuple):
            for item in result:
                self._recursive_log(step_name, item, prefix)
            return

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

    def _log_dict(self, step_name: str, result: Dict[str, Any], prefix: str, keys_to_avoid: List[str]):
        for key, value in result.items():
            if key in keys_to_avoid:
                self._recursive_log(step_name, value, prefix)
            else:
                print(f"{prefix}{key}: ", end="")
                self._recursive_log(step_name, value, "")

    def _display_image(self, step_name: str, prefix: str, image: Image.Image):
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{step_name} - Image")
        plt.show()

    def _display_image_list(self, step_name: str, prefix: str, images: List[Image.Image]):
        if not images:
            print(f"{prefix}[]")
            return
        
        fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
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
            
            if isinstance(current_step, list):  # Parallel steps
                current_return_types = [inspect.signature(step[1]).return_annotation for step in current_step]
            else:  # Sequential step
                current_return_types = [inspect.signature(current_step[1]).return_annotation]
            
            if isinstance(next_step, list):  # Parallel steps
                next_param_types = [list(inspect.signature(step[1]).parameters.values())[0].annotation for step in next_step]
            else:  # Sequential step
                next_param_types = [list(inspect.signature(next_step[1]).parameters.values())[0].annotation]
            
            for current_return_type in current_return_types:
                for next_param_type in next_param_types:
                    if current_return_type == Any or next_param_type == Any:
                        continue
                    
                    if not issubclass(current_return_type, next_param_type):
                        raise TypeError(f"Output type of {current_step[0] if isinstance(current_step, tuple) else 'parallel step'} ({current_return_type}) "
                                        f"is not compatible with input type of {next_step[0] if isinstance(next_step, tuple) else 'parallel step'} ({next_param_type})")
        
        print("Static analysis complete. All types are compatible.")

    def print_pipeline(self):
        headers = ["Step", "Function Name", "Input Type"]
        table_data = []
        
        for i, step in enumerate(self.steps, 1):
            if isinstance(step, list):  # Parallel steps
                for j, (step_name, func, _) in enumerate(step, 1):
                    input_type = list(inspect.signature(func).parameters.values())[0].annotation.__name__
                    table_data.append([f"{i}.{j} {step_name} (Parallel)", func.__name__, input_type])
            else:  # Sequential step
                step_name, func, _ = step
                input_type = list(inspect.signature(func).parameters.values())[0].annotation.__name__
                table_data.append([f"{i}. {step_name}", func.__name__, input_type])
        
        print("Pipeline Steps:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def add_step(self, step_name: str, func: Callable, verbose: bool = True):
        if self.steps and not self._are_types_compatible(self.steps[-1][1] if isinstance(self.steps[-1], tuple) else self.steps[-1], func):
            last_step_func = self.steps[-1][1] if isinstance(self.steps[-1], tuple) else self.steps[-1]
            last_step_output_type = inspect.signature(last_step_func).return_annotation.__name__
            next_step_input_type = inspect.signature(func).parameters[next(iter(inspect.signature(func).parameters))].annotation.__name__

            raise TypeError(f"Output type of {last_step_func.__name__} ({last_step_output_type}) is not compatible with input type of {func.__name__} ({next_step_input_type})")
        self.steps.append((step_name, func, verbose))

    def add_parallel_steps(self, *steps: Tuple[str, Callable, bool]):
        self.steps.append(list(steps))

    def replace_step(self, step_name: str, new_func: Callable, new_step_name: str = None, verbose: bool = True):
        for i, step in enumerate(self.steps):
            if isinstance(step, tuple) and step[0] == step_name:
                new_step_name = new_step_name or step_name
                self.steps[i] = (new_step_name, new_func, verbose)
                print(f"Step '{step_name}' replaced with '{new_step_name}'")
                return
            elif isinstance(step, list):
                for j, parallel_step in enumerate(step):
                    if parallel_step[0] == step_name:
                        new_step_name = new_step_name or step_name
                        step[j] = (new_step_name, new_func, verbose)
                        print(f"Parallel step '{step_name}' replaced with '{new_step_name}'")
                        return
        
        raise ValueError(f"Step '{step_name}' not found in the pipeline")

#%%
from clicking_client.types import File
from io import BytesIO
import json

# class SegmentationPrediction(NamedTuple):
#     masks: List[Dict[str, Any]]  

# class SegmentationResults(NamedTuple):
#     processed_samples: List[ProcessedSample]
#     localization_results: Dict[str, Dict[str, LocalizationResults]]
#     segmentation_results: Dict[str, Dict[str, SegmentationPrediction]]


class SegmentationResults(NamedTuple):
    processed_samples: List[ProcessedSample]
    predictions: Dict[str, List[SegmentationMask]] 

def image_to_http_file(image):
    # Convert PIL Image to bytes and create a File object
    image_byte_arr = BytesIO()
    image.save(image_byte_arr, format='JPEG')
    image_file = File(file_name="image.jpg", payload=image_byte_arr.getvalue(), mime_type="image/jpeg")
    return image_file

class LocalizationProcessor:
    def __init__(self, client: Client):
        self.client = client

    def get_localization_results(self, processed_result: ProcessedResult) -> LocalizationResults:
        set_model.sync(client=self.client, body=SetModelReq(name="florence2", variant="florence-2-base", task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB))
        
        all_predictions = {}
        for sample in processed_result.samples:
            image_file = image_to_http_file(sample.image)
            image_id = sample.image_id  # Assuming image_id is added to ProcessedSample

            all_predictions[image_id] = []
            for obj in sample.description["objects"]:
                request = BodyGetPrediction(
                    image=image_file,
                    task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
                    input_text=obj["description"]
                )
                response = get_prediction.sync(client=self.client, body=request)
                bboxes = [BoundingBox(bbox, mode=BBoxMode.XYWH, object_name=obj["name"], description=obj["description"]) 
                          for bbox in response.prediction.bboxes]
                all_predictions[image_id].extend(bboxes)
        
        return LocalizationResults(
            processed_samples=processed_result.samples,
            predictions=all_predictions
        )

class SegmentationProcessor:
    def __init__(self, client: Client):
        self.client = client

    def get_segmentation_results(self, data: LocalizationResults) -> SegmentationResults:
        set_model.sync(client=self.client, body=SetModelReq(name="sam2", variant="sam2_hiera_tiny", task=TaskType.SEGMENTATION_WITH_BBOX))
        
        segmentation_results = {}
        for sample in data.processed_samples:
            image_file = image_to_http_file(sample.image)
            image_id = sample.image_id
            seg_predictions = []
            for bbox in data.predictions[image_id]:
                request = BodyGetPrediction(
                    image=image_file,
                    task=TaskType.SEGMENTATION_WITH_BBOX,
                    input_boxes=json.dumps(bbox.get(mode=BBoxMode.XYWH))  # Ensure it's a JSON string
                )
                try:
                    response = get_prediction.sync(client=self.client, body=request)
                    if response is None or response.prediction is None:
                        print(f"Warning: No prediction received for image {image_id}, bbox {bbox}")
                        continue
                    
                    for mask_data in response.prediction.masks:
                        print(f"mask_data: {mask_data}")
                        seg_mask = SegmentationMask(
                            mask= mask_data,
                            mode=SegmentationMode.COCO_RLE,
                            object_name=bbox.object_name,
                            description=bbox.description
                        )
                        seg_predictions.append(seg_mask)
                except Exception as e:
                    print(f"Error processing segmentation for image {image_id}, bbox {bbox}: {str(e)}")
                    continue
            
            segmentation_results[image_id] = seg_predictions
        
        return SegmentationResults(
            processed_samples=data.processed_samples,
            predictions=segmentation_results
        )
#%%
import nest_asyncio
nest_asyncio.apply()

pipeline = Pipeline()
localization_processor = LocalizationProcessor(client)
segmentation_processor = SegmentationProcessor(client)

pipeline.add_step("Sample Dataset", coco_dataset.sample_dataset, verbose=True)
pipeline.add_step("Process Prompts", prompt_refiner.process_prompts, verbose=True)
pipeline.add_step("Get Localization Results", localization_processor.get_localization_results, verbose=True)
pipeline.add_step("Get Segmentation Results", segmentation_processor.get_segmentation_results, verbose=True)

# pipeline.add_parallel_steps(
#     ("Get Localization Results 1", localization_processor.get_localization_results, True),
#     ("Get Localization Results 2", localization_processor.get_localization_results, True),
# )

# Print the pipeline structure
pipeline.print_pipeline()

# Perform static analysis before running the pipeline
pipeline.static_analysis()

# Run the entire pipeline
image_ids = [22, 31, 34]
result = asyncio.run(pipeline.run(image_ids))

#%%

# You can also keep the same step name if you want
# pipeline.replace_step("Get Localization Results", new_localization_function)

# Later, run from a specific step
result = asyncio.run(pipeline.run_from_step("Get Segmentation Results"))
#%%
result
#%%
from clicking.vision_model.utils import get_mask_centroid
from clicking.vision_model.visualization import get_color
from clicking.vision_model.utils import get_mask_centroid
import cv2
import numpy as np
from pycocotools import mask as mask_utils

def show_clickpoint_predictions(segmentation_results: SegmentationResults, textbox_color='red', text_color='white', text_size=12, marker_size=100, marker_color='yellow'):
    for processed_sample in segmentation_results.processed_samples:
        image = processed_sample.image
        image_id = processed_sample.image_id
        predictions = segmentation_results.predictions[image_id]

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display the original image
        ax.imshow(image)

        borders = False
        mask_alpha = 0.7
        total_classes = len(set(mask.object_name for mask in predictions))

        for i, mask in enumerate(predictions):
            m = mask_utils.decode(mask.get(SegmentationMode.COCO_RLE))
            color_mask = get_color(i, total_classes)

            # Create color overlay with correct shape and alpha channel
            color_overlay = np.zeros((*np.array(image).shape[:2], 4))
            color_overlay[m == 1] = [*color_mask, mask_alpha] 
            color_overlay[m == 0] = [0, 0, 0, 0]  
            ax.imshow(color_overlay)

            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=2)
            
            # Get mask centroid and plot it as click point
            centroid = get_mask_centroid(m)
            ax.scatter(*centroid, marker='*', color=marker_color, s=marker_size, label=mask.object_name)  

            # Plot class label as text in a box on top of the mask
            offset_y = 60
            ax.text(centroid[0], centroid[1] - offset_y, mask.object_name, 
                    color=text_color, fontsize=text_size, 
                    bbox=dict(facecolor=textbox_color, edgecolor='none', alpha=1.0),
                    ha='center', va='center')

        ax.axis('off')
        plt.title(f"Image ID: {image_id}")
        plt.tight_layout()
        plt.show()

        # Print legend (object_name: description)
        print(f"Legend for Image ID: {image_id}")
        for mask in predictions:
            print(f"{mask.object_name}: {mask.description}")
        print("\n")

show_clickpoint_predictions(result)
#%%
