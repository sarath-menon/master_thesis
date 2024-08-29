#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, NamedTuple
from clicking.pipeline.core import Pipeline
from clicking.dataset_creator.core import CocoDataset
from clicking.dataset_creator.types import DatasetSample
from clicking.prompt_refinement.core import PromptRefiner, PromptMode, ProcessedResult, ProcessedSample
from clicking.vision_model.types import TaskType, LocalizationResp, SegmentationResp
from clicking.vision_model.visualization import show_localization_predictions, show_segmentation_prediction
from clicking.vision_model.bbox import BoundingBox, BBoxMode
from clicking.vision_model.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction


class LocalizationPrediction(NamedTuple):
    bboxes: List[BoundingBox]

class SegmentationPrediction(NamedTuple):
    masks: List[Dict[str, Any]]  

class LocalizationResults(NamedTuple):
    processed_samples: List[ProcessedSample]
    localization_results: Dict[str, Dict[str, LocalizationPrediction]]

class SegmentationResults(NamedTuple):
    processed_samples: List[ProcessedSample]
    localization_results: Dict[str, Dict[str, LocalizationPrediction]]
    segmentation_results: Dict[str, Dict[str, SegmentationPrediction]]

#%%

client = Client(base_url="http://localhost:8082")

prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

coco_dataset = CocoDataset('./datasets/label_studio_gen/coco_dataset/images', './datasets/label_studio_gen/coco_dataset/result.json')
#%%

from clicking_client.types import File
from io import BytesIO

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
        
        localization_results = {}
        for sample in processed_result.samples:
            image_file = image_to_http_file(sample.image)
            predictions = {}

            for obj in sample.description["objects"]:
                request = BodyGetPrediction(
                    image=image_file,
                    task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
                    input_text=obj["description"]
                )
                response = get_prediction.sync(client=self.client, body=request)
                bboxes = [BoundingBox(bbox, mode=BBoxMode.XYWH) for bbox in response.prediction.bboxes]
                predictions[obj["name"]] = LocalizationPrediction(bboxes=bboxes)
            
            image_filename = sample.image.filename if hasattr(sample.image, 'filename') else f"image_{id(sample.image)}"
            localization_results[image_filename] = predictions
        
        return LocalizationResults(
            processed_samples=processed_result.samples,
            localization_results=localization_results
        )

class SegmentationProcessor:
    def __init__(self, client: Client):
        self.client = client

    def get_segmentation_results(self, data: LocalizationResults) -> SegmentationResults:
        set_model.sync(client=self.client, body=SetModelReq(name="sam2", variant="sam2_hiera_tiny", task=TaskType.SEGMENTATION_WITH_BBOX))
        
        segmentation_results = {}
        for sample in data.processed_samples:
            image_file = image_to_http_file(sample.image)
            image_filename = sample.image.filename if hasattr(sample.image, 'filename') else f"image_{id(sample.image)}"
            seg_predictions = {}
            for obj_name, loc_result in data.localization_results[image_filename].items():
                request = BodyGetPrediction(
                    image=image_file,
                    task=TaskType.SEGMENTATION_WITH_BBOX,
                    input_boxes=[bbox.to_xywh() for bbox in loc_result.bboxes]
                )
                response = get_prediction.sync(client=self.client, body=request)
                seg_predictions[obj_name] = SegmentationPrediction(masks=response.prediction.masks)
            segmentation_results[image_filename] = seg_predictions
        
        return SegmentationResults(
            processed_samples=data.processed_samples,
            localization_results=data.localization_results,
            segmentation_results=segmentation_results
        )

#%%

from typing import Callable, List, Any, Dict, Tuple, TypedDict, get_origin, get_args
import inspect
import matplotlib.pyplot as plt
from PIL import Image

class Pipeline:
    def __init__(self):
        self.steps: List[Tuple[Callable, bool]] = []

    def add_step(self, func: Callable, verbose: bool = True):
        if self.steps and not self._are_types_compatible(self.steps[-1][0], func):
            last_step_func = self.steps[-1][0]
            last_step_output_type = inspect.signature(last_step_func).return_annotation.__name__
            next_step_input_type = inspect.signature(func).parameters[next(iter(inspect.signature(func).parameters))].annotation.__name__

            raise TypeError(f"Output type of {last_step_func.__name__} ({last_step_output_type}) is not compatible with input type of {func.__name__} ({next_step_input_type})")
        self.steps.append((func, verbose))

    def _are_types_compatible(self, prev_func: Callable, next_func: Callable) -> bool:
        prev_return_type = inspect.signature(prev_func).return_annotation
        next_param_types = [param.annotation for param in inspect.signature(next_func).parameters.values()]
        
        if not next_param_types:
            return True
        
        if prev_return_type == Any or next_param_types[0] == Any:
            return True
        
        # Handle TypedDict and other complex types
        if get_origin(prev_return_type) is not None:
            return self._check_complex_type_compatibility(prev_return_type, next_param_types[0])
        
        # For simple types, use isinstance check instead of issubclass
        return isinstance(prev_return_type, type(next_param_types[0]))

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

    def run(self, initial_input: Any) -> Any:
        result = initial_input
        for step, verbose in self.steps:
            result = step(result)
            if verbose:
                self._log_step_result(step.__name__, result)
        return result

    def _log_step_result(self, step_name: str, result: Any):
        print(f"\n--- Step: {step_name} ---")
        self._recursive_log(step_name, result)

    def _recursive_log(self, step_name: str, result: Any, prefix: str = "", is_list_item: bool = False):
        keys_to_avoid = ["objects"]  # Add more keys to this list if needed

        if isinstance(result, Image.Image):
            self._display_image(step_name, prefix, result)
        elif isinstance(result, list):
            if not result:
                print(f"{prefix}Empty list")
            elif all(isinstance(item, Image.Image) for item in result):
                self._display_image_list(step_name, prefix, result)
            else:
                for i, item in enumerate(result):
                    if i > 0:
                        print(f"{prefix}---")
                    self._recursive_log(step_name, item, prefix, True)
        elif isinstance(result, dict):
            for key, value in result.items():
                if key in keys_to_avoid:
                    self._recursive_log(step_name, value, prefix, is_list_item)
                else:
                    print(f"{prefix}{key}: ", end="")
                    self._recursive_log(step_name, value, "", is_list_item)
        elif isinstance(result, tuple):
            for item in result:
                self._recursive_log(step_name, item, prefix, is_list_item)
        else:
            print(f"{prefix}{result}")

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
            current_step, _ = self.steps[i]
            next_step, _ = self.steps[i + 1]
            
            current_return_type = inspect.signature(current_step).return_annotation
            next_param_types = list(inspect.signature(next_step).parameters.values())
            
            if not next_param_types:
                continue
            
            next_param_type = next_param_types[0].annotation
            
            if current_return_type == Any or next_param_type == Any:
                continue
            
            if not issubclass(current_return_type, next_param_type):
                raise TypeError(f"Output type of {current_step.__name__} ({current_return_type}) "
                                f"is not compatible with input type of {next_step.__name__} ({next_param_type})")
        
        print("Static analysis complete. All types are compatible.")



#%%
import nest_asyncio
nest_asyncio.apply()

def main():
    pipeline = Pipeline()
    pipeline.add_step(coco_dataset.sample_dataset, verbose=True)
    pipeline.add_step(prompt_refiner.process_prompts, verbose=True)
    
    localization_processor = LocalizationProcessor(client)
    pipeline.add_step(localization_processor.get_localization_results, verbose=True)
    
    # segmentation_processor = SegmentationProcessor(client)
    # pipeline.add_step(segmentation_processor.get_segmentation_results, verbose=True)
    
    # Perform static analysis before running the pipeline
    pipeline.static_analysis()

    image_ids = [22, 31, 34]
    result = pipeline.run(image_ids)
    print("\nFinal result:")
    print(result)

if __name__ == "__main__":
    main()

# %%
