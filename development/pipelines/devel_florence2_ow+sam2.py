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
from clicking.visualization.core import show_localization_predictions, show_segmentation_prediction
from clicking.visualization.bbox import BoundingBox, BBoxMode
from clicking.visualization.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction


class LocalizationPrediction(NamedTuple):
    bboxes: List[List[float]]

class SegmentationPrediction(NamedTuple):
    masks: List[Dict[str, Any]]  # Assuming masks are in COCO RLE format

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
                predictions[obj["name"]] = LocalizationPrediction(bboxes=response.prediction.bboxes)
            
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
                    input_boxes=loc_result.bboxes
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
