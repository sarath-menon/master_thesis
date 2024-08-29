#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from clicking.pipeline.core import Pipeline, pipeline_step
from clicking.dataset_creator.core import CocoDataset
from clicking.dataset_creator.types import DatasetSample
from clicking.prompt_refinement.core import PromptRefiner, PromptMode
from clicking.vision_model.types import TaskType, LocalizationResp, SegmentationResp
from clicking.visualization.core import show_localization_predictions, show_segmentation_prediction
from clicking.visualization.bbox import BoundingBox, BBoxMode
from clicking.visualization.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
from clicking.common.image_utils import image_to_http_file
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction

#%%

client = Client(base_url="http://localhost:8082")
prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")
coco_dataset = CocoDataset('./datasets/label_studio_gen/coco_dataset/images', './datasets/label_studio_gen/coco_dataset/result.json')

#%%

@pipeline_step
def get_localization_results(data: Dict[str, Any]) -> Dict[str, Any]:
    set_model.sync(client=client, body=SetModelReq(name="florence2", variant="florence-2-base", task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB))
    
    localization_results = {}
    for image, description in zip(data["images"], data["descriptions"]):
        image_file = image_to_http_file(image)
        predictions = {}
        for obj in description["objects"]:
            request = BodyGetPrediction(
                image=image_file,
                task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
                input_text=obj["description"]
            )
            predictions[obj["name"]] = get_prediction.sync(client=client, body=request)
        localization_results[image] = predictions
    
    data["localization_results"] = localization_results
    return data

@pipeline_step
def get_segmentation_results(data: Dict[str, Any]) -> Dict[str, Any]:
    set_model.sync(client=client, body=SetModelReq(name="sam2", variant="sam2_hiera_tiny", task=TaskType.SEGMENTATION_WITH_BBOX))
    
    segmentation_results = {}
    for sample in data["samples"]:
        image_file = image_to_http_file(sample.image)
        seg_predictions = {}
        for obj_name, loc_result in data["localization_results"][sample.image].items():
            request = BodyGetPrediction(
                image=image_file,
                task=TaskType.SEGMENTATION_WITH_BBOX,
                input_boxes=loc_result.prediction.bboxes
            )
            seg_predictions[obj_name] = get_prediction.sync(client=client, body=request)
        segmentation_results[sample.image] = seg_predictions
    
    data["segmentation_results"] = segmentation_results
    return data

@pipeline_step
def visualize_results(data: Dict[str, Any]) -> None:
    for sample in data["samples"]:
        loc_results = data["localization_results"][sample.image]
        show_localization_predictions(sample.image, loc_results)
    
    for sample in data["samples"]:
        seg_results = data["segmentation_results"][sample.image]
        for obj_name, seg_result in seg_results.items():
            masks = [SegmentationMask(mask, mode=SegmentationMode.COCO_RLE) for mask in seg_result.prediction.masks]
            show_segmentation_prediction(sample.image, masks)

#%%
import nest_asyncio
nest_asyncio.apply()

def main():
    pipeline = Pipeline()
    pipeline.add_step(coco_dataset.sample_dataset, verbose=True)
    pipeline.add_step(prompt_refiner.process_prompts, verbose=True)
    # pipeline.add_step(get_localization_results, verbose=True)
    # pipeline.add_step(get_segmentation_results, verbose=True)
    # pipeline.add_step(visualize_results, verbose=True)

    # Perform static analysis before running the pipeline
    pipeline.static_analysis()

    image_ids = [22, 31, 34]
    result = pipeline.run(image_ids)
    print("\nFinal result:")
    print(result)

if __name__ == "__main__":
    main()

# %%
