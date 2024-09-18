from typing import Dict, List
from clicking_client import Client
from clicking_client.models import PredictionReq
from clicking.common.data_structures import PipelineState, TaskType, ValidityStatus
from clicking.common.bbox import BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
import json
from clicking.vision_model.utils import pil_to_base64
from .base_processor import BaseProcessor

class Segmentation(BaseProcessor):
    def __init__(self, client: Client, config: Dict):
        super().__init__(client, config, 'segmentation')

    def create_prediction_request(self, image_base64: str, obj, mode: TaskType) -> PredictionReq:
        return PredictionReq(
            image=image_base64,
            task=mode,
            input_boxes=json.dumps(obj.bbox.get(mode=BBoxMode.XYXY)),
            reset_cache=True
        )

    def process_responses(self, state: PipelineState, responses: List) -> PipelineState:
        for response in responses:
            obj = state.find_object_by_id(response.id)
            if not obj:
                continue

            try:
                if not response.prediction or not response.prediction.masks:
                    print(f"No segmentation mask found for {obj.name}")
                    continue

                if len(response.prediction.masks) > 1:
                    print(f"Multiple masks found for {obj.name}: {len(response.prediction.masks)}. Using the first one.")

                obj.mask = SegmentationMask(coco_rle=response.prediction.masks[0], mode=SegmentationMode.COCO_RLE)

            except Exception as e:
                print(f"Error processing segmentation for object {obj.name}: {str(e)}")

        return state

    def get_segmentation_results(self, state: PipelineState, segmentation_mode: TaskType) -> PipelineState:
        return self.process_results(state, segmentation_mode)