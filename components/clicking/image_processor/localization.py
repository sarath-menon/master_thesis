from typing import Dict, List
from clicking_client import Client
from clicking_client.models import PredictionReq
from clicking.common.data_structures import TaskType, ModuleMode, ValidityStatus, PipelineState, ObjectValidity
from clicking.common.bbox import BoundingBox, BBoxMode
from enum import Enum
from clicking.vision_model.utils import pil_to_base64
from .base_processor import BaseProcessor

def process_description(description: str):
    return description

class LocalizerInput(Enum):
    OBJ_NAME = ModuleMode("obj_name", lambda obj: obj.name)
    OBJ_DESCRIPTION = ModuleMode("obj_description", lambda obj: process_description(obj.description))

class Localization(BaseProcessor):
    def __init__(self, client: Client, config: Dict):
        super().__init__(client, config, 'localization')
        self.localization_input_mode = LocalizerInput.OBJ_NAME

    def create_prediction_request(self, image_base64: str, obj, mode: TaskType) -> PredictionReq:
        input_text = self.localization_input_mode.value.handler(obj)
        return PredictionReq(
            image=image_base64,
            task=mode,
            input_text=input_text,
            reset_cache=True
        )

    def process_responses(self, state: PipelineState, responses: List) -> PipelineState:
        for response in responses:
            obj = state.find_object_by_id(response.id)
            if not obj:
                print(f"Object {response.id} not found in state")
                continue

            obj.bbox = None

            try:
                if not response.prediction or not response.prediction.bboxes:
                    print(f"No bounding box found for {obj.name}")
                    obj.validity = ObjectValidity(status=ValidityStatus.INVALID, reason="No bounding box found")
                    continue

                if len(response.prediction.bboxes) == 1:
                    obj.bbox = BoundingBox(bbox=response.prediction.bboxes[0], mode=BBoxMode.XYXY)
                else:
                    bboxes = [BoundingBox(bbox=bbox, mode=BBoxMode.XYXY) for bbox in response.prediction.bboxes]
                    obj.bbox = max(bboxes, key=lambda bbox: bbox.get_area())
                    print(f"Multiple bounding boxes found for {obj.name}: {len(response.prediction.bboxes)}. Using the largest one.")

            except Exception as e:
                print(f"Error processing localization for object {obj.name}: {str(e)}")
        return state

    def get_localization_results(self, state: PipelineState, localization_mode: TaskType, localization_input_mode: LocalizerInput) -> PipelineState:
        self.localization_input_mode = localization_input_mode
        return self.process_results(state, localization_mode)