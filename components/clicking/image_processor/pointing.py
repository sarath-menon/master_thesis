from typing import Dict, List
from clicking_client import Client
from clicking_client.models import PredictionReq
from clicking.common.data_structures import TaskType, ModuleMode, ValidityStatus, PipelineState, ObjectValidity
from enum import Enum
from clicking.vision_model.utils import pil_to_base64
from .base_processor import BaseProcessor

def process_description(description: str):
    return description

class PointingInput(Enum):
    OBJ_NAME = ModuleMode("obj_name", lambda obj: obj.name)
    OBJ_DESCRIPTION = ModuleMode("obj_description", lambda obj: process_description(obj.description))

class Pointing(BaseProcessor):
    def __init__(self, client: Client, config: Dict):
        super().__init__(client, config, 'pointing')
        self.pointing_input_mode = PointingInput.OBJ_NAME

    def create_prediction_request(self, image_base64: str, obj, mode: TaskType) -> PredictionReq:
        input_text = self.pointing_input_mode.value.handler(obj)
        input_text = f"Point to the {input_text.lower()}."
        print(input_text)
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

            print(response)
        return state

    def get_pointing_results(self, state: PipelineState, pointing_mode: TaskType, pointing_input_mode: PointingInput) -> PipelineState:
        self.pointing_input_mode = pointing_input_mode
        return self.process_results(state, pointing_mode)