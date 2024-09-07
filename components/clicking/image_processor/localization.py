from typing import Dict, List
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction
from clicking.common.data_structures import *
from clicking.pipeline.core import PipelineState
from .utils import image_to_http_file
from clicking.vision_model.data_structures import TaskType
from clicking.common.bbox import BoundingBox, BBoxMode
from enum import Enum

class InputMode(Enum):
    OBJ_NAME = ModuleMode("obj_name", lambda obj: obj.name)
    OBJ_DESCRIPTION = ModuleMode("obj_description", lambda obj: obj.description)

class Localization:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.tasks = self.load_tasks()
        self.set_localization_model()

    def load_tasks(self) -> List[TaskType]:
        task_strings = self.config['models']['localization']['tasks']
        return [TaskType[task] for task in task_strings]

    def set_localization_model(self):
        try:
            for task in self.tasks:
                response = set_model.sync(client=self.client, body=SetModelReq(
                    name=self.config['models']['localization']['name'],
                    variant=self.config['models']['localization']['variant'],
                    task=task
                ))
                print(f"Set model for task {task}: {response}")
        except Exception as e:
            print(f"Error setting localization model: {str(e)}")

    def get_localization_results(self, state: PipelineState, localization_mode: TaskType = TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB, localization_input_mode: InputMode = InputMode.OBJ_DESCRIPTION) -> PipelineState:
        
        for clicking_image in state.images:
            image_file = image_to_http_file(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:
                request = BodyGetPrediction(image=image_file)
                
                # Use the lambda function associated with the input_mode
                input_text = localization_input_mode.value.get_input(obj)

                try:
                    response = get_prediction.sync(
                        client=self.client,
                        body=request,
                        task=localization_mode,
                        input_text=input_text,
                        reset_cache=True
                    )

                    if len(response.prediction.bboxes) > 1:
                        print(f"Multiple bounding boxes found for {obj.name}")
                        obj.bbox = BoundingBox(bbox=response.prediction.bboxes[0], mode=BBoxMode.XYXY)
                    elif len(response.prediction.bboxes) == 1:
                        obj.bbox = BoundingBox(bbox=response.prediction.bboxes[0], mode=BBoxMode.XYXY)
                    else:
                        print(f"No bounding box found for {obj.name}")

                except Exception as e:
                    print(f"Error getting prediction for image {clicking_image.id}, object {obj.name}: {str(e)}")
        
        return state