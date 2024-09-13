from typing import Dict, List
from clicking_client import Client
from tqdm import tqdm
from clicking_client.api.default import set_model, get_prediction, get_batch_prediction
from .utils import image_to_http_file
from clicking.common.data_structures import TaskType, ModuleMode, ValidityStatus, PipelineState, ObjectValidity
from clicking_client.models import SetModelReq, PredictionReq
from clicking.common.bbox import BoundingBox, BBoxMode
from enum import Enum
from clicking.vision_model.utils import pil_to_base64

def process_description(description: str):
    return description

class LocalizerInput(Enum):
    OBJ_NAME = ModuleMode("obj_name", lambda obj: obj.name)
    OBJ_DESCRIPTION = ModuleMode("obj_description", lambda obj: process_description(obj.description))

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

    def get_localization_results(self, state: PipelineState, localization_mode: TaskType, localization_input_mode: LocalizerInput) -> PipelineState:

        batch_requests = []

        for clicking_image in state.images:
            image_base64 = pil_to_base64(clicking_image.image)
            for obj in clicking_image.predicted_objects:

                # if obj.validity.status is ValidityStatus.INVALID:
                #     print(f"Skipping localization for {obj.name} because it is invalid")
                #     continue
                
                # Use the lambda function associated with the input_mode
                input_text = localization_input_mode.value.handler(obj)

                request = PredictionReq(
                    image=image_base64,
                    task=localization_mode,
                    input_text=input_text,
                    reset_cache=True
                )

                batch_requests.append(request)
                request.id = str(obj.id)

        batch_size = 20
        responses = []
        total_batches = (len(batch_requests) + batch_size - 1) // batch_size
        batch_time = 0

        for batch_start in tqdm(range(0, len(batch_requests), batch_size), total=total_batches, desc="Localizing image batches", unit="batch", unit_scale=True):
            batch_end = min(batch_start + batch_size, len(batch_requests))
            requests = batch_requests[batch_start:batch_end]
            batch_response = get_batch_prediction.sync(
                client=self.client,
                body=requests
            )
            responses.extend(batch_response.responses)

            batch_time += batch_response.inference_time
            
        print(f"Batch time: {batch_time}")
        print(f"Responses: {len(responses)}")


        for response in responses:
            obj = state.find_object_by_id(response.id)
            if not obj:
                print(f"Object {response.id} not found in state")
                continue

            # clear any existing bbox
            obj.bbox = None

            try:
                if not response.prediction or not response.prediction.bboxes:
                    print(f"No bounding box found for {obj.name}")
                    obj.validity = ObjectValidity(status=ValidityStatus.INVALID, reason="No bounding box found")
                    continue

                if len(response.prediction.bboxes) == 1:
                    obj.bbox = BoundingBox(bbox=response.prediction.bboxes[0], mode=BBoxMode.XYXY)

                else:
                    # If multiple bounding boxes are found, use the largest one
                    bboxes = [BoundingBox(bbox=bbox, mode=BBoxMode.XYXY) for bbox in response.prediction.bboxes]
                    obj.bbox = max(bboxes, key=lambda bbox: bbox.get_area())
                    print(f"Multiple bounding boxes found for {obj.name}: {len(response.prediction.bboxes)}. Using the largest one.")


            except Exception as e:
                print(f"Error processing localization for object {obj.name}: {str(e)}")
        return state