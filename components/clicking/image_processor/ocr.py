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


class OCR:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.tasks = self.load_tasks()
        self.set_ocr_model()

    def load_tasks(self) -> List[TaskType]:
        task_strings = self.config['models']['ocr']['tasks']
        return [TaskType[task] for task in task_strings]

    def set_ocr_model(self):
        try:
            for task in self.tasks:
                response = set_model.sync(client=self.client, body=SetModelReq(
                    name=self.config['models']['ocr']['name'],
                    variant=self.config['models']['ocr']['variant'],
                    task=task
                ))
                print(f"Set model for task {task}: {response}")
        except Exception as e:
            print(f"Error setting ocr model: {str(e)}")

    def get_ocr_results(self, state: PipelineState) -> PipelineState:

        batch_requests = []

        for clicking_image in state.images:
            image_base64 = pil_to_base64(clicking_image.image)
            # if obj.validity.status is ValidityStatus.INVALID:
            #     print(f"Skipping ocr for {obj.name} because it is invalid")
            #     continue
            
            request = PredictionReq(
                image=image_base64,
                task=TaskType.OCR,
                reset_cache=True
            )

            batch_requests.append(request)
            request.id = str(clicking_image.id)

        batch_size = 20
        responses = []
        total_batches = (len(batch_requests) + batch_size - 1) // batch_size
        batch_time = 0

        for batch_start in tqdm(range(0, len(batch_requests), batch_size), total=total_batches, desc="OCR batches", unit="batch", unit_scale=True):
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
            print(f"Response: {response}")
            for image in state.images:
                if image.id != response.id:
                    continue
                for element in image.ui_elements:
                    print(f"Element, {element.name}, label, {response.prediction.labels[0]}")    
                    if element.name == response.prediction.labels[0]:
                        print('Matched', element.name, response.prediction.labels[0])
                    


            # try:
            #     if not response.prediction or not response.prediction.quad_box:
            #         print(f"No quad box found for {obj.name}")
            #         obj.validity = ObjectValidity(status=ValidityStatus.INVALID, reason="No bounding box found")
            #         continue

            #     if len(response.prediction.quad_box) == 1:
            #         obj.bbox = BoundingBox(bbox=response.prediction.quad_box[0], mode=BBoxMode.XYXY)

            #     else:
            #         # If multiple bounding boxes are found, use the largest one
            #         bboxes = [BoundingBox(bbox=bbox, mode=BBoxMode.XYXY) for bbox in response.prediction.quad_box]
            #         obj.bbox = max(bboxes, key=lambda bbox: bbox.get_area())
            #         print(f"Multiple bounding boxes found for {obj.name}: {len(response.prediction.quad_box)}. Using the largest one.")


            # except Exception as e:
            #     print(f"Error processing ocr for object {obj.name}: {str(e)}")
        # return state

        return responses