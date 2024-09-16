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
        batch_responses = []
        total_batches = (len(batch_requests) + batch_size - 1) // batch_size
        batch_time = 0

        for batch_start in tqdm(range(0, len(batch_requests), batch_size), total=total_batches, desc="OCR batches", unit="batch", unit_scale=True):
            batch_end = min(batch_start + batch_size, len(batch_requests))
            requests = batch_requests[batch_start:batch_end]
            batch_response = get_batch_prediction.sync(
                client=self.client,
                body=requests
            )
            batch_responses.extend(batch_response.responses)

            batch_time += batch_response.inference_time
            
        print(f"Batch time: {batch_time}")

        for image in state.images:
            for element in image.ui_elements:
                for response in batch_responses:

                    if image.id != response.id:
                        continue

                    # Skip if elements that are not named
                    if element.name is None:
                        continue
                        
                    element_name = element.name.lower()
                    # remove leading and trailing whitespace
                    element_name = element_name.strip()

                    # remove !,.?
                    element_name = element_name.replace("!", "").replace(".", "").replace(",", "").replace("?", "")

                    compared_labels = []
                    for (bbox, label) in zip(response.prediction.bboxes, response.prediction.labels):

                        label = label.lower()
                        label = label.strip()
                        label = label.replace("!", "").replace(".", "").replace(",", "").replace("?", "")

                        if element_name in label or label in element_name:
                            element.bbox = bbox
                            break
                        else:
                            compared_labels.append(label)
        return state

        