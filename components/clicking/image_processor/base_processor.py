from typing import Dict, List
from clicking_client import Client
from clicking_client.models import SetModelReq, PredictionReq
from clicking_client.api.default import set_model, get_batch_prediction
from clicking.common.data_structures import PipelineState, TaskType, ValidityStatus
from tqdm import tqdm
from clicking.vision_model.utils import pil_to_base64

class BaseProcessor:
    def __init__(self, client: Client, config: Dict, model_key: str):
        self.client = client
        self.config = config
        self.model_key = model_key
        self.tasks = self.load_tasks()
        self.set_model()

    def load_tasks(self) -> List[TaskType]:
        task_strings = self.config['models'][self.model_key]['tasks']
        return [TaskType[task] for task in task_strings]

    def set_model(self):
        try:
            for task in self.tasks:
                response = set_model.sync(client=self.client, body=SetModelReq(
                    name=self.config['models'][self.model_key]['name'],
                    variant=self.config['models'][self.model_key]['variant'],
                    task=task
                ))
                print(f"Set model for task {task}: {response}")
        except Exception as e:
            print(f"Error setting {self.model_key} model: {str(e)}")

    def process_results(self, state: PipelineState, mode: TaskType) -> PipelineState:
        batch_requests = self.prepare_batch_requests(state, mode)
        responses = self.get_batch_predictions(batch_requests)
        return self.process_responses(state, responses)

    def prepare_batch_requests(self, state: PipelineState, mode: TaskType) -> List[PredictionReq]:
        batch_requests = []
        for clicking_image in state.images:
            image_base64 = pil_to_base64(clicking_image.image)
            for obj in clicking_image.predicted_objects:
                if obj.validity.status is ValidityStatus.INVALID:
                    print(f"Skipping {self.model_key} for {obj.name} because it is invalid")
                    continue
                request = self.create_prediction_request(image_base64, obj, mode)
                batch_requests.append(request)
                request.id = str(obj.id)
        return batch_requests

    def create_prediction_request(self, image_base64: str, obj, mode: TaskType) -> PredictionReq:
        # This method should be implemented by subclasses
        raise NotImplementedError

    def get_batch_predictions(self, batch_requests: List[PredictionReq]) -> List:
        batch_size = 20
        responses = []
        total_batches = (len(batch_requests) + batch_size - 1) // batch_size
        batch_time = 0

        for batch_start in tqdm(range(0, len(batch_requests), batch_size), total=total_batches, desc=f"Processing {self.model_key} batches", unit="batch", unit_scale=True):
            batch_end = min(batch_start + batch_size, len(batch_requests))
            requests = batch_requests[batch_start:batch_end]
            batch_response = get_batch_prediction.sync(
                client=self.client,
                body=requests
            )
            responses.extend(batch_response.responses)
            batch_time += batch_response.inference_time

        print(f"Batch time: {batch_time}")
        return responses

    def process_responses(self, state: PipelineState, responses: List) -> PipelineState:
        # This method should be implemented by subclasses
        raise NotImplementedError