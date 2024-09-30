from typing import Dict, List, Union
from clicking_client import Client
from clicking_client.models import SetModelReq, PredictionReq, PredictionResp
from clicking_client.types import Response
from clicking.common.data_structures import PipelineState, TaskType, ValidityStatus
from tqdm import tqdm
from clicking.vision_model.utils import pil_to_base64
from clicking_client.api.default import set_model, get_batch_prediction, get_prediction
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_chunk(self, chunk):
        try:
            chunk_response = await get_batch_prediction.asyncio(
                client=self.client,
                body=chunk
            )
            return chunk_response.responses, chunk_response.inference_time
        except KeyError as e:
            logging.error(f"KeyError encountered: {e}")
            logging.error(f"Chunk data: {chunk}")
            raise

    async def get_batch_predictions_async(self, batch_requests: List[PredictionReq]) -> List:
        chunk_size = 50
        max_concurrent_chunks = 5
        semaphore = asyncio.Semaphore(max_concurrent_chunks)
        responses = []
        total_time = 0

        async def process_chunk_with_semaphore(chunk):
            async with semaphore:
                return await self.process_chunk(chunk)

        chunks = [batch_requests[i:i+chunk_size] for i in range(0, len(batch_requests), chunk_size)]
        total_chunks = len(chunks)

        async def process_all_chunks():
            nonlocal total_time
            tasks = [process_chunk_with_semaphore(chunk) for chunk in chunks]
            for completed in tqdm(asyncio.as_completed(tasks), total=total_chunks, desc=f"Processing {self.model_key} chunks", unit="chunk"):
                chunk_responses, chunk_time = await completed
                responses.extend(chunk_responses)
                total_time += chunk_time

        await process_all_chunks()

        print(f"Total inference time: {total_time:.2f} seconds")
        return responses

    def get_batch_predictions(self, batch_requests: List[PredictionReq]) -> List:
        return asyncio.run(self.get_batch_predictions_async(batch_requests))

    def process_responses(self, state: PipelineState, responses: List) -> PipelineState:
        # This method should be implemented by subclasses
        raise NotImplementedError

    def get_single_prediction(self, req: PredictionReq) -> Union[PredictionResp, Response]:
        return get_prediction.sync(
            client=self.client,
            body=req
        )

    def process_single_result(self, state: PipelineState, mode: TaskType, obj_id: str) -> PipelineState:
        clicking_image = state.images[0]
        obj = clicking_image.predicted_objects[0]

        image_base64 = pil_to_base64(clicking_image.image)
        request = self.create_prediction_request(image_base64, obj, mode)
        request.id = str(obj.id)

        response = self.get_single_prediction(request)

        if isinstance(response, PredictionResp):
            return self.process_single_response(state, response, obj.id)
        else:
            print(f"Error in prediction: {response.status_code}")
            return state

    def process_single_response(self, state: PipelineState, response, obj_id: str) -> PipelineState:
        # This method should be implemented by subclasses
        raise NotImplementedError