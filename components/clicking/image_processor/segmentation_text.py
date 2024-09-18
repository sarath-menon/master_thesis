from typing import Dict, List
from clicking_client import Client
from clicking_client.models import SetModelReq, PredictionReq
from clicking_client.api.default import set_model, get_prediction
from clicking.common.data_structures import PipelineState
from .utils import image_to_http_file
from clicking.common.data_structures import TaskType
from clicking.common.bbox import BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
import json
from clicking.common.data_structures import ValidityStatus
from clicking.vision_model.utils import pil_to_base64
from clicking_client.api.default import get_batch_prediction
from tqdm.asyncio import tqdm as async_tqdm
import asyncio

class SegmentationText:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.tasks = self.load_tasks()
        self.set_segmentation_model()

    def load_tasks(self) -> List[TaskType]:
        task_strings = self.config['models']['segmentation_with_text']['tasks']
        return [TaskType[task] for task in task_strings]

    def set_segmentation_model(self):
        try:
            for task in self.tasks:
                response = set_model.sync(client=self.client, body=SetModelReq(
                    name=self.config['models']['segmentation_with_text']['name'],
                    variant=self.config['models']['segmentation_with_text']['variant'],
                    task=task
                ))
                print(f"Set model for task {task}: {response}")
        except Exception as e:
            print(f"Error setting segmentation model: {str(e)}")

    async def get_segmentation_results_async(self, state: PipelineState, segmentation_mode: TaskType) -> PipelineState:
        batch_requests = []

        for clicking_image in state.images:
            image_base64 = pil_to_base64(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:
                if obj.validity.status is ValidityStatus.INVALID:
                    print(f"Skipping segmentation for {obj.name} because it is invalid: {obj.validity.status}")
                    continue   

                request = PredictionReq(
                    image=image_base64,
                    task=segmentation_mode,
                    input_text=obj.description,
                    reset_cache=True
                )
                batch_requests.append(request)
                request.id = str(obj.id)

        batch_size = 5
        responses = []
        total_batches = (len(batch_requests) + batch_size - 1) // batch_size
        batch_time = 0

        for batch_start in async_tqdm(range(0, len(batch_requests), batch_size), total=total_batches, desc="Segmenting image batches", unit="batch", unit_scale=True):
            batch_end = min(batch_start + batch_size, len(batch_requests))
            requests = batch_requests[batch_start:batch_end]
            batch_response = await get_batch_prediction.asyncio(
                client=self.client,
                body=requests
            )
            responses.extend(batch_response.responses)
            batch_time += batch_response.inference_time

        print(f"Batch time: {batch_time}")

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
        """
        Synchronous wrapper for get_segmentation_results_async.

        Args:
            state (PipelineState): The current pipeline state.
            segmentation_mode (TaskType): The segmentation mode to use.

        Returns:
            PipelineState: The updated pipeline state with segmentation results.
        """
        return asyncio.run(self.get_segmentation_results_async(state, segmentation_mode))


# for clicking_image in loaded_state_3.images:
#     image = clicking_image.image
#     for obj in clicking_image.predicted_objects:
#         obj.validity.status = ValidityStatus.UNKNOWN
#         obj.mask.denoise_mask()

#         extracted_area = obj.mask.extract_area(image)

#         if extracted_area.width >= image.width or extracted_area.height >= image.height:
#             obj.validity.status = ValidityStatus.INVALID
#             print(f"Invalid mask: {obj.name}")

#             print(f"Image size: {image.width,}")
#             print(f"Mask size: {extracted_area.size}")