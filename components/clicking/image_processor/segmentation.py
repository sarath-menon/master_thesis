from typing import Dict, List
from clicking_client import Client
from clicking_client.models import SetModelReq, PredictionReq
from clicking_client.api.default import set_model, get_prediction, get_batch_prediction
from clicking.common.data_structures import PipelineState
from .utils import image_to_http_file
from clicking.common.data_structures import TaskType
from clicking.common.bbox import BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
import json
from clicking.common.data_structures import ValidityStatus
from tqdm import tqdm
from clicking.vision_model.utils import pil_to_base64

class Segmentation:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.tasks = self.load_tasks()
        self.set_segmentation_model()

    def load_tasks(self) -> List[TaskType]:
        task_strings = self.config['models']['segmentation']['tasks']
        return [TaskType[task] for task in task_strings]

    def set_segmentation_model(self):
        try:
            for task in self.tasks:
                response = set_model.sync(client=self.client, body=SetModelReq(
                    name=self.config['models']['segmentation']['name'],
                    variant=self.config['models']['segmentation']['variant'],
                    task=task
                ))
                print(f"Set model for task {task}: {response}")
        except Exception as e:
            print(f"Error setting segmentation model: {str(e)}")

    def get_segmentation_results(self, state: PipelineState, segmentation_mode: TaskType) -> PipelineState:
        batch_requests = []

        for clicking_image in state.images:
            image_base64 = pil_to_base64(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:
                if obj.validity.status is ValidityStatus.INVALID:
                    print(f"Skipping segmentation for {obj.name} because it is invalid")
                    continue
                

                request = PredictionReq(
                    image=image_base64,
                    task=segmentation_mode,
                    input_boxes=json.dumps(obj.bbox.get(mode=BBoxMode.XYXY)),
                    reset_cache=True
                )
                batch_requests.append(request)
                request.id = str(obj.id)

        batch_size = 20
        responses = []
        total_batches = (len(batch_requests) + batch_size - 1) // batch_size
        batch_time = 0

        for batch_start in tqdm(range(0, len(batch_requests), batch_size), total=total_batches, desc="Segmenting image batches", unit="batch", unit_scale=True):
            batch_end = min(batch_start + batch_size, len(batch_requests))
            requests = batch_requests[batch_start:batch_end]
            batch_response = get_batch_prediction.sync(
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