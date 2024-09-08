from typing import Dict, List
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction
from clicking.common.data_structures import PipelineState
from .utils import image_to_http_file
from clicking.vision_model.data_structures import TaskType
from clicking.common.bbox import BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
import json
from clicking.common.data_structures import ValidityStatus

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

    def get_segmentation_results(self, state: PipelineState) -> PipelineState:
        for clicking_image in state.images:
            image_file = image_to_http_file(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:

                print(type(ValidityStatus.VALID), type(obj.validity.status))

                if obj.validity.status is not ValidityStatus.VALID:
                    print(f"Skipping segmentation for {obj.name} because it is invalid or not visible: {obj.validity.status}")
                    continue

                request = BodyGetPrediction(image=image_file)
                try:
                    response = get_prediction.sync(
                        client=self.client,
                        body=request,
                        task=TaskType.SEGMENTATION_WITH_BBOX,
                        input_boxes=json.dumps(obj.bbox.get(mode=BBoxMode.XYXY))
                    )
                    
                    # error handling
                    if len(response.prediction.masks) > 1:
                        print(f"Multiple masks found for {obj.name}: {len(response.prediction.masks)}. Ignoring.")
                    elif len(response.prediction.masks) == 0:
                        print(f"No segmentation mask found for {obj.name}")

                    obj.mask = SegmentationMask(coco_rle=response.prediction.masks[0], mode=SegmentationMode.COCO_RLE)

                except Exception as e:
                    print(f"Error processing segmentation for image {clicking_image.id}, object {obj.name}: {str(e)}")
        
        return state