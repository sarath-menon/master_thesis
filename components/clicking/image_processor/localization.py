
from typing import Dict
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction
from clicking.common.data_structures import *
from clicking.pipeline.core import PipelineState
from .utils import image_to_http_file
from clicking.vision_model.data_structures import TaskType
from clicking.common.bbox import BoundingBox, BBoxMode

class Localization:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.set_localization_model()

    def set_localization_model(self):
        try:
            response= set_model.sync(client=self.client, body=SetModelReq(
                name=self.config['models']['localization']['name'],
                variant=self.config['models']['localization']['variant'],
                task=TaskType[self.config['models']['localization']['task']]
            ))
            print(response)
        except Exception as e:
            print(f"Error setting localization model: {str(e)}")

    def get_localization_results(self, state: PipelineState) -> PipelineState:

        print("state.images", state)
        
        for clicking_image in state.images:
            image_file = image_to_http_file(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:
                request = BodyGetPrediction(image=image_file)
                try:
                    response = get_prediction.sync(
                        client=self.client,
                        body=request,
                        task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB,
                        input_text=obj.description,
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