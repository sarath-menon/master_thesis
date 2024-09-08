from typing import Dict
from clicking_client import Client
from clicking_client.models import SetModelReq, BodyGetPrediction
from clicking_client.api.default import set_model, get_prediction
from clicking.common.data_structures import *
from clicking.common.data_structures import PipelineState
from .utils import image_to_http_file
from clicking.vision_model.data_structures import TaskType
from clicking.common.bbox import BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
import json

class Segmentation:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.set_segmentation_model()

    def set_segmentation_model(self):
        try:
            response= set_model.sync(client=self.client, body=SetModelReq(
                name=self.config['models']['segmentation']['name'],
                variant=self.config['models']['segmentation']['variant'],
                task=TaskType[self.config['models']['segmentation']['task']]
            ))
            print(response)
        except Exception as e:
            print(f"Error setting segmentation model: {str(e)}")

    def get_segmentation_results(self, state: PipelineState) -> PipelineState:
        
        for clicking_image in state.images:
            image_file = image_to_http_file(clicking_image.image)
            
            for obj in clicking_image.predicted_objects:
                request = BodyGetPrediction(image=image_file)
                try:
                    response = get_prediction.sync(
                        client=self.client,
                        body=request,
                        task=TaskType.SEGMENTATION_WITH_BBOX,
                        input_boxes=json.dumps(obj.bbox.get(mode=BBoxMode.XYWH))
                    )
                    
                    if response.prediction.masks:
                        obj.mask = SegmentationMask(coco_rle=response.prediction.masks[0], mode=SegmentationMode.COCO_RLE)
                    else:
                        print(f"No segmentation mask found for {obj.name}")
                except Exception as e:
                    print(f"Error processing segmentation for image {clicking_image.id}, object {obj.name}: {str(e)}")
        
        return state