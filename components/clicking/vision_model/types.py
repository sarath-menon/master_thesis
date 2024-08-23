from pydantic import BaseModel, ConfigDict
from typing import Any, Union, Optional
from enum import Enum, auto
from PIL import Image

class TaskType(str, Enum):
    LOCALIZATION_WITH_TEXT = "LOCALIZATION_WITH_TEXT"
    LOCALIZATION_WITH_TEXT_GROUNDED = "LOCALIZATION_WITH_TEXT_GROUNDED"
    LOCALIZATION_WITH_TEXT_OPEN_VOCAB = "LOCALIZATION_WITH_TEXT_OPEN_VOCAB"
    SEGMENTATION_WITH_TEXT = "SEGMENTATION_WITH_TEXT"
    SEGMENTATION_WITH_CLICKPOINT = "SEGMENTATION_WITH_CLICKPOINT"
    SEGMENTATION_WITH_BBOX = "SEGMENTATION_WITH_BBOX"
    SEGMENTATION_WITH_CLICKPOINT_AND_BBOX = "SEGMENTATION_WITH_CLICKPOINT_AND_BBOX"
    CAPTIONING = "CAPTIONING"
    SEGMENTATION_AUTO_ANNOTATION = "SEGMENTATION_AUTO_ANNOTATION"

class SegmentationReq(BaseModel):  
    image: Any
    input_boxes: list

class ModelInfo(BaseModel):
    name: str
    variants: list
    tasks: list

class GetModelsResp(BaseModel):
    models: list[ModelInfo]

class GetModelResp(BaseModel):
    name: str
    variant: str

class SetModelReq(BaseModel):
    name: str
    variant: str 
    task: TaskType

class GetModelReq(BaseModel):
    task: TaskType

class LocalizationReq(BaseModel):  
    image: str
    text_input: str
    task_prompt: str

class LocalizationResp(BaseModel):
    bboxes: list
    labels: list

class SegmentationResp(BaseModel):
    masks: list
    scores: Optional[list] = None

class PredictionReq(BaseModel):
    image: Image.Image
    task: TaskType
    input_point: Optional[list] = None
    input_label: Optional[list] = None
    input_box: Optional[list] = None
    input_text: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class PredictionResp(BaseModel):
    inference_time: Optional[float] = 0.0
    prediction: Union[LocalizationResp, SegmentationResp]

class AutoAnnotationReq(BaseModel):
    image: Image.Image
    task: TaskType
    min_mask_region_area: int 
    pred_iou_thresh: float 
    output_mode: str 

    model_config = ConfigDict(arbitrary_types_allowed=True)

class AutoAnnotationResp(BaseModel):
    prediction: SegmentationResp
    inference_time: Optional[float] = 0.0